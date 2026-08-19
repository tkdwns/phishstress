"""자명한 베이스라인 — 전부 `Detector` 계약으로 구현한다.

두 가지 목적이 있다.

1. **평가 코드 검증.** 학습 모델이 없는 Day 3에 지표 구현이 맞는지 확인하려면
   정답을 아는 예측기가 필요하다. Oracle은 PR-AUC 1.0, Random은 양성비율,
   Constant는 다수클래스 정확도가 나와야 한다. 안 나오면 지표 코드가 틀린 것이다.

2. **성능 수치의 기준선.** `LengthDetector`가 이 프로젝트에서 가장 중요한 베이스라인이다.
   KorCCVi v2에서 글자 수만 세도 정확도 96.3%가 나온다. 앞으로 어떤 모델을 만들든
   이 숫자를 넘지 못하면 의미가 없고, 넘더라도 얼마나 넘었는지를 같이 보고해야 한다.

전부 Detector를 구현하므로 서빙 게이트웨이에 그대로 꽂아볼 수도 있다.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable

from ..detectors.base import AudioChunk, DetectionResult, Detector


class ConstantScoreDetector(Detector):
    """항상 같은 점수. score=0.0이면 '전부 정상으로 찍기' 전략이 된다."""

    modality = "text"

    def __init__(self, score: float = 0.0, name: str = "baseline-constant") -> None:
        self.name = name
        self._score = float(score)

    def _predict(self, x: AudioChunk | str) -> DetectionResult:
        return DetectionResult(score=self._score, confidence=1.0, detail={"baseline": "constant"})


class RandomDetector(Detector):
    """입력 해시 기반 결정적 의사난수. seed가 같으면 항상 같은 결과가 나온다."""

    modality = "text"

    def __init__(self, seed: int = 0, name: str = "baseline-random") -> None:
        self.name = name
        self.seed = seed

    def _predict(self, x: AudioChunk | str) -> DetectionResult:
        payload = f"{self.seed}:{x}".encode()
        digest = hashlib.sha256(payload).digest()
        score = int.from_bytes(digest[:8], "big") / 2**64
        return DetectionResult(score=score, confidence=0.5, detail={"baseline": "random"})


class OracleDetector(Detector):
    """정답을 아는 예측기. **평가 코드 검증 전용이며 성능 보고에 쓰면 안 된다.**

    noise를 주면 완벽하지 않은 예측기를 흉내 낼 수 있어 지표의 중간 구간도 확인된다.
    """

    modality = "text"
    name = "baseline-oracle"

    def __init__(self, labels: dict[str, int], noise: float = 0.0) -> None:
        if not 0.0 <= noise <= 1.0:
            raise ValueError("noise는 [0,1] 범위여야 합니다.")
        self._labels = labels
        self.noise = noise

    def _predict(self, x: AudioChunk | str) -> DetectionResult:
        label = self._labels.get(str(x))
        if label is None:
            return DetectionResult(score=0.5, confidence=0.0, detail={"baseline": "oracle_miss"})
        base = float(label)
        if self.noise > 0.0:
            jitter = int.from_bytes(hashlib.sha256(str(x).encode()).digest()[:8], "big") / 2**64
            base = (1.0 - self.noise) * base + self.noise * jitter
        return DetectionResult(
            score=min(max(base, 0.0), 1.0), confidence=1.0, detail={"baseline": "oracle"}
        )


class LengthDetector(Detector):
    """전사문 **글자 수만** 보는 판별기. 이 프로젝트에서 가장 불편한 베이스라인이다.

    KorCCVi v2는 두 클래스가 서로 다른 출처에서 수집되어 길이 분포가 갈라져 있다.
    정상 통화는 최소 1,740자인데 피싱 전사문의 중앙값은 534자다. 그래서 글자 수를
    세는 것만으로 정확도 96.3%, F1 91.9%가 나온다 — 같은 데이터에서 F1 99.31%를
    보고한 선행 연구(Mathematics 11(14), 3217)와 7.4점 차이밖에 나지 않는다.

    `direction`: -1이면 '짧을수록 피싱'(KorCCVi v2), +1이면 '길수록 피싱'(KorCCViD v1.3).
    데이터셋마다 아티팩트 방향이 반대라 `fit()`이 학습 데이터에서 자동으로 정한다.
    """

    modality = "text"

    def __init__(
        self,
        pivot: float = 2000.0,
        scale: float = 500.0,
        direction: int = -1,
        name: str = "baseline-length",
    ) -> None:
        self.name = name
        self.pivot = float(pivot)
        self.scale = max(float(scale), 1e-6)
        self.direction = -1 if direction < 0 else 1

    def fit(self, texts: Iterable[str], labels: Iterable[int]) -> LengthDetector:
        """학습 데이터에서 pivot/scale/direction을 정한다. 테스트 데이터를 보면 안 된다."""
        lens_pos = [len(t) for t, y in zip(texts, labels, strict=False) if y == 1]
        texts = list(texts) if not isinstance(texts, list) else texts
        lens_all = [len(t) for t in texts]
        lens_neg = [n for n, y in zip(lens_all, labels, strict=False) if y == 0]
        if not lens_pos or not lens_neg:
            return self
        mp = sorted(lens_pos)[len(lens_pos) // 2]
        mn = sorted(lens_neg)[len(lens_neg) // 2]
        self.pivot = (mp + mn) / 2.0
        self.scale = max(abs(mn - mp) / 4.0, 1.0)
        self.direction = -1 if mp < mn else 1
        return self

    def _predict(self, x: AudioChunk | str) -> DetectionResult:
        n = len(str(x))
        z = self.direction * (n - self.pivot) / self.scale
        score = 1.0 / (1.0 + math.exp(-max(min(z, 60.0), -60.0)))
        return DetectionResult(
            score=score,
            confidence=1.0,
            detail={"baseline": "length", "length": n, "pivot": self.pivot},
        )
