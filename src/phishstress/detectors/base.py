"""판별기 플러그인 계약.

이 프로젝트의 핵심 설계 결정: 판별기는 게이트웨이 뒤의 **교체 가능한 플러그인**이다.
아래 `Detector` 계약만 지키면 TF-IDF든 RoBERTa든 AASIST든 동일하게 꽂힌다.

이렇게 설계한 이유:
1. 모델 학습이 늦어져도 서빙 파이프라인은 더미 판별기로 먼저 완성·검증할 수 있다.
2. 강건성 회귀 CI가 판별기 구현과 무관하게 동일한 인터페이스로 스트레스 스위트를 돌린다.
3. `confidence` 필드가 정책 레이어의 '열화 인지 가중치 감쇠'를 가능하게 한다 —
   입력이 열화됐다고 판별기 스스로 신고하면 정책이 그 발언권을 줄인다.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

Modality = Literal["audio", "text"]


@dataclass(frozen=True)
class AudioChunk:
    """판별기에 전달되는 오디오 윈도우.

    samples: float32, 범위 [-1.0, 1.0], 모노
    """

    samples: np.ndarray
    sample_rate: int
    start_sec: float = 0.0

    def __post_init__(self) -> None:
        if self.samples.ndim != 1:
            raise ValueError(f"모노 1차원 배열이어야 합니다. got ndim={self.samples.ndim}")
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate는 양수여야 합니다. got {self.sample_rate}")

    @property
    def duration_sec(self) -> float:
        return len(self.samples) / self.sample_rate


@dataclass(frozen=True)
class DetectionResult:
    """판별기 1회 추론 결과.

    score:      위험 확률 [0, 1]. 1에 가까울수록 사기/합성음성.
    confidence: 이 판단을 얼마나 신뢰하는가 [0, 1].
                입력이 열화됐다고 판단되면 판별기가 스스로 낮춘다.
                정책 레이어가 이 값으로 가중치를 감쇠한다.
    latency_ms: 추론 소요 시간.
    detail:     판별기별 부가 정보(근거, 열화 추정치 등). 직렬화 가능해야 한다.
    """

    score: float
    confidence: float = 1.0
    latency_ms: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score는 [0,1] 범위여야 합니다. got {self.score}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence는 [0,1] 범위여야 합니다. got {self.confidence}")


class Detector(ABC):
    """모든 판별기가 구현해야 하는 계약.

    구현체는 `name`, `modality`를 클래스 속성으로 선언한다.
    """

    name: str = "unnamed"
    modality: Modality = "text"

    @abstractmethod
    def _predict(self, x: AudioChunk | str) -> DetectionResult:
        """실제 추론. 구현체가 오버라이드한다. latency_ms는 채우지 않아도 된다."""

    def predict(self, x: AudioChunk | str) -> DetectionResult:
        """입력 검증 + 지연 측정을 감싼 공개 진입점."""
        self._validate_input(x)
        t0 = time.perf_counter()
        result = self._predict(x)
        elapsed = (time.perf_counter() - t0) * 1000.0
        # 구현체가 latency를 채우지 않았으면 측정값으로 대체
        if result.latency_ms == 0.0:
            result = DetectionResult(
                score=result.score,
                confidence=result.confidence,
                latency_ms=elapsed,
                detail=result.detail,
            )
        return result

    def _validate_input(self, x: AudioChunk | str) -> None:
        if self.modality == "audio" and not isinstance(x, AudioChunk):
            raise TypeError(f"{self.name}은(는) AudioChunk를 받습니다. got {type(x).__name__}")
        if self.modality == "text" and not isinstance(x, str):
            raise TypeError(f"{self.name}은(는) str을 받습니다. got {type(x).__name__}")

    def warmup(self) -> None:
        """콜드 스타트 완화용. 무거운 모델은 여기서 더미 추론 1회를 돌린다."""
        return None

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "modality": self.modality, "class": type(self).__name__}


class DetectorRegistry:
    """이름으로 판별기를 등록/조회한다. 게이트웨이가 기동 시 채운다."""

    def __init__(self) -> None:
        self._items: dict[str, Detector] = {}

    def register(self, detector: Detector) -> Detector:
        if detector.name in self._items:
            raise ValueError(f"이미 등록된 판별기 이름입니다: {detector.name}")
        self._items[detector.name] = detector
        return detector

    def get(self, name: str) -> Detector:
        if name not in self._items:
            raise KeyError(f"등록되지 않은 판별기: {name}")
        return self._items[name]

    def by_modality(self, modality: Modality) -> list[Detector]:
        return [d for d in self._items.values() if d.modality == modality]

    def all(self) -> list[Detector]:
        return list(self._items.values())

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items
