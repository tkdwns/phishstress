"""KO-VP-Stress 변환 계약.

이 프로젝트의 최상위 지표는 정확도가 아니라 **Robustness Gap**이다.
Gap을 재려면 "깨끗한 조건"과 짝을 이루는 "열화 조건"을 코드로 만들어야 하고,
그 코드가 이 패키지다.

원본 데이터는 배포하지 않고 **변환 스크립트만 배포**한다(docs/ETHICS.md).
누구나 원본을 받아 같은 스트레스 슬라이스를 재현할 수 있다.

## 축 구성

| 축 | 무엇을 흔드는가 | 라벨 |
|---|---|---|
| A | 사기범이 문구를 바꾼다 (동의어·우회·표기교란·수치표기·화계·어순) | 유지 |
| C | STT가 전사를 망가뜨린다 (삭제·발음혼동·조사손실·숫자오인식) | 유지 |
| D | 배포 환경의 어려운 음성 샘플 (진짜 금융 대화) | **바뀐다** |

축 A와 C는 **라벨 보존 변환**이다. 의미가 유지되므로 정답이 그대로여야 하고,
성능이 떨어지면 그건 모델의 취약점이다.

축 D는 다르다. Day 3 감사에서 공개 데이터의 정상 클래스가 은행 통화가 아니라
여행·반려동물 잡담임을 확인했다(일상 어휘 포함률 정상 94.9% vs 피싱 14.5%).
그래서 축 D는 **새 라벨을 가진 샘플을 만든다** — 금융 어휘를 담고 있지만 사기가
아닌 통화. 모델이 주제가 아니라 행위를 보는지 시험한다.
"""

from __future__ import annotations

import hashlib
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..data.loaders import Example

Axis = str  # "A" | "C" | "D"


@dataclass(frozen=True)
class TransformMeta:
    axis: Axis
    key: str
    name: str
    label_preserving: bool
    description: str
    # 이 변환이 **어떤 라벨의 샘플에 적용되는가**. 위협 모델을 코드에 박아둔다.
    #   축 A(적대적 문구) → (1,)   사기범이 자기 대본을 고친다. 정상 발신자는 그러지 않는다.
    #   축 C(전사 오류)   → (0, 1) STT 오류는 통화 종류를 가리지 않는다.
    #   D1 → (0,)  정상 통화에 금융 문맥을 넣는다
    #   D2 → (1,)  피싱 전사문을 무해화한다
    applies_to: tuple[int, ...] = (0, 1)
    # 적용 대상이 아닌 샘플을 어떻게 할 것인가.
    #   False = 원본 그대로 통과 (축 A/C — 슬라이스가 두 클래스를 유지해야 한다)
    #   True  = 슬라이스에서 제외 (축 D — 새 라벨 집합을 구성하는 중이다)
    drop_non_applicable: bool = False


class Transform(ABC):
    """변환 하나.

    구현체는 `meta`를 정의하고 `_apply`를 구현한다.

    **결정성이 요구사항이다.** 같은 (변환, 샘플, seed)는 항상 같은 결과를 내야 한다.
    그래야 어제의 Gap과 오늘의 Gap을 비교할 수 있다. 전역 난수를 쓰지 않고
    샘플 uid에서 시드를 유도하는 이유다.
    """

    meta: TransformMeta

    def __init__(self, seed: int = 20260816, strength: float = 1.0) -> None:
        if not 0.0 <= strength <= 1.0:
            raise ValueError("strength는 [0,1] 범위여야 합니다.")
        self.seed = seed
        self.strength = strength

    # ---- 내부 ----------------------------------------------------------

    def _rng(self, uid: str) -> random.Random:
        digest = hashlib.sha256(f"{self.seed}:{self.meta.key}:{uid}".encode()).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))

    @abstractmethod
    def _apply(self, ex: Example, rng: random.Random) -> Example | None:
        """변환을 적용한다. 적용 불가면 None을 낸다(그 샘플은 슬라이스에서 빠진다)."""

    # ---- 공개 진입점 ----------------------------------------------------

    def apply(self, ex: Example, rng: random.Random | None = None) -> Example | None:
        if ex.label not in self.meta.applies_to:
            return None if self.meta.drop_non_applicable else ex
        out = self._apply(ex, rng or self._rng(ex.uid))
        if out is None:
            return None
        if not out.text.strip():
            return None  # 빈 텍스트를 내는 변환은 무효로 본다
        if self.meta.label_preserving and out.label != ex.label:
            raise AssertionError(
                f"{self.meta.key}는 라벨 보존 변환인데 "
                f"라벨이 {ex.label}→{out.label}로 바뀌었습니다."
            )
        return out

    def apply_all(self, examples: list[Example]) -> list[Example]:
        out = []
        for ex in examples:
            got = self.apply(ex)
            if got is not None:
                out.append(got)
        return out

    def describe(self) -> dict:
        return {
            "axis": self.meta.axis,
            "key": self.meta.key,
            "name": self.meta.name,
            "label_preserving": self.meta.label_preserving,
            "applies_to": list(self.meta.applies_to),
            "strength": self.strength,
            "description": self.meta.description,
        }


@dataclass
class StressSlice:
    """변환으로 만들어진 평가 슬라이스."""

    key: str
    name: str
    axis: Axis
    examples: list[Example] = field(default_factory=list)
    coverage: float = 1.0  # 원본 대비 남은 비율 (변환 실패분 제외)
    change_rate: float = 1.0  # 텍스트가 실제로 바뀐 비율
    # 축 D는 변환 결과가 전부 음성이라 단일 클래스가 된다. 그대로 두면 AUC가
    # 정의되지 않으므로 원본 양성을 섞어 평가 가능한 슬라이스로 만든다.
    # 이 값은 그때 섞어 넣은 양성 개수다(0이면 혼합하지 않음).
    mixed_base_positives: int = 0

    def __len__(self) -> int:
        return len(self.examples)

    def to_dict(self) -> dict:
        pos = sum(e.label for e in self.examples)
        return {
            "key": self.key,
            "name": self.name,
            "axis": self.axis,
            "n": len(self.examples),
            "positive": pos,
            "negative": len(self.examples) - pos,
            "coverage": round(self.coverage, 4),
            "change_rate": round(self.change_rate, 4),
            "mixed_base_positives": self.mixed_base_positives,
        }


def build_slice(transform: Transform, examples: list[Example]) -> StressSlice:
    """변환을 적용하고 커버리지·변경률을 함께 기록한다.

    변경률을 재는 이유: 변환이 조용히 아무것도 안 바꾸면 Gap이 0으로 나오는데,
    그건 "모델이 강건하다"가 아니라 "시험을 안 했다"이다. 둘을 구분해야 한다.
    """
    out: list[Example] = []
    changed = 0
    for ex in examples:
        got = transform.apply(ex)
        if got is None:
            continue
        out.append(got)
        if got.text != ex.text:
            changed += 1
    n = len(examples)
    return StressSlice(
        key=transform.meta.key,
        name=transform.meta.name,
        axis=transform.meta.axis,
        examples=out,
        coverage=len(out) / n if n else 0.0,
        change_rate=changed / len(out) if out else 0.0,
    )
