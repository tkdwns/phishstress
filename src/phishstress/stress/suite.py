"""KO-VP-Stress 스위트 — 축을 모아 평가 슬라이스를 만든다."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..data.loaders import Example
from .axis_a import AXIS_A
from .axis_c import AXIS_C
from .axis_d import AXIS_D
from .base import StressSlice, Transform, build_slice

ALL_TRANSFORMS: tuple[type[Transform], ...] = AXIS_A + AXIS_C + AXIS_D


def build_transforms(
    seed: int = 20260816, axes: tuple[str, ...] = ("A", "C", "D")
) -> list[Transform]:
    out = [cls(seed=seed) for cls in ALL_TRANSFORMS]
    return [t for t in out if t.meta.axis in axes]


def get_transform(key: str, seed: int = 20260816) -> Transform:
    for cls in ALL_TRANSFORMS:
        if cls.meta.key == key:
            return cls(seed=seed)
    raise KeyError(f"알 수 없는 변환: {key}. 사용 가능: {[c.meta.key for c in ALL_TRANSFORMS]}")


@dataclass
class StressSuite:
    """원본 슬라이스 하나에 모든 변환을 적용한 결과."""

    base_name: str
    base: list[Example]
    slices: list[StressSlice] = field(default_factory=list)

    def as_eval_slices(self) -> dict[str, list[Example]]:
        """`eval.runner.evaluate`가 받는 형태로 변환한다."""
        out = {self.base_name: self.base}
        for s in self.slices:
            if s.examples:
                out[f"{self.base_name}/{s.key}"] = s.examples
        return out

    def by_axis(self, axis: str) -> list[StressSlice]:
        return [s for s in self.slices if s.axis == axis]

    def to_dict(self) -> dict:
        return {
            "base": self.base_name,
            "base_n": len(self.base),
            "slices": [s.to_dict() for s in self.slices],
        }


def build_suite(
    examples: list[Example],
    base_name: str = "test",
    seed: int = 20260816,
    axes: tuple[str, ...] = ("A", "C", "D"),
    min_slice_size: int = 20,
) -> StressSuite:
    """모든 변환을 적용해 스트레스 슬라이스를 만든다.

    min_slice_size 미만으로 줄어든 슬라이스는 버린다 — 표본이 너무 작으면
    지표가 잡음이고, 그런 수치를 Gap이라고 보고하면 오히려 해롭다.
    """
    suite = StressSuite(base_name=base_name, base=examples)
    base_positives = [e for e in examples if e.label == 1]

    for t in build_transforms(seed=seed, axes=axes):
        s = build_slice(t, examples)

        # 축 D의 변환 결과는 전부 음성(하드 네거티브)이다. 단일 클래스에서는
        # PR-AUC/ROC-AUC가 정의되지 않으므로 원본 양성을 섞어 준다.
        # 이렇게 만들어진 슬라이스가 이 프로젝트에서 가장 어려운 테스트셋이다 —
        # 금융 어휘를 담은 정상 통화와 진짜 피싱을 같은 자리에 놓고 묻는다.
        needs_positives = (
            t.meta.axis == "D"
            and s.examples
            and base_positives
            and all(e.label == 0 for e in s.examples)
        )
        if needs_positives:
            s.examples = s.examples + base_positives
            s.mixed_base_positives = len(base_positives)

        if len(s) >= min_slice_size:
            suite.slices.append(s)
    return suite
