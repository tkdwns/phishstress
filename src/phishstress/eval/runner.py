"""판별기를 슬라이스에 돌려 지표를 뽑는다.

`Detector` 계약만 알면 되므로 더미든 학습 모델이든 동일하게 평가된다.
Day 5에 KLUE-RoBERTa가 들어와도 이 파일은 바뀌지 않는다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..data.loaders import Example
from ..detectors.base import Detector
from .metrics import MetricSet, compute, robustness_gap


@dataclass
class SliceResult:
    slice_name: str
    detector: str
    metrics: MetricSet
    latency_ms_mean: float
    latency_ms_p95: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "slice": self.slice_name,
            "detector": self.detector,
            "latency_ms": {
                "mean": round(self.latency_ms_mean, 3),
                "p95": round(self.latency_ms_p95, 3),
            },
            **self.metrics.to_dict(),
        }


@dataclass
class EvalReport:
    detector: str
    results: list[SliceResult] = field(default_factory=list)

    def by_slice(self, name: str) -> SliceResult | None:
        return next((r for r in self.results if r.slice_name == name), None)

    def gap(self, clean: str, stressed: str, metric: str = "recall_at_fpr01") -> float:
        a, b = self.by_slice(clean), self.by_slice(stressed)
        if a is None or b is None:
            return float("nan")
        return robustness_gap(a.metrics, b.metrics, metric)

    def to_dict(self) -> dict[str, Any]:
        return {"detector": self.detector, "slices": [r.to_dict() for r in self.results]}


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = min(int(round((len(s) - 1) * q)), len(s) - 1)
    return s[k]


def evaluate_slice(
    detector: Detector, examples: list[Example], slice_name: str, threshold: float = 0.5
) -> SliceResult:
    if not examples:
        raise ValueError(f"슬라이스 '{slice_name}'가 비어 있습니다.")
    if detector.modality != "text":
        raise ValueError(f"{detector.name}은(는) 텍스트 판별기가 아닙니다.")

    scores: list[float] = []
    labels: list[int] = []
    latencies: list[float] = []

    for ex in examples:
        t0 = time.perf_counter()
        res = detector.predict(ex.text)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        scores.append(res.score)
        labels.append(ex.label)

    return SliceResult(
        slice_name=slice_name,
        detector=detector.name,
        metrics=compute(labels, scores, threshold=threshold),
        latency_ms_mean=sum(latencies) / len(latencies),
        latency_ms_p95=_percentile(latencies, 0.95),
    )


def evaluate(
    detector: Detector, slices: dict[str, list[Example]], threshold: float = 0.5
) -> EvalReport:
    report = EvalReport(detector=detector.name)
    for name, examples in slices.items():
        if not examples:
            continue
        report.results.append(evaluate_slice(detector, examples, name, threshold))
    return report
