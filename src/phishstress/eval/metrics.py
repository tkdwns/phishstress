"""판별 성능 지표.

scikit-learn을 core 의존성으로 넣지 않고 numpy로 직접 구현했다. 이유는 둘이다.
1. 서빙 컨테이너는 `.[redis]`만 설치한다. 평가용 무거운 패키지가 들어갈 자리가 아니다.
2. 직접 구현한 지표는 테스트에서 scikit-learn과 교차 검증할 수 있다 —
   dev 의존성으로만 sklearn을 두고 `tests/test_metrics.py`가 두 구현을 대조한다.

이 프로젝트가 Accuracy를 주지표로 쓰지 않는 이유는 `trivial_baselines`가 보여준다.
KorCCVi v2에서는 전부 정상으로 찍기만 해도 76.3%, **글자 수만 세도 96.3%** 가 나온다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

ArrayLike = np.ndarray | list


def _as_arrays(y_true: ArrayLike, y_score: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=np.int64).ravel()
    s = np.asarray(y_score, dtype=np.float64).ravel()
    if y.shape != s.shape:
        raise ValueError(f"길이가 다릅니다: y_true={y.shape}, y_score={s.shape}")
    if y.size == 0:
        raise ValueError("빈 입력입니다.")
    bad = set(np.unique(y).tolist()) - {0, 1}
    if bad:
        raise ValueError(f"y_true는 0/1만 허용합니다. 발견: {sorted(bad)}")
    if not np.isfinite(s).all():
        raise ValueError("y_score에 NaN 또는 inf가 있습니다.")
    return y, s


def roc_curve(y_true: ArrayLike, y_score: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """(fpr, tpr)을 반환한다. 동점 점수는 하나의 임계값으로 묶는다."""
    y, s = _as_arrays(y_true, y_score)
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    # 동점 경계에서만 절단
    distinct = np.where(np.diff(s))[0]
    idx = np.r_[distinct, y.size - 1]
    tp = np.cumsum(y)[idx]
    fp = np.cumsum(1 - y)[idx]
    P, N = y.sum(), y.size - y.sum()
    tpr = np.r_[0.0, tp / P] if P else np.zeros(idx.size + 1)
    fpr = np.r_[0.0, fp / N] if N else np.zeros(idx.size + 1)
    return fpr, tpr


def roc_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    y, _ = _as_arrays(y_true, y_score)
    if y.sum() in (0, y.size):
        return float("nan")  # 한 클래스만 있으면 정의되지 않는다
    fpr, tpr = roc_curve(y_true, y_score)
    return float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))


def average_precision(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """PR-AUC. scikit-learn과 동일한 계단식(step) 정의를 쓴다 — 사다리꼴 적분은
    PR 곡선에서 성능을 낙관적으로 부풀린다."""
    y, s = _as_arrays(y_true, y_score)
    P = y.sum()
    if P == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    distinct = np.where(np.diff(s))[0]
    idx = np.r_[distinct, y.size - 1]
    tp = np.cumsum(y)[idx]
    fp = np.cumsum(1 - y)[idx]
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / P
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def recall_at_fpr(y_true: ArrayLike, y_score: ArrayLike, max_fpr: float = 0.01) -> float:
    """ "정상 통화 100건 중 1건만 오탐하는 조건에서 사기를 몇 % 잡나."

    운영 가능성의 실질 척도라 이 프로젝트의 주력 지표로 쓴다."""
    if not 0 < max_fpr <= 1:
        raise ValueError("max_fpr은 (0,1] 범위여야 합니다.")
    y, _ = _as_arrays(y_true, y_score)
    if y.sum() == 0 or y.sum() == y.size:
        return float("nan")
    fpr, tpr = roc_curve(y_true, y_score)
    ok = fpr <= max_fpr + 1e-12
    return float(tpr[ok].max()) if ok.any() else 0.0


def equal_error_rate(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """EER — 음성 판별기의 표준 지표. |FPR - FNR|이 최소인 지점."""
    y, _ = _as_arrays(y_true, y_score)
    if y.sum() == 0 or y.sum() == y.size:
        return float("nan")
    fpr, tpr = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    i = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2)


def accuracy_at(y_true: ArrayLike, y_score: ArrayLike, threshold: float = 0.5) -> float:
    y, s = _as_arrays(y_true, y_score)
    return float(((s >= threshold).astype(np.int64) == y).mean())


def f1_at(y_true: ArrayLike, y_score: ArrayLike, threshold: float = 0.5) -> float:
    y, s = _as_arrays(y_true, y_score)
    pred = (s >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    if tp == 0:
        return 0.0
    prec, rec = tp / (tp + fp), tp / (tp + fn)
    return float(2 * prec * rec / (prec + rec))


@dataclass(frozen=True)
class MetricSet:
    """한 슬라이스에 대한 지표 묶음."""

    n: int
    positives: int
    positive_rate: float
    pr_auc: float
    roc_auc: float
    recall_at_fpr01: float
    eer: float
    accuracy: float
    f1: float
    majority_accuracy: float

    def to_dict(self) -> dict:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def compute(y_true: ArrayLike, y_score: ArrayLike, threshold: float = 0.5) -> MetricSet:
    y, _ = _as_arrays(y_true, y_score)
    pos = int(y.sum())
    rate = pos / y.size
    return MetricSet(
        n=int(y.size),
        positives=pos,
        positive_rate=round(rate, 6),
        pr_auc=average_precision(y, y_score),
        roc_auc=roc_auc(y, y_score),
        recall_at_fpr01=recall_at_fpr(y, y_score, 0.01),
        eer=equal_error_rate(y, y_score),
        accuracy=accuracy_at(y, y_score, threshold),
        f1=f1_at(y, y_score, threshold),
        majority_accuracy=max(rate, 1 - rate),
    )


def robustness_gap(clean: MetricSet, stressed: MetricSet, metric: str = "recall_at_fpr01") -> float:
    """이 프로젝트의 최상위 지표.

    Gap = Metric(깨끗한 조건) − Metric(열화 조건).
    선행 연구 기준선: KoBERT가 적대적 조건에서 정확도 98.52% → 56.25%로 42.27%p 붕괴
    (arXiv 2506.06180). 우리 목표는 15%p 이내.
    """
    a, b = getattr(clean, metric), getattr(stressed, metric)
    if a != a or b != b:  # NaN
        return float("nan")
    return float(a - b)


def bootstrap_ci(
    y_true: ArrayLike,
    y_score: ArrayLike,
    metric_fn=average_precision,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 20260816,
) -> tuple[float, float]:
    """부트스트랩 신뢰구간.

    길이정합 슬라이스처럼 표본이 작을 때(수십 건) 점추정만 보고하면 잡음을 실력으로
    착각하게 된다. 작은 슬라이스를 쓰기로 한 이상 불확실성을 같이 보고하는 것이 정직하다.
    """
    y, s = _as_arrays(y_true, y_score)
    rng = np.random.default_rng(seed)
    stats: list[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, y.size, y.size)
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == yb.size:
            continue  # 한 클래스만 뽑힌 표본은 지표가 정의되지 않는다
        val = metric_fn(yb, s[idx])
        if val == val:
            stats.append(val)
    if len(stats) < 20:
        return (float("nan"), float("nan"))
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return (lo, hi)
