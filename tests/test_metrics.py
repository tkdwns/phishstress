"""지표 구현 검증.

핵심 전략 두 가지.

1. **해석적으로 답을 아는 입력**으로 확인한다. 완벽한 예측기는 PR-AUC 1.0,
   무작위 예측기는 양성비율, 상수 예측기는 ROC-AUC 0.5여야 한다.
2. **scikit-learn과 교차 검증**한다. numpy로 직접 구현한 이유는 core 의존성을
   가볍게 유지하기 위해서지, 정확도를 포기해서가 아니다. sklearn은 dev 전용이므로
   테스트에서만 대조한다.
"""

from __future__ import annotations

import numpy as np
import pytest

from phishstress.eval.metrics import (
    accuracy_at,
    average_precision,
    bootstrap_ci,
    compute,
    equal_error_rate,
    f1_at,
    recall_at_fpr,
    robustness_gap,
    roc_auc,
)

sk = pytest.importorskip("sklearn.metrics", reason="교차 검증용 dev 의존성")


def rng_case(n=400, seed=0, pos_rate=0.25, signal=1.0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < pos_rate).astype(int)
    s = rng.normal(0, 1, n) + signal * y
    return y, 1 / (1 + np.exp(-s))


class TestAnalytic:
    def test_perfect_predictor(self):
        y = np.array([0, 0, 1, 1])
        s = np.array([0.1, 0.2, 0.8, 0.9])
        assert average_precision(y, s) == pytest.approx(1.0)
        assert roc_auc(y, s) == pytest.approx(1.0)
        assert recall_at_fpr(y, s, 0.01) == pytest.approx(1.0)
        assert equal_error_rate(y, s) == pytest.approx(0.0)

    def test_inverted_predictor(self):
        y = np.array([0, 0, 1, 1])
        s = np.array([0.9, 0.8, 0.2, 0.1])
        assert roc_auc(y, s) == pytest.approx(0.0)
        assert equal_error_rate(y, s) == pytest.approx(1.0)

    def test_constant_predictor_is_chance(self):
        y = np.array([0] * 75 + [1] * 25)
        s = np.full(100, 0.5)
        assert roc_auc(y, s) == pytest.approx(0.5)
        # 상수 예측기의 PR-AUC는 양성비율이다
        assert average_precision(y, s) == pytest.approx(0.25)

    def test_random_predictor_converges_to_positive_rate(self):
        y, _ = rng_case(n=20000, seed=7, pos_rate=0.3, signal=0.0)
        s = np.random.default_rng(99).random(y.size)
        assert average_precision(y, s) == pytest.approx(0.3, abs=0.02)
        assert roc_auc(y, s) == pytest.approx(0.5, abs=0.02)

    def test_single_class_returns_nan(self):
        y = np.zeros(10, dtype=int)
        s = np.linspace(0, 1, 10)
        assert np.isnan(roc_auc(y, s))
        assert np.isnan(recall_at_fpr(y, s))
        assert np.isnan(equal_error_rate(y, s))


class TestAgainstSklearn:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_average_precision_matches(self, seed):
        y, s = rng_case(seed=seed)
        assert average_precision(y, s) == pytest.approx(sk.average_precision_score(y, s), abs=1e-9)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_roc_auc_matches(self, seed):
        y, s = rng_case(seed=seed)
        assert roc_auc(y, s) == pytest.approx(sk.roc_auc_score(y, s), abs=1e-9)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_accuracy_and_f1_match(self, seed):
        y, s = rng_case(seed=seed)
        pred = (s >= 0.5).astype(int)
        assert accuracy_at(y, s) == pytest.approx(sk.accuracy_score(y, pred))
        assert f1_at(y, s) == pytest.approx(sk.f1_score(y, pred, zero_division=0))

    def test_handles_heavy_ties(self):
        """동점 점수가 많을 때 sklearn과 갈리기 쉬운 지점."""
        y = np.array([1, 0, 1, 0, 1, 0, 0, 1])
        s = np.array([0.5, 0.5, 0.5, 0.5, 0.9, 0.9, 0.1, 0.1])
        assert average_precision(y, s) == pytest.approx(sk.average_precision_score(y, s), abs=1e-9)
        assert roc_auc(y, s) == pytest.approx(sk.roc_auc_score(y, s), abs=1e-9)

    def test_extreme_imbalance(self):
        y = np.array([0] * 999 + [1])
        s = np.r_[np.random.default_rng(3).random(999) * 0.4, 0.99]
        assert average_precision(y, s) == pytest.approx(sk.average_precision_score(y, s), abs=1e-9)


class TestRecallAtFpr:
    def test_respects_fpr_budget(self):
        """FPR 예산 안에서 달성 가능한 최대 TPR을 골라야 한다."""
        y = np.array([0] * 100 + [1] * 100)
        s = np.r_[np.linspace(0, 0.5, 100), np.linspace(0.5, 1.0, 100)]
        assert recall_at_fpr(y, s, 0.01) >= 0.9

    def test_zero_when_no_separation(self):
        y = np.array([0] * 100 + [1] * 100)
        s = np.r_[np.linspace(0.5, 1.0, 100), np.linspace(0.0, 0.5, 100)]
        assert recall_at_fpr(y, s, 0.01) < 0.05

    def test_rejects_bad_budget(self):
        with pytest.raises(ValueError, match="max_fpr"):
            recall_at_fpr([0, 1], [0.1, 0.9], 0.0)


class TestValidation:
    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="길이가 다릅니다"):
            roc_auc([0, 1], [0.1, 0.2, 0.3])

    def test_non_binary_labels(self):
        with pytest.raises(ValueError, match="0/1"):
            roc_auc([0, 1, 2], [0.1, 0.2, 0.3])

    def test_nan_scores(self):
        with pytest.raises(ValueError, match="NaN"):
            roc_auc([0, 1], [0.1, float("nan")])

    def test_empty(self):
        with pytest.raises(ValueError, match="빈 입력"):
            roc_auc([], [])


class TestMetricSet:
    def test_compute_reports_majority_baseline(self):
        y = [0] * 76 + [1] * 24
        s = np.random.default_rng(1).random(100)
        m = compute(y, s)
        assert m.n == 100 and m.positives == 24
        assert m.majority_accuracy == pytest.approx(0.76)

    def test_to_dict_is_json_safe(self):
        import json

        y, s = rng_case(seed=5)
        json.dumps(compute(y, s).to_dict())


class TestRobustnessGap:
    def test_positive_gap_means_degradation(self):
        y, s = rng_case(seed=2, signal=3.0)
        clean = compute(y, s)
        stressed = compute(y, np.random.default_rng(0).random(y.size))
        gap = robustness_gap(clean, stressed)
        assert gap > 0, "열화 조건에서 성능이 떨어지면 Gap은 양수여야 한다"

    def test_zero_gap_when_identical(self):
        y, s = rng_case(seed=2)
        m = compute(y, s)
        assert robustness_gap(m, m) == pytest.approx(0.0)


class TestBootstrap:
    def test_ci_brackets_point_estimate(self):
        y, s = rng_case(n=300, seed=11, signal=2.0)
        point = average_precision(y, s)
        lo, hi = bootstrap_ci(y, s, n_resamples=400)
        assert lo <= point <= hi

    def test_small_sample_gives_wide_interval(self):
        y_small, s_small = rng_case(n=40, seed=4, signal=1.0)
        y_big, s_big = rng_case(n=2000, seed=4, signal=1.0)
        w_small = np.subtract(*reversed(bootstrap_ci(y_small, s_small, n_resamples=400)))
        w_big = np.subtract(*reversed(bootstrap_ci(y_big, s_big, n_resamples=400)))
        assert w_small > w_big, "표본이 작을수록 구간이 넓어야 한다"

    def test_deterministic(self):
        y, s = rng_case(n=200, seed=8)
        assert bootstrap_ci(y, s, n_resamples=200) == bootstrap_ci(y, s, n_resamples=200)
