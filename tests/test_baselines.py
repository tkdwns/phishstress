"""베이스라인 판별기 + 데이터셋 진단 검증.

베이스라인의 존재 이유는 두 가지다 — 평가 코드가 맞는지 확인하는 것,
그리고 앞으로 만들 모델이 넘어야 할 바닥을 못 박는 것.
"""

from __future__ import annotations

import numpy as np
import pytest

from phishstress.data.loaders import Example
from phishstress.detectors.base import AudioChunk
from phishstress.eval.baselines import (
    ConstantScoreDetector,
    LengthDetector,
    OracleDetector,
    RandomDetector,
)
from phishstress.eval.diagnostics import (
    diagnose,
    rank_auc,
    vocab_hit_rate,
)
from phishstress.eval.metrics import roc_auc
from phishstress.eval.runner import evaluate, evaluate_slice


def toy(n_pos=60, n_neg=140, pos_len=400, neg_len=4000) -> list[Example]:
    out = [Example(f"p{i}", "계좌 이체 송금 " * (pos_len // 8 + i), 1) for i in range(n_pos)]
    out += [Example(f"n{i}", "여행 친구 음식 " * (neg_len // 8 + i), 0) for i in range(n_neg)]
    return out


# ------------------------------------------------------------- 계약 준수


class TestContract:
    @pytest.mark.parametrize(
        "det",
        [
            ConstantScoreDetector(),
            RandomDetector(),
            LengthDetector(),
            OracleDetector(labels={}),
        ],
    )
    def test_all_are_text_detectors(self, det):
        assert det.modality == "text"
        r = det.predict("아무 텍스트")
        assert 0.0 <= r.score <= 1.0
        assert 0.0 <= r.confidence <= 1.0

    def test_rejects_audio_input(self):
        chunk = AudioChunk(samples=np.zeros(10, dtype=np.float32), sample_rate=16000)
        with pytest.raises(TypeError):
            LengthDetector().predict(chunk)


# ------------------------------------------- 평가 코드 검증용 성질


class TestOracle:
    def test_perfect_oracle_scores_one(self):
        ex = toy()
        det = OracleDetector({e.text: e.label for e in ex})
        r = evaluate_slice(det, ex, "test")
        assert r.metrics.pr_auc == pytest.approx(1.0)
        assert r.metrics.roc_auc == pytest.approx(1.0)
        assert r.metrics.recall_at_fpr01 == pytest.approx(1.0)

    def test_noisy_oracle_degrades_monotonically(self):
        ex = toy()
        labels = {e.text: e.label for e in ex}
        aucs = [
            evaluate_slice(OracleDetector(labels, noise=nz), ex, "t").metrics.roc_auc
            for nz in (0.0, 0.4, 0.8, 1.0)
        ]
        assert aucs == sorted(aucs, reverse=True), f"노이즈가 커지면 성능이 내려가야 한다: {aucs}"

    def test_unknown_text_falls_back(self):
        r = OracleDetector({}).predict("본 적 없는 텍스트")
        assert r.score == pytest.approx(0.5)
        assert r.confidence == 0.0

    def test_rejects_bad_noise(self):
        with pytest.raises(ValueError, match="noise"):
            OracleDetector({}, noise=1.5)


class TestConstantAndRandom:
    def test_constant_zero_equals_majority_accuracy(self):
        ex = toy()
        m = evaluate_slice(ConstantScoreDetector(0.0), ex, "t").metrics
        assert m.accuracy == pytest.approx(m.majority_accuracy)
        assert m.roc_auc == pytest.approx(0.5)

    def test_constant_pr_auc_equals_positive_rate(self):
        ex = toy()
        m = evaluate_slice(ConstantScoreDetector(0.0), ex, "t").metrics
        assert m.pr_auc == pytest.approx(m.positive_rate, abs=1e-9)

    def test_random_is_near_chance(self):
        ex = toy(n_pos=500, n_neg=1500)
        m = evaluate_slice(RandomDetector(seed=1), ex, "t").metrics
        assert m.roc_auc == pytest.approx(0.5, abs=0.05)
        assert m.pr_auc == pytest.approx(m.positive_rate, abs=0.05)

    def test_random_is_deterministic(self):
        d = RandomDetector(seed=42)
        assert d.predict("같은 입력").score == d.predict("같은 입력").score

    def test_random_seed_changes_output(self):
        a = RandomDetector(seed=1).predict("x").score
        b = RandomDetector(seed=2).predict("x").score
        assert a != b


class TestLengthDetector:
    def test_fit_learns_direction_short_is_positive(self):
        ex = toy(pos_len=400, neg_len=4000)
        det = LengthDetector().fit([e.text for e in ex], [e.label for e in ex])
        assert det.direction == -1

    def test_fit_learns_direction_long_is_positive(self):
        ex = toy(pos_len=4000, neg_len=400)
        det = LengthDetector().fit([e.text for e in ex], [e.label for e in ex])
        assert det.direction == 1

    def test_exploits_length_artifact(self):
        """길이가 갈린 데이터에서는 글자 수만 세도 거의 완벽하다 — 이게 문제다."""
        ex = toy()
        det = LengthDetector().fit([e.text for e in ex], [e.label for e in ex])
        m = evaluate_slice(det, ex, "t").metrics
        assert m.roc_auc > 0.95, "길이 아티팩트가 있으면 높게 나와야 한다"

    def test_is_chance_when_lengths_match(self):
        ex = [Example(f"p{i}", "가" * 1000, 1) for i in range(50)]
        ex += [Example(f"n{i}", "나" * 1000, 0) for i in range(50)]
        det = LengthDetector().fit([e.text for e in ex], [e.label for e in ex])
        assert evaluate_slice(det, ex, "t").metrics.roc_auc == pytest.approx(0.5)

    def test_score_stays_in_range_for_extreme_lengths(self):
        det = LengthDetector(pivot=1000, scale=10)
        assert 0.0 <= det.predict("").score <= 1.0
        assert 0.0 <= det.predict("가" * 500000).score <= 1.0

    def test_fit_is_noop_on_single_class(self):
        det = LengthDetector()
        before = (det.pivot, det.scale, det.direction)
        det.fit(["가" * 10], [1])
        assert (det.pivot, det.scale, det.direction) == before


# ----------------------------------------------------------- 진단 모듈


class TestRankAuc:
    def test_matches_roc_auc(self):
        rng = np.random.default_rng(0)
        y = (rng.random(300) < 0.3).astype(int)
        v = rng.normal(0, 1, 300) + y
        assert rank_auc(list(v), list(y)) == pytest.approx(roc_auc(y, v), abs=1e-9)

    def test_all_ties_is_half(self):
        assert rank_auc([1.0] * 10, [0, 1] * 5) == pytest.approx(0.5)

    def test_single_class_is_nan(self):
        assert np.isnan(rank_auc([1.0, 2.0], [0, 0]))


class TestDiagnose:
    def test_flags_length_artifact(self):
        d = diagnose(toy())
        assert d.length_ceiling_accuracy > 0.9
        assert any("길이 아티팩트" in w for w in d.warnings)

    def test_flags_topical_separation(self):
        d = diagnose(toy())
        assert any("주제 분리" in w for w in d.warnings)
        assert any("하드 네거티브" in w for w in d.warnings)

    def test_clean_dataset_has_no_warnings(self):
        """길이도 어휘도 갈리지 않은 데이터에는 경고가 없어야 한다 (거짓 경보 방지)."""
        rng = np.random.default_rng(3)
        ex = []
        for i in range(400):
            label = i % 2
            n = int(rng.integers(900, 1100))
            ex.append(Example(f"x{i}", "계좌 여행 " * (n // 6), label))
        d = diagnose(ex)
        assert d.warnings == [], f"거짓 경보: {d.warnings}"

    def test_vocab_hit_rate(self):
        ex = toy()
        assert vocab_hit_rate(ex, ("계좌",), 1) == pytest.approx(1.0)
        assert vocab_hit_rate(ex, ("계좌",), 0) == pytest.approx(0.0)

    def test_separability_properties_in_range(self):
        d = diagnose(toy())
        assert 0.0 <= d.length_separability <= 1.0
        assert 0.0 <= d.topical_separability <= 1.0

    def test_to_dict_is_json_safe(self):
        import json

        json.dumps(diagnose(toy()).to_dict())

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="빈 데이터셋"):
            diagnose([])

    def test_rejects_single_class(self):
        with pytest.raises(ValueError, match="한 클래스만"):
            diagnose([Example("a", "가", 1), Example("b", "나", 1)])


class TestRunner:
    def test_evaluate_multiple_slices(self):
        ex = toy()
        rep = evaluate(ConstantScoreDetector(0.0), {"a": ex, "b": ex[:100]})
        assert {r.slice_name for r in rep.results} == {"a", "b"}
        assert rep.by_slice("a") is not None
        assert rep.by_slice("zzz") is None

    def test_gap_is_zero_for_identical_slices(self):
        ex = toy()
        rep = evaluate(RandomDetector(seed=5), {"a": ex, "b": ex})
        assert rep.gap("a", "b") == pytest.approx(0.0)

    def test_empty_slice_is_skipped(self):
        rep = evaluate(ConstantScoreDetector(), {"a": toy(), "empty": []})
        assert [r.slice_name for r in rep.results] == ["a"]

    def test_records_latency(self):
        r = evaluate_slice(ConstantScoreDetector(), toy(), "t")
        assert r.latency_ms_mean >= 0.0
        assert r.latency_ms_p95 >= r.latency_ms_mean * 0.5

    def test_rejects_empty_slice_directly(self):
        with pytest.raises(ValueError, match="비어 있습니다"):
            evaluate_slice(ConstantScoreDetector(), [], "t")

    def test_rejects_audio_detector(self):
        from phishstress.detectors.dummy import DummyAudioDetector

        with pytest.raises(ValueError, match="텍스트 판별기가 아닙니다"):
            evaluate_slice(DummyAudioDetector(), toy(), "t")

    def test_report_to_dict_is_json_safe(self):
        import json

        rep = evaluate(LengthDetector(), {"a": toy()})
        json.dumps(rep.to_dict())
