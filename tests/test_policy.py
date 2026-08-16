"""정책 레이어 테스트 — 히스테리시스, EWMA, 신뢰도 가중 감쇠."""

from __future__ import annotations

import pytest

from phishstress.detectors.base import DetectionResult
from phishstress.policy.risk import PolicyConfig, RiskGrade, RiskPolicy


def r(score: float, confidence: float = 1.0) -> DetectionResult:
    return DetectionResult(score=score, confidence=confidence)


class TestConfigValidation:
    def test_rejects_non_hysteresis_thresholds(self):
        with pytest.raises(ValueError, match="warn_exit"):
            PolicyConfig(warn_enter=0.5, warn_exit=0.6)

    def test_rejects_warn_above_block(self):
        with pytest.raises(ValueError, match="warn_enter"):
            PolicyConfig(warn_enter=0.9, warn_exit=0.8, block_enter=0.85, block_exit=0.7)

    def test_rejects_bad_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            PolicyConfig(alpha=0.0)


class TestEwma:
    def test_first_update_takes_instant_value(self):
        p = RiskPolicy(PolicyConfig(alpha=0.3))
        state = p.update({"text": r(0.8)})
        assert state.risk == pytest.approx(0.8)
        assert state.instant_risk == pytest.approx(0.8)

    def test_smoothing_dampens_a_single_spike(self):
        """한 청크 튄다고 등급이 바로 올라가면 안 된다 — 이게 EWMA를 넣은 이유."""
        cfg = PolicyConfig(alpha=0.3, warn_enter=0.6)
        p = RiskPolicy(cfg)
        for _ in range(5):
            p.update({"text": r(0.1)})
        assert p.grade is RiskGrade.SAFE

        state = p.update({"text": r(1.0)})  # 순간 최대 위험
        assert state.instant_risk == pytest.approx(1.0)
        assert state.risk < cfg.warn_enter  # 누적값은 아직 임계 미만
        assert state.grade is RiskGrade.SAFE

    def test_sustained_signal_eventually_escalates(self):
        p = RiskPolicy(PolicyConfig(alpha=0.3, warn_enter=0.6))
        grades = [p.update({"text": r(0.95)}).grade for _ in range(10)]
        assert grades[0] is RiskGrade.BLOCK or RiskGrade.WARN in grades
        assert p.grade in (RiskGrade.WARN, RiskGrade.BLOCK)


class TestHysteresis:
    def test_warn_does_not_flap_between_thresholds(self):
        """warn_exit(0.45)와 warn_enter(0.60) 사이에서는 기존 등급을 유지해야 한다."""
        cfg = PolicyConfig(alpha=1.0, warn_enter=0.60, warn_exit=0.45)
        p = RiskPolicy(cfg)

        assert p.update({"text": r(0.70)}).grade is RiskGrade.WARN
        # 0.50은 enter 미만이지만 exit 초과 → WARN 유지
        assert p.update({"text": r(0.50)}).grade is RiskGrade.WARN
        assert p.update({"text": r(0.55)}).grade is RiskGrade.WARN
        # exit 아래로 내려가야 비로소 해제
        assert p.update({"text": r(0.40)}).grade is RiskGrade.SAFE

    def test_safe_does_not_enter_warn_in_dead_zone(self):
        cfg = PolicyConfig(alpha=1.0, warn_enter=0.60, warn_exit=0.45)
        p = RiskPolicy(cfg)
        assert p.update({"text": r(0.50)}).grade is RiskGrade.SAFE

    def test_block_downgrades_through_warn(self):
        cfg = PolicyConfig(
            alpha=1.0, warn_enter=0.60, warn_exit=0.45, block_enter=0.85, block_exit=0.70
        )
        p = RiskPolicy(cfg)
        assert p.update({"text": r(0.90)}).grade is RiskGrade.BLOCK
        assert p.update({"text": r(0.75)}).grade is RiskGrade.BLOCK  # exit 미달, 유지
        assert p.update({"text": r(0.55)}).grade is RiskGrade.WARN  # BLOCK 해제 → WARN
        assert p.update({"text": r(0.20)}).grade is RiskGrade.SAFE

    def test_direct_jump_safe_to_block(self):
        cfg = PolicyConfig(alpha=1.0, block_enter=0.85)
        p = RiskPolicy(cfg)
        assert p.update({"text": r(0.99)}).grade is RiskGrade.BLOCK


class TestConfidenceWeighting:
    def test_low_confidence_detector_loses_influence(self):
        """열화된 입력을 본 판별기는 발언권이 줄어야 한다 — 정책 레이어의 핵심 규칙."""
        cfg = PolicyConfig(alpha=1.0, base_weights={"text": 0.5, "audio": 0.5})

        full = RiskPolicy(cfg).update({"text": r(0.0, 1.0), "audio": r(1.0, 1.0)})
        assert full.instant_risk == pytest.approx(0.5)

        degraded = RiskPolicy(cfg).update({"text": r(0.0, 1.0), "audio": r(1.0, 0.2)})
        assert degraded.instant_risk < full.instant_risk
        assert degraded.degraded is True

    def test_degraded_flag_only_when_confidence_low(self):
        cfg = PolicyConfig(alpha=1.0)
        assert RiskPolicy(cfg).update({"text": r(0.5, 0.9)}).degraded is False
        assert RiskPolicy(cfg).update({"text": r(0.5, 0.3)}).degraded is True

    def test_contributions_normalize_to_one(self):
        cfg = PolicyConfig(alpha=1.0, base_weights={"text": 0.7, "audio": 0.3})
        state = RiskPolicy(cfg).update({"text": r(0.4, 0.8), "audio": r(0.6, 0.5)})
        total = sum(c["effective_weight"] for c in state.contributions.values())
        assert total == pytest.approx(1.0)

    def test_zero_confidence_does_not_erase_detector(self):
        """가중치 바닥(min_confidence_floor)이 0 나눗셈과 판별기 소멸을 막는다."""
        cfg = PolicyConfig(alpha=1.0, base_weights={"text": 1.0})
        state = RiskPolicy(cfg).update({"text": r(0.9, 0.0)})
        assert state.instant_risk == pytest.approx(0.9)

    def test_unknown_modality_ignored(self):
        cfg = PolicyConfig(alpha=1.0, base_weights={"text": 1.0})
        state = RiskPolicy(cfg).update({"text": r(0.8), "video": r(0.1)})
        assert "video" not in state.contributions
        assert state.instant_risk == pytest.approx(0.8)

    def test_empty_results_yield_zero_risk(self):
        state = RiskPolicy(PolicyConfig(alpha=1.0)).update({})
        assert state.instant_risk == pytest.approx(0.0)
        assert state.grade is RiskGrade.SAFE


class TestSnapshot:
    def test_snapshot_restore_roundtrip(self):
        cfg = PolicyConfig(alpha=0.3)
        p1 = RiskPolicy(cfg)
        for _ in range(4):
            p1.update({"text": r(0.9)})
        snap = p1.snapshot()

        p2 = RiskPolicy(cfg)
        p2.restore(snap)
        assert p2.risk == pytest.approx(p1.risk)
        assert p2.grade is p1.grade

        # 복원 후 이어지는 갱신이 원본과 동일하게 진행되어야 한다
        assert p2.update({"text": r(0.9)}).risk == pytest.approx(p1.update({"text": r(0.9)}).risk)

    def test_reset_clears(self):
        p = RiskPolicy(PolicyConfig(alpha=1.0))
        p.update({"text": r(0.99)})
        p.reset()
        assert p.grade is RiskGrade.SAFE
        assert p.risk == pytest.approx(0.0)


def test_state_to_dict_is_json_serializable():
    import json

    state = RiskPolicy().update({"text": r(0.7, 0.9), "audio": r(0.3, 0.4)})
    payload = json.dumps(state.to_dict())
    assert '"grade"' in payload
