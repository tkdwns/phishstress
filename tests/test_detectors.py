"""판별기 계약 테스트.

여기서 검증하는 것은 더미 판별기의 '성능'이 아니라 **계약 준수 여부**다.
Day 5/Day 9에 학습 모델이 들어와도 이 테스트는 그대로 통과해야 한다.
"""

from __future__ import annotations

import numpy as np
import pytest

from phishstress.detectors.base import (
    AudioChunk,
    DetectionResult,
    Detector,
    DetectorRegistry,
)
from phishstress.detectors.dummy import (
    ConstantDetector,
    DummyAudioDetector,
    KeywordTextDetector,
    estimate_bandwidth_ratio,
)

SR = 16000


def tone(freq: float, sec: float = 3.0, sr: int = SR) -> np.ndarray:
    t = np.arange(int(sr * sec), dtype=np.float32) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


class TestAudioChunk:
    def test_rejects_2d(self):
        with pytest.raises(ValueError, match="1차원"):
            AudioChunk(samples=np.zeros((2, 10), dtype=np.float32), sample_rate=SR)

    def test_rejects_bad_sample_rate(self):
        with pytest.raises(ValueError, match="sample_rate"):
            AudioChunk(samples=np.zeros(10, dtype=np.float32), sample_rate=0)

    def test_duration(self):
        c = AudioChunk(samples=np.zeros(SR * 3, dtype=np.float32), sample_rate=SR)
        assert c.duration_sec == pytest.approx(3.0)


class TestDetectionResult:
    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_score_range_enforced(self, bad):
        with pytest.raises(ValueError, match="score"):
            DetectionResult(score=bad)

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_confidence_range_enforced(self, bad):
        with pytest.raises(ValueError, match="confidence"):
            DetectionResult(score=0.5, confidence=bad)


class TestContract:
    def test_latency_is_filled_by_base_class(self):
        d = ConstantDetector(score=0.5)
        assert d.predict("안녕하세요").latency_ms > 0.0

    def test_text_detector_rejects_audio_input(self):
        d = KeywordTextDetector()
        chunk = AudioChunk(samples=np.zeros(100, dtype=np.float32), sample_rate=SR)
        with pytest.raises(TypeError, match="str"):
            d.predict(chunk)

    def test_audio_detector_rejects_text_input(self):
        with pytest.raises(TypeError, match="AudioChunk"):
            DummyAudioDetector().predict("문자열")

    def test_abstract_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            Detector()  # type: ignore[abstract]


class TestBandwidthEstimation:
    """confidence의 근거가 되는 실제 신호처리. 여기가 Day 8 코덱 축의 씨앗이다."""

    def test_wideband_tone_has_high_frequency_energy(self):
        ratio = estimate_bandwidth_ratio(tone(6000.0), SR)
        assert ratio > 0.9

    def test_narrowband_tone_has_almost_none(self):
        ratio = estimate_bandwidth_ratio(tone(800.0), SR)
        assert ratio < 0.01

    def test_returns_zero_when_nyquist_below_telephone_band(self):
        """8kHz 샘플레이트는 나이퀴스트 4kHz라 판단은 가능하지만,
        6.8kHz 이하면 3.4kHz 초과 대역 자체가 거의 없어 판단 불가로 본다."""
        assert estimate_bandwidth_ratio(tone(1000.0, sr=6000), 6000) == 0.0

    def test_returns_zero_on_silence(self):
        assert estimate_bandwidth_ratio(np.zeros(SR, dtype=np.float32), SR) == 0.0

    def test_returns_zero_on_empty(self):
        assert estimate_bandwidth_ratio(np.zeros(0, dtype=np.float32), SR) == 0.0


class TestDummyAudioDetector:
    def test_confidence_drops_for_narrowband_input(self):
        """전화 대역으로 잘린 신호에서는 판별기가 스스로 신뢰도를 낮춰야 한다."""
        d = DummyAudioDetector()
        wide = d.predict(AudioChunk(samples=tone(6000.0), sample_rate=SR))
        narrow = d.predict(AudioChunk(samples=tone(800.0), sample_rate=SR))
        assert wide.confidence > narrow.confidence
        assert narrow.detail["narrowband_suspected"] is True
        assert wide.detail["narrowband_suspected"] is False

    def test_confidence_never_below_floor(self):
        d = DummyAudioDetector(min_confidence=0.35)
        res = d.predict(AudioChunk(samples=np.zeros(SR, dtype=np.float32), sample_rate=SR))
        assert res.confidence >= 0.35

    def test_score_is_deterministic(self):
        """테스트 재현성을 위해 난수 대신 해시를 쓴다."""
        d = DummyAudioDetector()
        chunk = AudioChunk(samples=tone(1000.0), sample_rate=SR)
        assert d.predict(chunk).score == d.predict(chunk).score

    def test_different_audio_gives_different_score(self):
        d = DummyAudioDetector()
        a = d.predict(AudioChunk(samples=tone(1000.0), sample_rate=SR)).score
        b = d.predict(AudioChunk(samples=tone(2000.0), sample_rate=SR)).score
        assert a != b


class TestKeywordTextDetector:
    def test_neutral_text_scores_zero(self):
        res = KeywordTextDetector().predict("오늘 점심 뭐 먹을지 고민이네요 날씨도 좋고")
        assert res.score == pytest.approx(0.0)

    def test_marker_rich_text_scores_high(self):
        text = "저는 금융감독원 직원입니다. 안전계좌로 즉시 이체 송금 하셔야 합니다."
        res = KeywordTextDetector().predict(text)
        assert res.score >= 0.75
        assert len(res.detail["matched_markers"]) >= 3

    def test_empty_text_has_low_confidence(self):
        res = KeywordTextDetector().predict("   ")
        assert res.score == pytest.approx(0.0)
        assert res.confidence < 0.5

    def test_short_text_lowers_confidence(self):
        d = KeywordTextDetector()
        short = d.predict("계좌")
        long = d.predict("계좌 관련해서 안내드릴 내용이 있어 연락드렸습니다 확인 부탁드립니다")
        assert long.confidence > short.confidence

    def test_score_is_capped_at_one(self):
        text = "검찰 금융감독원 계좌 이체 송금 대출 저금리 명의도용 안전계좌 체포영장"
        assert KeywordTextDetector().predict(text).score == pytest.approx(1.0)


class TestRegistry:
    def test_register_and_get(self):
        reg = DetectorRegistry()
        d = reg.register(KeywordTextDetector())
        assert reg.get("dummy-text") is d
        assert len(reg) == 1
        assert "dummy-text" in reg

    def test_duplicate_name_rejected(self):
        reg = DetectorRegistry()
        reg.register(KeywordTextDetector())
        with pytest.raises(ValueError, match="이미 등록"):
            reg.register(KeywordTextDetector())

    def test_unknown_name_raises(self):
        with pytest.raises(KeyError, match="등록되지 않은"):
            DetectorRegistry().get("nope")

    def test_by_modality_filters(self):
        reg = DetectorRegistry()
        reg.register(KeywordTextDetector())
        reg.register(DummyAudioDetector())
        assert [d.name for d in reg.by_modality("text")] == ["dummy-text"]
        assert [d.name for d in reg.by_modality("audio")] == ["dummy-audio"]

    def test_describe_shape(self):
        info = KeywordTextDetector().describe()
        assert info == {
            "name": "dummy-text",
            "modality": "text",
            "class": "KeywordTextDetector",
        }
