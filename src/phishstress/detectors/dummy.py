"""Day 1용 더미 판별기.

학습된 모델 없이 게이트웨이 전 구간을 검증하기 위한 스텁이다.
단, `DummyAudioDetector`의 대역폭 추정은 **실제 신호처리**로 구현했다 —
이 값이 그대로 정책 레이어의 열화 인지 입력이 되고, Day 8 이후
코덱 스트레스 축에서도 재사용되기 때문이다.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .base import AudioChunk, DetectionResult, Detector

# G.711 / AMR-NB 등 협대역 전화망의 상한. 이 위 대역이 비면 전화 코덱을 통과한 신호로 본다.
TELEPHONE_BAND_HZ = 3400.0


def estimate_bandwidth_ratio(samples: np.ndarray, sample_rate: int) -> float:
    """전체 스펙트럼 에너지 중 3.4kHz 초과 대역이 차지하는 비율을 반환한다.

    광대역 원음이면 유의미한 값이 나오고, 전화 코덱을 통과했다면 0에 가까워진다.
    나이퀴스트가 3.4kHz 이하(즉 sample_rate <= 6800)면 판단 자체가 불가하므로 0.0을 준다.
    """
    if samples.size == 0:
        return 0.0
    nyquist = sample_rate / 2.0
    if nyquist <= TELEPHONE_BAND_HZ:
        return 0.0

    spectrum = np.abs(np.fft.rfft(samples.astype(np.float64)))
    power = spectrum**2
    total = float(power.sum())
    if total <= 0.0:
        return 0.0

    freqs = np.fft.rfftfreq(samples.size, d=1.0 / sample_rate)
    high = float(power[freqs > TELEPHONE_BAND_HZ].sum())
    return high / total


def _stable_unit_float(payload: bytes) -> float:
    """입력에 대해 결정적인 [0,1) 값. 테스트 재현성을 위해 난수 대신 해시를 쓴다."""
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


class ConstantDetector(Detector):
    """항상 같은 점수를 내는 판별기. 정책 레이어 단위 테스트용."""

    def __init__(
        self,
        name: str = "constant",
        modality: str = "text",
        score: float = 0.5,
        confidence: float = 1.0,
    ) -> None:
        self.name = name
        self.modality = modality  # type: ignore[assignment]
        self._score = score
        self._confidence = confidence

    def _predict(self, x: AudioChunk | str) -> DetectionResult:
        return DetectionResult(
            score=self._score, confidence=self._confidence, detail={"stub": True}
        )


class DummyAudioDetector(Detector):
    """합성음성 판별기 자리를 채우는 스텁 (Day 9에 AASIST-L로 교체).

    점수는 결정적 해시로 만들지만, `confidence`는 실제 대역폭 추정에서 나온다.
    협대역 입력일수록 confidence가 떨어지고 정책 레이어가 발언권을 줄인다.
    """

    name = "dummy-audio"
    modality = "audio"

    def __init__(self, min_confidence: float = 0.35) -> None:
        self.min_confidence = min_confidence

    def _predict(self, x: AudioChunk | str) -> DetectionResult:
        assert isinstance(x, AudioChunk)
        ratio = estimate_bandwidth_ratio(x.samples, x.sample_rate)
        # 고주파 에너지가 남아있을수록 판단 신뢰도가 높다. 상한 0.15에서 포화.
        confidence = self.min_confidence + (1.0 - self.min_confidence) * min(ratio / 0.15, 1.0)

        payload = x.samples.astype(np.float32).tobytes()[:4096]
        score = _stable_unit_float(payload)

        return DetectionResult(
            score=score,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            detail={
                "stub": True,
                "bandwidth_ratio_above_3400hz": round(ratio, 6),
                "narrowband_suspected": bool(ratio < 0.01),
                "duration_sec": round(x.duration_sec, 3),
            },
        )


class KeywordTextDetector(Detector):
    """사기 대화 판별기 자리를 채우는 규칙 스텁 (Day 5에 학습 모델로 교체).

    Day 3~5의 TF-IDF 베이스라인이 들어오기 전까지 데모가 그럴듯하게 동작하도록,
    공개 보도자료 수준의 일반 어휘만 사용한다. 실제 사기 대본은 포함하지 않는다.
    """

    name = "dummy-text"
    modality = "text"

    # 금융감독원·경찰청 공개 안내문에서 반복 등장하는 일반 어휘.
    DEFAULT_MARKERS: tuple[str, ...] = (
        "검찰",
        "금융감독원",
        "계좌",
        "이체",
        "송금",
        "대출",
        "저금리",
        "명의도용",
        "안전계좌",
        "체포영장",
        "개인정보",
        "인증번호",
        "앱 설치",
        "원격",
    )

    def __init__(self, markers: tuple[str, ...] | None = None, saturation: int = 4) -> None:
        self.markers = markers or self.DEFAULT_MARKERS
        self.saturation = max(saturation, 1)

    def _predict(self, x: AudioChunk | str) -> DetectionResult:
        assert isinstance(x, str)
        text = x.strip()
        if not text:
            return DetectionResult(score=0.0, confidence=0.2, detail={"reason": "empty_text"})

        hits = [m for m in self.markers if m in text]
        score = min(len(hits) / self.saturation, 1.0)

        # 텍스트가 너무 짧으면 판단 근거가 빈약하므로 신뢰도를 낮춘다.
        confidence = min(len(text) / 40.0, 1.0) * 0.6 + 0.4

        return DetectionResult(
            score=score,
            confidence=round(confidence, 4),
            detail={"stub": True, "matched_markers": hits, "text_length": len(text)},
        )
