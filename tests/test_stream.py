"""링버퍼 윈도잉 정확성 테스트.

여기가 틀리면 이후 모든 지표가 조용히 틀어진다. 경계 조건을 촘촘히 본다.
"""

from __future__ import annotations

import numpy as np
import pytest

from phishstress.serving.stream import RingBuffer, StreamConfig, pcm16_to_float32

SR = 16000
CFG = StreamConfig(sample_rate=SR, window_sec=3.0, hop_sec=1.0)


def ramp(n: int, start: int = 0) -> np.ndarray:
    """샘플 위치를 값으로 갖는 신호. 윈도우가 어느 구간을 잘랐는지 검증할 수 있다."""
    return np.arange(start, start + n, dtype=np.float32)


def test_config_derived_sizes():
    assert CFG.window_samples == 48000
    assert CFG.hop_samples == 16000


def test_config_rejects_hop_larger_than_window():
    with pytest.raises(ValueError, match="hop_sec"):
        StreamConfig(sample_rate=SR, window_sec=1.0, hop_sec=2.0)


def test_no_window_before_buffer_fills():
    buf = RingBuffer(CFG)
    out = buf.push(ramp(SR * 2))  # 2초 < 3초 윈도우
    assert out == []
    assert buf.windows_emitted == 0


def test_first_window_emitted_exactly_at_window_length():
    buf = RingBuffer(CFG)
    assert buf.push(ramp(CFG.window_samples - 1)) == []
    out = buf.push(ramp(1, start=CFG.window_samples - 1))
    assert len(out) == 1
    assert out[0].samples.size == CFG.window_samples
    assert out[0].start_sec == pytest.approx(0.0)


def test_window_covers_correct_absolute_range():
    """윈도우 k는 절대 구간 [kH, kH+W)를 담아야 한다."""
    buf = RingBuffer(CFG)
    out = buf.push(ramp(CFG.window_samples + CFG.hop_samples * 2))
    assert len(out) == 3  # k=0,1,2
    for k, chunk in enumerate(out):
        lo = k * CFG.hop_samples
        assert chunk.samples[0] == pytest.approx(float(lo))
        assert chunk.samples[-1] == pytest.approx(float(lo + CFG.window_samples - 1))
        assert chunk.start_sec == pytest.approx(lo / SR)


def test_windows_are_independent_of_push_chunking():
    """한 번에 밀어넣든 잘게 나눠 넣든 동일한 윈도우가 나와야 한다."""
    total = CFG.window_samples + CFG.hop_samples * 3
    signal = ramp(total)

    bulk = RingBuffer(CFG)
    bulk_out = bulk.push(signal)

    piecemeal = RingBuffer(CFG)
    piece_out = []
    step = 777  # 윈도우/홉과 서로소인 지저분한 크기
    for i in range(0, total, step):
        piece_out.extend(piecemeal.push(signal[i : i + step]))

    assert len(bulk_out) == len(piece_out) == 4
    for a, b in zip(bulk_out, piece_out, strict=True):
        np.testing.assert_array_equal(a.samples, b.samples)
        assert a.start_sec == pytest.approx(b.start_sec)


def test_memory_stays_bounded_over_long_call():
    """10분 통화를 흘려도 버퍼가 윈도우 길이를 넘지 않아야 한다."""
    buf = RingBuffer(CFG)
    for _ in range(600):  # 1초씩 600회 = 10분
        buf.push(np.zeros(SR, dtype=np.float32))
        assert buf.buffered_samples <= CFG.window_samples
    assert buf.total_samples == SR * 600
    assert buf.windows_emitted == 598  # 3초에 첫 윈도우, 이후 1초마다


def test_flush_pads_short_tail():
    buf = RingBuffer(CFG)
    buf.push(ramp(SR))  # 1초만 넣고 종료
    tail = buf.flush()
    assert tail is not None
    assert tail.samples.size == CFG.window_samples
    assert tail.samples[0] == pytest.approx(0.0)  # 앞쪽 제로 패딩
    assert tail.samples[-1] == pytest.approx(float(SR - 1))


def test_flush_returns_none_when_empty():
    assert RingBuffer(CFG).flush() is None


def test_reset_clears_state():
    buf = RingBuffer(CFG)
    buf.push(ramp(CFG.window_samples * 2))
    buf.reset()
    assert buf.total_samples == 0
    assert buf.windows_emitted == 0
    assert buf.push(ramp(10)) == []


def test_push_rejects_2d_array():
    with pytest.raises(ValueError, match="1차원"):
        RingBuffer(CFG).push(np.zeros((2, 10), dtype=np.float32))


class TestPcm16Conversion:
    def test_roundtrip_range(self):
        ints = np.array([0, 32767, -32768, 16384], dtype="<i2")
        out = pcm16_to_float32(ints.tobytes())
        assert out.dtype == np.float32
        np.testing.assert_allclose(out, [0.0, 32767 / 32768, -1.0, 0.5], atol=1e-6)

    def test_empty_payload(self):
        assert pcm16_to_float32(b"").size == 0

    def test_odd_length_rejected(self):
        with pytest.raises(ValueError, match="짝수"):
            pcm16_to_float32(b"\x00\x01\x02")
