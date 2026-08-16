"""슬라이딩 윈도우 링버퍼.

통화 오디오를 3초 윈도우 / 1초 홉으로 잘라 판별기에 넘긴다.
윈도우를 겹치는 이유: 사기 발화가 청크 경계에 걸려 잘리는 것을 막기 위해서다.

메모리 상한이 설계 요건이다 — 통화가 아무리 길어도 버퍼는 윈도우 길이 미만으로 유지된다.
"오디오를 디스크에 쓰지 않는다"는 프로젝트 원칙(docs/ETHICS.md)이 여기서 구현된다.

윈도우 경계는 **절대 샘플 인덱스**로 관리한다. k번째 윈도우는
    [W + kH - W, W + kH)  =  [kH, kH + W)
구간이며, 버퍼를 잘라내도 경계 계산이 흔들리지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..detectors.base import AudioChunk


@dataclass(frozen=True)
class StreamConfig:
    sample_rate: int = 16000
    window_sec: float = 3.0
    hop_sec: float = 1.0

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate는 양수여야 합니다.")
        if self.window_sec <= 0 or self.hop_sec <= 0:
            raise ValueError("window_sec, hop_sec는 양수여야 합니다.")
        if self.hop_sec > self.window_sec:
            raise ValueError("hop_sec는 window_sec보다 클 수 없습니다.")

    @property
    def window_samples(self) -> int:
        return int(round(self.sample_rate * self.window_sec))

    @property
    def hop_samples(self) -> int:
        return int(round(self.sample_rate * self.hop_sec))


class RingBuffer:
    """PCM 샘플을 받아 완성된 윈도우만 방출한다.

    buf = RingBuffer(StreamConfig())
    for chunk in buf.push(samples):   # 완성된 윈도우가 없으면 빈 리스트
        detector.predict(chunk)
    """

    def __init__(self, config: StreamConfig | None = None) -> None:
        self.config = config or StreamConfig()
        self._buffer = np.zeros(0, dtype=np.float32)
        self._buffer_start = 0  # _buffer[0]의 절대 샘플 인덱스
        self._total_samples = 0
        self._next_end = self.config.window_samples  # 다음 윈도우가 끝나는 절대 인덱스
        self._windows_emitted = 0

    # ---- 관측용 프로퍼티 -------------------------------------------------

    @property
    def total_samples(self) -> int:
        return self._total_samples

    @property
    def windows_emitted(self) -> int:
        return self._windows_emitted

    @property
    def elapsed_sec(self) -> float:
        return self._total_samples / self.config.sample_rate

    @property
    def buffered_samples(self) -> int:
        return int(self._buffer.size)

    # ---- 핵심 동작 -------------------------------------------------------

    def push(self, samples: np.ndarray) -> list[AudioChunk]:
        """샘플을 적재하고, 이번 호출로 완성된 윈도우들을 시간 순으로 반환한다."""
        if samples.ndim != 1:
            raise ValueError(f"모노 1차원 배열이어야 합니다. got ndim={samples.ndim}")
        if samples.size == 0:
            return []

        cfg = self.config
        w, h = cfg.window_samples, cfg.hop_samples

        self._buffer = np.concatenate([self._buffer, samples.astype(np.float32, copy=False)])
        self._total_samples += int(samples.size)

        out: list[AudioChunk] = []
        while self._total_samples >= self._next_end:
            lo_abs = self._next_end - w
            lo = lo_abs - self._buffer_start
            hi = self._next_end - self._buffer_start
            if lo < 0:
                # 방어적 처리: 정상 흐름에서는 트리밍이 이 상황을 막는다.
                break
            out.append(
                AudioChunk(
                    samples=self._buffer[lo:hi].copy(),
                    sample_rate=cfg.sample_rate,
                    start_sec=lo_abs / cfg.sample_rate,
                )
            )
            self._windows_emitted += 1
            self._next_end += h

        # 다음 윈도우 시작점 이전 구간은 더 이상 필요 없다 → 버림 (메모리 상한)
        keep_from_abs = max(0, self._next_end - w)
        drop = keep_from_abs - self._buffer_start
        if drop > 0:
            self._buffer = self._buffer[drop:]
            self._buffer_start += drop

        return out

    def flush(self) -> AudioChunk | None:
        """통화 종료 시 남은 잔여 구간을 마지막 윈도우로 한 번 방출한다.

        윈도우 길이에 못 미치면 앞쪽을 0으로 패딩한다. 남은 샘플이 없으면 None.
        """
        if self._buffer.size == 0:
            return None
        cfg = self.config
        w = cfg.window_samples
        window = self._buffer
        if window.size < w:
            window = np.pad(window, (w - window.size, 0))
        window = window[-w:].copy()
        start_abs = max(0, self._total_samples - w)
        self._windows_emitted += 1
        return AudioChunk(
            samples=window,
            sample_rate=cfg.sample_rate,
            start_sec=start_abs / cfg.sample_rate,
        )

    def reset(self) -> None:
        self._buffer = np.zeros(0, dtype=np.float32)
        self._buffer_start = 0
        self._total_samples = 0
        self._next_end = self.config.window_samples
        self._windows_emitted = 0


def pcm16_to_float32(payload: bytes) -> np.ndarray:
    """WebSocket으로 들어온 int16 리틀엔디언 PCM을 [-1,1] float32로 변환한다."""
    if len(payload) % 2 != 0:
        raise ValueError("int16 PCM은 바이트 길이가 짝수여야 합니다.")
    if not payload:
        return np.zeros(0, dtype=np.float32)
    ints = np.frombuffer(payload, dtype="<i2")
    return (ints.astype(np.float32) / 32768.0).copy()
