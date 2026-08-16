"""환경변수 기반 설정."""

from __future__ import annotations

import os
from dataclasses import dataclass

from ..policy.risk import PolicyConfig
from .stream import StreamConfig


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class AppConfig:
    redis_url: str | None = None
    session_ttl_sec: int = 300
    max_ws_message_bytes: int = 1 << 20  # 1MiB
    stream: StreamConfig = StreamConfig()
    policy: PolicyConfig = PolicyConfig()

    @classmethod
    def from_env(cls) -> AppConfig:
        return cls(
            redis_url=os.getenv("REDIS_URL") or None,
            session_ttl_sec=_env_int("SESSION_TTL_SEC", 300),
            max_ws_message_bytes=_env_int("MAX_WS_MESSAGE_BYTES", 1 << 20),
            stream=StreamConfig(
                sample_rate=_env_int("SAMPLE_RATE", 16000),
                window_sec=_env_float("WINDOW_SEC", 3.0),
                hop_sec=_env_float("HOP_SEC", 1.0),
            ),
            policy=PolicyConfig(
                alpha=_env_float("POLICY_ALPHA", 0.3),
                warn_enter=_env_float("WARN_ENTER", 0.60),
                warn_exit=_env_float("WARN_EXIT", 0.45),
                block_enter=_env_float("BLOCK_ENTER", 0.85),
                block_exit=_env_float("BLOCK_EXIT", 0.70),
            ),
        )
