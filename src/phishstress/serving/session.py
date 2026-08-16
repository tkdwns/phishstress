"""세션 상태 저장소.

저장하는 것은 **정책 스냅샷뿐**이다 — EWMA 위험도, 등급, 갱신 횟수 세 개.
오디오도 전사 텍스트도 저장하지 않는다(docs/ETHICS.md). 재접속 시 위험도 누적을
이어가기 위한 최소한의 상태만 TTL과 함께 보관한다.

기본은 인메모리이고, REDIS_URL이 설정되어 있으면 Redis를 쓴다.
Redis 연결에 실패하면 인메모리로 자동 폴백한다 — Day 1의 목표는
'모델도 Redis도 없이 docker compose up 하나로 도는 시스템'이기 때문이다.
"""

from __future__ import annotations

import json
import time
from typing import Any, Protocol


class SessionStore(Protocol):
    """세션 스냅샷 저장 계약."""

    def get(self, session_id: str) -> dict[str, Any] | None: ...

    def set(self, session_id: str, snapshot: dict[str, Any]) -> None: ...

    def delete(self, session_id: str) -> None: ...

    @property
    def backend(self) -> str: ...


class InMemorySessionStore:
    """단일 프로세스용 기본 저장소. TTL을 직접 관리한다."""

    def __init__(self, ttl_sec: int = 300) -> None:
        self.ttl_sec = ttl_sec
        self._data: dict[str, tuple[float, dict[str, Any]]] = {}

    @property
    def backend(self) -> str:
        return "memory"

    def _purge_expired(self) -> None:
        now = time.monotonic()
        expired = [k for k, (exp, _) in self._data.items() if exp <= now]
        for k in expired:
            del self._data[k]

    def get(self, session_id: str) -> dict[str, Any] | None:
        self._purge_expired()
        entry = self._data.get(session_id)
        return None if entry is None else entry[1]

    def set(self, session_id: str, snapshot: dict[str, Any]) -> None:
        self._purge_expired()
        self._data[session_id] = (time.monotonic() + self.ttl_sec, snapshot)

    def delete(self, session_id: str) -> None:
        self._data.pop(session_id, None)

    def __len__(self) -> int:
        self._purge_expired()
        return len(self._data)


class RedisSessionStore:
    """다중 워커 환경용. TTL은 Redis가 관리한다."""

    def __init__(self, url: str, ttl_sec: int = 300, key_prefix: str = "phishstress:sess:") -> None:
        import redis  # 선택 의존성 — extras [redis]

        self._client = redis.Redis.from_url(url, decode_responses=True)
        self._client.ping()  # 연결 실패 시 여기서 예외 → 호출부가 폴백
        self.ttl_sec = ttl_sec
        self.key_prefix = key_prefix

    @property
    def backend(self) -> str:
        return "redis"

    def _key(self, session_id: str) -> str:
        return f"{self.key_prefix}{session_id}"

    def get(self, session_id: str) -> dict[str, Any] | None:
        raw = self._client.get(self._key(session_id))
        return None if raw is None else json.loads(raw)

    def set(self, session_id: str, snapshot: dict[str, Any]) -> None:
        self._client.setex(self._key(session_id), self.ttl_sec, json.dumps(snapshot))

    def delete(self, session_id: str) -> None:
        self._client.delete(self._key(session_id))


def build_session_store(redis_url: str | None, ttl_sec: int = 300) -> SessionStore:
    """REDIS_URL이 있으면 Redis, 없거나 연결 실패면 인메모리."""
    if not redis_url:
        return InMemorySessionStore(ttl_sec=ttl_sec)
    try:
        return RedisSessionStore(redis_url, ttl_sec=ttl_sec)
    except Exception:  # noqa: BLE001 — 어떤 실패든 서비스는 떠야 한다
        return InMemorySessionStore(ttl_sec=ttl_sec)
