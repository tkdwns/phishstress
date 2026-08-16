"""세션 저장소 테스트 — TTL, 폴백, 저장 범위."""

from __future__ import annotations

import time

from phishstress.serving.session import InMemorySessionStore, build_session_store


class TestInMemoryStore:
    def test_set_get_delete(self):
        s = InMemorySessionStore()
        s.set("a", {"risk": 0.5, "grade": "WARN", "updates": 3})
        assert s.get("a") == {"risk": 0.5, "grade": "WARN", "updates": 3}
        s.delete("a")
        assert s.get("a") is None

    def test_missing_key_returns_none(self):
        assert InMemorySessionStore().get("nope") is None

    def test_ttl_expires_entry(self):
        s = InMemorySessionStore(ttl_sec=0)
        s.set("a", {"risk": 0.1})
        time.sleep(0.01)
        assert s.get("a") is None
        assert len(s) == 0

    def test_delete_is_idempotent(self):
        s = InMemorySessionStore()
        s.delete("never-existed")  # 예외가 나면 안 된다

    def test_backend_name(self):
        assert InMemorySessionStore().backend == "memory"


class TestBuildSessionStore:
    def test_no_url_gives_memory(self):
        assert build_session_store(None).backend == "memory"

    def test_empty_url_gives_memory(self):
        assert build_session_store("").backend == "memory"

    def test_unreachable_redis_falls_back_to_memory(self):
        """Redis가 없어도 서비스는 떠야 한다 — Day 1의 핵심 요구사항."""
        store = build_session_store("redis://127.0.0.1:6399/0")
        assert store.backend == "memory"
