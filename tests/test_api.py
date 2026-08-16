"""게이트웨이 E2E 테스트.

Day 1의 완료 기준을 그대로 검증한다:
"웹소켓에 오디오를 흘리면 위험 등급이 돌아온다" — 학습된 모델 없이.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from phishstress.detectors.base import DetectorRegistry
from phishstress.detectors.dummy import ConstantDetector, DummyAudioDetector, KeywordTextDetector
from phishstress.policy.risk import PolicyConfig
from phishstress.serving.app import create_app
from phishstress.serving.config import AppConfig
from phishstress.serving.stream import StreamConfig

SR = 8000  # 테스트를 빠르게 하려고 낮춘다. 프로덕션 기본값은 16000.
STREAM = StreamConfig(sample_rate=SR, window_sec=1.0, hop_sec=0.5)


def make_client(registry: DetectorRegistry | None = None, policy: PolicyConfig | None = None):
    cfg = AppConfig(redis_url=None, stream=STREAM, policy=policy or PolicyConfig())
    if registry is None:
        registry = DetectorRegistry()
        registry.register(KeywordTextDetector())
        registry.register(DummyAudioDetector())
    return TestClient(create_app(config=cfg, registry=registry))


def pcm(n: int, amplitude: float = 0.3, freq: float = 440.0, sr: int = SR) -> bytes:
    t = np.arange(n, dtype=np.float32) / sr
    sig = (np.sin(2 * np.pi * freq * t) * amplitude * 32767).astype("<i2")
    return sig.tobytes()


# ---------------------------------------------------------------- REST


class TestRest:
    def test_health(self):
        with make_client() as c:
            body = c.get("/health").json()
            assert body["status"] == "ok"
            assert body["detectors"] == 2
            assert body["session_backend"] == "memory"

    def test_list_detectors(self):
        with make_client() as c:
            names = {d["name"] for d in c.get("/detectors").json()["detectors"]}
            assert names == {"dummy-text", "dummy-audio"}

    def test_predict_text_default_detector(self):
        with make_client() as c:
            res = c.post(
                "/predict/text",
                json={"text": "금융감독원입니다. 안전계좌로 이체 송금 바랍니다."},
            )
            assert res.status_code == 200
            body = res.json()
            assert body["detector"] == "dummy-text"
            assert body["score"] > 0.5
            assert body["latency_ms"] >= 0.0

    def test_predict_text_neutral(self):
        with make_client() as c:
            body = c.post("/predict/text", json={"text": "내일 회의 몇 시죠"}).json()
            assert body["score"] == pytest.approx(0.0)

    def test_predict_text_unknown_detector_404(self):
        with make_client() as c:
            res = c.post("/predict/text", json={"text": "x", "detector": "nope"})
            assert res.status_code == 404

    def test_predict_text_wrong_modality_400(self):
        with make_client() as c:
            res = c.post("/predict/text", json={"text": "x", "detector": "dummy-audio"})
            assert res.status_code == 400

    def test_predict_text_requires_text_field(self):
        with make_client() as c:
            assert c.post("/predict/text", json={}).status_code == 422

    def test_no_text_detector_returns_503(self):
        reg = DetectorRegistry()
        reg.register(DummyAudioDetector())
        with make_client(registry=reg) as c:
            assert c.post("/predict/text", json={"text": "x"}).status_code == 503


# ---------------------------------------------------------------- WebSocket


class TestWebSocketStream:
    def test_ready_frame_describes_config(self):
        with make_client() as c, c.websocket_connect("/ws/stream") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "ready"
            assert msg["config"]["sample_rate"] == SR
            assert msg["config"]["audio_detectors"] == ["dummy-audio"]
            assert len(msg["session_id"]) == 32

    def test_audio_stream_produces_risk_frames(self):
        """Day 1 완료 기준 — 오디오를 흘리면 등급이 돌아온다."""
        with make_client() as c, c.websocket_connect("/ws/stream") as ws:
            ws.receive_json()  # ready
            ws.send_bytes(pcm(SR * 2))  # 2초 → 윈도우 1.0s/홉 0.5s면 3개
            frames = [ws.receive_json() for _ in range(3)]

            assert all(f["type"] == "risk" for f in frames)
            for f in frames:
                assert f["grade"] in {"SAFE", "WARN", "BLOCK"}
                assert 0.0 <= f["risk"] <= 1.0
                assert "audio" in f["contributions"]
            assert [f["updates"] for f in frames] == [1, 2, 3]

    def test_short_audio_yields_no_frame_yet(self):
        with make_client() as c, c.websocket_connect("/ws/stream") as ws:
            ws.receive_json()
            ws.send_bytes(pcm(SR // 4))  # 0.25초 < 1초 윈도우
            ws.send_json({"type": "stop"})
            # flush로 잔여 윈도우 1개 + closed
            msg = ws.receive_json()
            assert msg["type"] == "risk"
            assert ws.receive_json()["type"] == "closed"

    def test_transcript_message_triggers_text_detection(self):
        with make_client() as c, c.websocket_connect("/ws/stream") as ws:
            ws.receive_json()
            ws.send_json({"type": "transcript", "text": "검찰입니다 안전계좌로 송금 이체 하세요"})
            frame = ws.receive_json()
            assert frame["type"] == "risk"
            assert "text" in frame["contributions"]
            assert frame["contributions"]["text"]["score"] > 0.5

    def test_text_and_audio_combine_in_contributions(self):
        with make_client() as c, c.websocket_connect("/ws/stream") as ws:
            ws.receive_json()
            ws.send_json({"type": "transcript", "text": "안전계좌 이체 송금 검찰"})
            ws.receive_json()
            ws.send_bytes(pcm(SR))
            frame = ws.receive_json()
            assert set(frame["contributions"]) == {"text", "audio"}
            total = sum(v["effective_weight"] for v in frame["contributions"].values())
            assert total == pytest.approx(1.0, abs=1e-6)

    def test_stop_returns_summary(self):
        with make_client() as c, c.websocket_connect("/ws/stream") as ws:
            ws.receive_json()
            ws.send_bytes(pcm(SR * 2))
            for _ in range(3):
                ws.receive_json()
            ws.send_json({"type": "stop"})
            ws.receive_json()  # flush 윈도우
            closed = ws.receive_json()
            assert closed["type"] == "closed"
            assert closed["summary"]["chunks_processed"] == 4
            assert closed["summary"]["final_grade"] in {"SAFE", "WARN", "BLOCK"}

    def test_start_message_is_accepted(self):
        with make_client() as c, c.websocket_connect("/ws/stream") as ws:
            ws.receive_json()
            ws.send_json({"type": "start", "sample_rate": SR})
            ws.send_json({"type": "stop"})
            assert ws.receive_json()["type"] == "closed"

    def test_malformed_json_returns_error_not_crash(self):
        with make_client() as c, c.websocket_connect("/ws/stream") as ws:
            ws.receive_json()
            ws.send_text("{ this is not json")
            err = ws.receive_json()
            assert err["type"] == "error"
            # 연결은 살아있어야 한다
            ws.send_json({"type": "stop"})
            assert ws.receive_json()["type"] == "closed"

    def test_unknown_message_type_returns_error(self):
        with make_client() as c, c.websocket_connect("/ws/stream") as ws:
            ws.receive_json()
            ws.send_json({"type": "teleport"})
            assert ws.receive_json()["type"] == "error"

    def test_odd_length_pcm_returns_error_not_crash(self):
        with make_client() as c, c.websocket_connect("/ws/stream") as ws:
            ws.receive_json()
            ws.send_bytes(b"\x00\x01\x02")
            err = ws.receive_json()
            assert err["type"] == "error"
            assert "짝수" in err["message"]

    def test_oversized_message_rejected(self):
        cfg = AppConfig(redis_url=None, stream=STREAM, max_ws_message_bytes=128)
        reg = DetectorRegistry()
        reg.register(DummyAudioDetector())
        client = TestClient(create_app(config=cfg, registry=reg))
        with client as c, c.websocket_connect("/ws/stream") as ws:
            ws.receive_json()
            ws.send_bytes(pcm(SR))
            assert ws.receive_json()["type"] == "error"

    def test_sessions_are_isolated(self):
        with (
            make_client() as c,
            c.websocket_connect("/ws/stream") as a,
            c.websocket_connect("/ws/stream") as b,
        ):
            id_a = a.receive_json()["session_id"]
            id_b = b.receive_json()["session_id"]
            assert id_a != id_b

            a.send_json({"type": "transcript", "text": "검찰 안전계좌 송금 이체 대출"})
            fa = a.receive_json()
            b.send_json({"type": "transcript", "text": "점심 뭐 먹지"})
            fb = b.receive_json()
            assert fa["risk"] > fb["risk"]

    def test_grade_escalates_with_sustained_high_risk(self):
        """정책이 서빙 경로에서 실제로 동작하는지 — 단위 테스트가 아닌 E2E로 확인."""
        reg = DetectorRegistry()
        reg.register(ConstantDetector(name="always-high", modality="text", score=1.0))
        policy = PolicyConfig(alpha=0.5, warn_enter=0.6, warn_exit=0.45)
        client = make_client(registry=reg, policy=policy)
        with client as c, c.websocket_connect("/ws/stream") as ws:
            ws.receive_json()
            grades = []
            for _ in range(3):
                ws.send_json({"type": "transcript", "text": "무엇이든"})
                grades.append(ws.receive_json()["grade"])
            assert grades[-1] in {"WARN", "BLOCK"}

    def test_no_text_detector_transcript_returns_error(self):
        reg = DetectorRegistry()
        reg.register(DummyAudioDetector())
        with make_client(registry=reg) as c, c.websocket_connect("/ws/stream") as ws:
            ws.receive_json()
            ws.send_json({"type": "transcript", "text": "안녕"})
            assert ws.receive_json()["type"] == "error"


class TestPrivacy:
    def test_session_state_is_deleted_after_close(self):
        """통화가 끝나면 세션 상태가 남아 있으면 안 된다 (docs/ETHICS.md)."""
        app_client = make_client()
        with app_client as c:
            store = c.app.state.sessions
            with c.websocket_connect("/ws/stream") as ws:
                ws.receive_json()
                ws.send_json({"type": "transcript", "text": "계좌 이체"})
                ws.receive_json()
                assert len(store) == 1
            assert len(store) == 0

    def test_risk_frame_does_not_echo_transcript(self):
        """응답에 원문 텍스트가 실려 나가면 안 된다."""
        secret = "제 계좌번호는 110-1234-567890 입니다"
        with make_client() as c, c.websocket_connect("/ws/stream") as ws:
            ws.receive_json()
            ws.send_json({"type": "transcript", "text": secret})
            frame = ws.receive_json()
            assert "110-1234-567890" not in str(frame)
