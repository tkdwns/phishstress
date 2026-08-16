"""PhishStress 실시간 게이트웨이.

Day 1 목표: **학습된 모델이 하나도 없는 상태에서** 오디오가 들어가면 위험 등급이
돌아오는 전 구간을 완성한다. 판별기는 `Detector` 계약 뒤의 플러그인이므로
Day 5/Day 9에 학습 모델로 교체할 때 이 파일은 손대지 않는다.

WebSocket 프로토콜 (client → server)
    {"type": "start", "session_id": "...", "sample_rate": 16000}   선택
    <binary>                                                        int16 LE PCM
    {"type": "transcript", "text": "..."}                           텍스트 채널
    {"type": "stop"}

WebSocket 프로토콜 (server → client)
    {"type": "ready",  "session_id": ..., "config": {...}}
    {"type": "risk",   "grade": "SAFE|WARN|BLOCK", ...}
    {"type": "error",  "message": ...}
    {"type": "closed", "summary": {...}}
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from .. import __version__
from ..detectors.base import DetectionResult, Detector, DetectorRegistry
from ..detectors.dummy import DummyAudioDetector, KeywordTextDetector
from ..policy.risk import RiskPolicy
from .config import AppConfig
from .session import build_session_store
from .stream import RingBuffer, pcm16_to_float32


def build_default_registry() -> DetectorRegistry:
    """Day 1 기본 구성: 더미 판별기 2종.

    Day 5에 `text_tfidf` / `text_roberta`, Day 9에 `audio_aasist`가 여기에 추가된다.
    """
    registry = DetectorRegistry()
    registry.register(KeywordTextDetector())
    registry.register(DummyAudioDetector())
    for d in registry.all():
        d.warmup()
    return registry


# --------------------------------------------------------------------------
# 요청/응답 스키마
# --------------------------------------------------------------------------


class TextPredictRequest(BaseModel):
    text: str = Field(..., description="판별할 통화 전사 텍스트")
    detector: str | None = Field(None, description="사용할 판별기 이름. 생략 시 기본 텍스트 판별기")


class DetectionResponse(BaseModel):
    detector: str
    modality: str
    score: float
    confidence: float
    latency_ms: float
    detail: dict[str, Any]


def _to_response(name: str, modality: str, r: DetectionResult) -> DetectionResponse:
    return DetectionResponse(
        detector=name,
        modality=modality,
        score=round(r.score, 6),
        confidence=round(r.confidence, 6),
        latency_ms=round(r.latency_ms, 3),
        detail=r.detail,
    )


# --------------------------------------------------------------------------
# 애플리케이션 팩토리
# --------------------------------------------------------------------------


def create_app(
    config: AppConfig | None = None,
    registry: DetectorRegistry | None = None,
) -> FastAPI:
    cfg = config or AppConfig.from_env()
    reg = registry if registry is not None else build_default_registry()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.config = cfg
        app.state.registry = reg
        app.state.sessions = build_session_store(cfg.redis_url, cfg.session_ttl_sec)
        yield

    app = FastAPI(
        title="PhishStress Gateway",
        version=__version__,
        description="한국어 보이스피싱 판별기 강건성 벤치마크 & 실시간 게이트웨이",
        lifespan=lifespan,
    )

    # ---- REST ------------------------------------------------------------

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "detectors": len(app.state.registry),
            "session_backend": app.state.sessions.backend,
        }

    @app.get("/detectors")
    async def list_detectors() -> dict[str, Any]:
        return {"detectors": [d.describe() for d in app.state.registry.all()]}

    @app.post("/predict/text", response_model=DetectionResponse)
    async def predict_text(req: TextPredictRequest) -> DetectionResponse:
        from fastapi import HTTPException

        reg_: DetectorRegistry = app.state.registry
        if req.detector:
            try:
                detector = reg_.get(req.detector)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            if detector.modality != "text":
                raise HTTPException(
                    status_code=400, detail=f"{detector.name}은(는) 텍스트 판별기가 아닙니다."
                )
        else:
            candidates = reg_.by_modality("text")
            if not candidates:
                raise HTTPException(status_code=503, detail="등록된 텍스트 판별기가 없습니다.")
            detector = candidates[0]

        result = detector.predict(req.text)
        return _to_response(detector.name, detector.modality, result)

    # ---- WebSocket -------------------------------------------------------

    @app.websocket("/ws/stream")
    async def stream(ws: WebSocket) -> None:
        await ws.accept()
        cfg_: AppConfig = app.state.config
        reg_: DetectorRegistry = app.state.registry
        sessions = app.state.sessions

        session_id = uuid.uuid4().hex
        policy = RiskPolicy(cfg_.policy)
        buffer = RingBuffer(cfg_.stream)

        audio_detectors: list[Detector] = reg_.by_modality("audio")
        text_detectors: list[Detector] = reg_.by_modality("text")

        # 마지막 텍스트 판별 결과를 들고 있다가 오디오 청크와 함께 정책에 넣는다.
        # (Day 10에 STT가 붙으면 이 자리를 실시간 전사 결과가 채운다)
        last_text_result: DetectionResult | None = None
        chunks_processed = 0
        started = False

        await ws.send_json(
            {
                "type": "ready",
                "session_id": session_id,
                "config": {
                    "sample_rate": cfg_.stream.sample_rate,
                    "window_sec": cfg_.stream.window_sec,
                    "hop_sec": cfg_.stream.hop_sec,
                    "audio_detectors": [d.name for d in audio_detectors],
                    "text_detectors": [d.name for d in text_detectors],
                },
            }
        )

        def evaluate(results: dict[str, DetectionResult]) -> dict[str, Any]:
            state = policy.update(results)
            sessions.set(session_id, policy.snapshot())
            payload = {"type": "risk", "session_id": session_id, **state.to_dict()}
            payload["elapsed_sec"] = round(buffer.elapsed_sec, 3)
            return payload

        try:
            while True:
                message = await ws.receive()

                if message["type"] == "websocket.disconnect":
                    break

                # ---- 바이너리: PCM 오디오 ----
                if (data := message.get("bytes")) is not None:
                    if len(data) > cfg_.max_ws_message_bytes:
                        await ws.send_json({"type": "error", "message": "메시지가 너무 큽니다."})
                        continue
                    try:
                        samples = pcm16_to_float32(data)
                    except ValueError as exc:
                        await ws.send_json({"type": "error", "message": str(exc)})
                        continue

                    for chunk in buffer.push(samples):
                        results: dict[str, DetectionResult] = {}
                        if audio_detectors:
                            results["audio"] = audio_detectors[0].predict(chunk)
                        if last_text_result is not None:
                            results["text"] = last_text_result
                        if results:
                            chunks_processed += 1
                            await ws.send_json(evaluate(results))
                    continue

                # ---- 텍스트: 제어 메시지 / 전사 ----
                raw = message.get("text")
                if raw is None:
                    continue

                import json as _json

                try:
                    payload = _json.loads(raw)
                except _json.JSONDecodeError:
                    await ws.send_json({"type": "error", "message": "JSON 파싱 실패"})
                    continue

                mtype = payload.get("type")

                if mtype == "start":
                    started = True
                    continue

                if mtype == "transcript":
                    text = str(payload.get("text", ""))
                    if not text_detectors:
                        await ws.send_json(
                            {"type": "error", "message": "등록된 텍스트 판별기가 없습니다."}
                        )
                        continue
                    last_text_result = text_detectors[0].predict(text)
                    results = {"text": last_text_result}
                    chunks_processed += 1
                    await ws.send_json(evaluate(results))
                    continue

                if mtype == "stop":
                    tail = buffer.flush()
                    if tail is not None and audio_detectors:
                        results = {"audio": audio_detectors[0].predict(tail)}
                        if last_text_result is not None:
                            results["text"] = last_text_result
                        chunks_processed += 1
                        await ws.send_json(evaluate(results))
                    await ws.send_json(
                        {
                            "type": "closed",
                            "session_id": session_id,
                            "summary": {
                                "final_grade": policy.grade.value,
                                "final_risk": round(policy.risk, 4),
                                "chunks_processed": chunks_processed,
                                "windows_emitted": buffer.windows_emitted,
                                "elapsed_sec": round(buffer.elapsed_sec, 3),
                                "started_explicitly": started,
                            },
                        }
                    )
                    break

                await ws.send_json({"type": "error", "message": f"알 수 없는 메시지 유형: {mtype}"})

        except WebSocketDisconnect:
            pass
        finally:
            # 통화가 끝나면 세션 상태를 즉시 지운다 (docs/ETHICS.md)
            sessions.delete(session_id)

    return app


app = create_app()
