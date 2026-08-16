#!/usr/bin/env python3
"""실제 서버에 붙어보는 E2E 스모크 클라이언트.

TestClient는 ASGI를 인메모리로 호출하므로 실제 네트워크 경로를 검증하지 못한다.
이 스크립트는 uvicorn으로 띄운 서버에 진짜 WebSocket으로 붙어 전 구간을 확인한다.

사용:
    uvicorn phishstress.serving.app:app --port 8000 &
    python bench/smoke_client.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import urllib.request

import numpy as np

try:
    import websockets
except ImportError:  # pragma: no cover
    print("websockets가 필요합니다: pip install websockets", file=sys.stderr)
    raise SystemExit(2) from None


def make_pcm(seconds: float, sample_rate: int, freq: float = 440.0) -> bytes:
    """테스트 톤을 int16 리틀엔디언 PCM으로 만든다."""
    t = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
    sig = np.sin(2 * np.pi * freq * t) * 0.3
    return (sig * 32767).astype("<i2").tobytes()


def check_health(base_url: str) -> dict:
    with urllib.request.urlopen(f"{base_url}/health", timeout=5) as resp:
        return json.loads(resp.read())


async def run(base_url: str, ws_url: str, seconds: float) -> int:
    print("=" * 62)
    print("PhishStress E2E 스모크 테스트")
    print("=" * 62)

    health = check_health(base_url)
    print(f"[1] GET /health            → {health}")
    assert health["status"] == "ok", "health가 ok가 아닙니다"

    with urllib.request.urlopen(f"{base_url}/detectors", timeout=5) as resp:
        detectors = json.loads(resp.read())
    print(f"[2] GET /detectors         → {[d['name'] for d in detectors['detectors']]}")

    req = urllib.request.Request(
        f"{base_url}/predict/text",
        data=json.dumps({"text": "금융감독원입니다 안전계좌로 즉시 이체 송금 바랍니다"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        pred = json.loads(resp.read())
    print(f"[3] POST /predict/text     → score={pred['score']} conf={pred['confidence']}")
    assert pred["score"] > 0.5, "사기 문구 점수가 너무 낮습니다"

    print(f"[4] WS {ws_url}")
    risk_frames = []
    async with websockets.connect(ws_url) as ws:
        ready = json.loads(await ws.recv())
        assert ready["type"] == "ready", ready
        sr = ready["config"]["sample_rate"]
        print(f"    ready: session={ready['session_id'][:8]}… sample_rate={sr}")

        await ws.send(json.dumps({"type": "start"}))

        # 전사 채널 — 사기 문구를 넣어 위험도를 올린다
        await ws.send(
            json.dumps(
                {"type": "transcript", "text": "검찰입니다 안전계좌로 송금 이체 하셔야 합니다"}
            )
        )
        frame = json.loads(await ws.recv())
        risk_frames.append(frame)
        print(f"    transcript → grade={frame['grade']} risk={frame['risk']}")

        # 오디오 채널 — 1초씩 나눠 보내 스트리밍 경로를 실제로 태운다
        chunk = make_pcm(1.0, sr)
        for i in range(int(seconds)):
            await ws.send(chunk)
            while True:
                try:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=0.5))
                except TimeoutError:
                    break
                if msg["type"] == "risk":
                    risk_frames.append(msg)
                    print(
                        f"    audio {i + 1}s → grade={msg['grade']} "
                        f"risk={msg['risk']} degraded={msg['degraded']} "
                        f"parts={list(msg['contributions'])}"
                    )
                elif msg["type"] == "error":
                    print(f"    !! error: {msg['message']}")
                    return 1

        await ws.send(json.dumps({"type": "stop"}))
        closed = None
        while True:
            msg = json.loads(await ws.recv())
            if msg["type"] == "risk":
                risk_frames.append(msg)
            elif msg["type"] == "closed":
                closed = msg
                break

    print(f"[5] closed                 → {closed['summary']}")

    print("-" * 62)
    assert risk_frames, "위험 프레임을 하나도 받지 못했습니다"
    assert all(0.0 <= f["risk"] <= 1.0 for f in risk_frames), "risk 범위 위반"
    assert closed["summary"]["chunks_processed"] == len(risk_frames), "청크 카운트 불일치"
    final = closed["summary"]["final_grade"]
    print(f"통과: 위험 프레임 {len(risk_frames)}개 수신, 최종 등급 {final}")
    print("=" * 62)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--seconds", type=float, default=5)
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    ws = f"ws://{args.host}:{args.port}/ws/stream"
    return asyncio.run(run(base, ws, args.seconds))


if __name__ == "__main__":
    raise SystemExit(main())
