# PhishStress

**한국어 보이스피싱 판별기 강건성 벤치마크 & 실시간 게이트웨이**

> 보고된 정확도 99%는 깨끗한 조건에서만 성립한다.
> 실제 사기범은 문구를 계속 바꾸고, 통화는 언제나 압축된 8kHz로 들어온다.
> 이 프로젝트는 그 간극을 **측정 가능한 벤치마크**로 만들고, 간극을 좁힌 판별기와
> 이를 상시 감시하는 서빙 파이프라인을 만든다.

[![CI](https://github.com/tkdwns/phishstress/actions/workflows/ci.yml/badge.svg)](https://github.com/tkdwns/phishstress/actions/workflows/ci.yml)

---

## 이 프로젝트가 답하려는 모순

한국어 보이스피싱 탐지 연구는 이미 성능 포화로 보고된다.

| 연구 | 데이터 | 보고 성능 |
|---|---|---|
| Mathematics 11(14), 3217 (2023) | KorCCVi v2 | 정확도 99.32%, F1 99.31% |
| Applied Sciences 15(20), 11170 (2025) | 금감원 2,542건 | 텍스트 99.96%, 음성 99.92% |
| arXiv 2506.06180 (Llama3-8B FT) | 자체 수집 | 정확도 100% |

**그런데 실제 피해액은 2022년 1,451억 원에서 2025년 8,255억 원으로 5.7배 늘었다**
(금융감독원 자료, 2026-08-13 보도. 5년 누적 1조 7,731억 원 / 환급률 22.4%).

답은 같은 논문 안에 있다.

| arXiv 2506.06180 — KoBERT | 정확도 |
|---|---|
| 정상 테스트셋 | 98.52% |
| **적대적 테스트셋** | **56.25%** |
| **Robustness Gap** | **42.27%p** |

문구를 바꾸자 98.52%가 56.25%로 떨어졌다. 이진분류에서 56%는 사실상 무작위다.
음성 쪽도 같다 — 실제 통신 코덱을 통과시키면 기준 모델의 EER이 평균 **5.30%p 증가**한다
(arXiv 2504.12423).

**이 저장소의 최상위 지표는 정확도가 아니라 Robustness Gap이다.**

---

## 현재 상태 — Day 1 / 14

| Day | 산출물 | 상태 |
|---|---|---|
| 1 | Detector 플러그인 계약, 링버퍼, 정책 레이어, WebSocket 게이트웨이, Docker, CI | ✅ 완료 |
| 2 | CI 그린화, 저장소 설정 검증 테스트, 줄바꿈 정규화 | ✅ 완료 |
| 3–4 | 평가 프로토콜, KO-VP-Stress 축 A(적대적 문구)·축 C(전사 오류) | ⬜ |
| 5–7 | 텍스트 판별기 학습, 붕괴 재현, 증강 학습으로 Gap 축소 | ⬜ |
| 8–10 | 축 B(코덱 열화), 음성 판별기, STT 통합 | ⬜ |
| 11–12 | 열화 인지 정책 튜닝, **강건성 회귀 CI 게이트**, 부하 테스트 | ⬜ |
| 13–14 | 데모 UI, 실험 리포트 | ⬜ |

> **성능표는 아직 비어 있다.** Day 1의 목표는 *학습된 모델이 하나도 없는 상태에서*
> 오디오가 들어가면 위험 등급이 돌아오는 전 구간을 완성하는 것이었다.
> 판별기는 `Detector` 계약 뒤의 플러그인이므로, 모델이 들어와도 서빙 코드는 바뀌지 않는다.

### Day 1 검증 결과

```
105 passed              # pytest — Python 3.10 / 3.11 / 3.12 전부
All checks passed!      # ruff check
25 files formatted      # ruff format --check
```

실제 uvicorn 서버 대상 E2E 스모크 (`bench/smoke_client.py`):

```
[1] GET /health        → {'status':'ok','detectors':2,'session_backend':'memory'}
[3] POST /predict/text → score=1.0 conf=0.82
[4] WS /ws/stream
    transcript → grade=BLOCK risk=1.0
    audio 3s   → grade=BLOCK risk=0.9745 degraded=True parts=['audio','text']
    audio 5s   → grade=BLOCK risk=0.9441 degraded=True parts=['audio','text']
[5] closed     → chunks_processed=5, windows_emitted=4, final_grade=BLOCK
```

---

## 빠른 시작

```bash
# Docker (Redis 포함)
docker compose up --build
curl localhost:8000/health

# 또는 로컬
pip install -e ".[dev]"
uvicorn phishstress.serving.app:app --reload
python bench/smoke_client.py        # E2E 스모크
```

API 문서는 http://localhost:8000/docs

---

## 아키텍처

```
WebSocket ──▶ RingBuffer (3s 윈도우 / 1s 홉) ──┬──▶ [audio Detector] ──┐
                                              │                       ├──▶ Policy ──▶ 등급
                              transcript ─────┴──▶ [text Detector] ───┘       │
                                                                       SessionStore
                                                                    (memory | Redis)
```

### 설계 결정 1 — 판별기는 플러그인이다

`Detector.predict(x) -> DetectionResult` 계약만 지키면 TF-IDF든 RoBERTa든 AASIST든 꽂힌다.
덕분에 **모델 학습이 늦어져도 서빙 파이프라인을 먼저 완성·검증할 수 있었다.**
Day 5/Day 9에 학습 모델을 넣을 때 `serving/app.py`는 손대지 않는다.

### 설계 결정 2 — `confidence`가 정책의 입력이다

`DetectionResult`에는 `score`뿐 아니라 **`confidence`** 가 있다.
판별기가 "입력이 열화돼서 자신 없다"고 스스로 신고하면 정책 레이어가 그 발언권을 줄인다.

이미 동작한다. `DummyAudioDetector`는 3.4kHz 초과 대역의 에너지 비율을 실제로 계산해
협대역(전화 코덱) 입력이면 confidence를 떨어뜨린다. 이 코드가 Day 8 코덱 스트레스 축의 씨앗이다.

### 설계 결정 3 — Fusion을 학습하지 않는다

v1 설계에는 학습형 융합(`w1·logit(D1) + w2·logit(D2)`)이 있었다. **폐기했다.**

음성 위조 라벨과 사기 대화 라벨이 **동시에 붙은 데이터가 존재하지 않기 때문이다.**
ASVspoof에는 사기 라벨이 없고 KorCCVi에는 원본 오디오가 없다. 융합 가중치를 학습할
방법이 없으며, 선행 연구(Applied Sciences 2025)조차 이를 못 풀고 0.8/0.2로 손으로 정했다.

대신 명시적 규칙 정책으로 재정의했다 — 신뢰도 가중, 열화 인지 감쇠, EWMA 누적, 히스테리시스.
자세한 논거는 [`docs/RED_TEAM_REVIEW.md`](docs/RED_TEAM_REVIEW.md) C-3 참조.

### 설계 결정 4 — Accuracy를 주지표로 쓰지 않는다

공개 한국어 보이스피싱 데이터는 양성 254건 / 음성 1,123건 수준이다.
**전부 음성으로 찍기만 해도 정확도 81.6%가 나온다.** 이 프로젝트는 PR-AUC와
Recall@FPR=1%, 그리고 무엇보다 **Robustness Gap**을 본다.

---

## WebSocket 프로토콜

**client → server**

```jsonc
{"type": "start", "session_id": "...", "sample_rate": 16000}   // 선택
<binary>                                                        // int16 LE PCM
{"type": "transcript", "text": "..."}                           // 텍스트 채널
{"type": "stop"}
```

**server → client**

```jsonc
{"type": "ready",  "session_id": "...", "config": {...}}
{"type": "risk",   "grade": "SAFE|WARN|BLOCK", "risk": 0.94,
                   "instant_risk": 0.91, "degraded": true,
                   "contributions": {"audio": {...}, "text": {...}}}
{"type": "error",  "message": "..."}
{"type": "closed", "summary": {...}}
```

---

## 설정

| 환경변수 | 기본값 | 설명 |
|---|---|---|
| `REDIS_URL` | (없음) | 미설정/연결 실패 시 인메모리로 자동 폴백 |
| `SESSION_TTL_SEC` | `300` | 세션 스냅샷 TTL |
| `SAMPLE_RATE` | `16000` | 입력 PCM 샘플레이트 |
| `WINDOW_SEC` / `HOP_SEC` | `3.0` / `1.0` | 슬라이딩 윈도우 |
| `POLICY_ALPHA` | `0.3` | EWMA 평활 계수 |
| `WARN_ENTER` / `WARN_EXIT` | `0.60` / `0.45` | WARN 히스테리시스 |
| `BLOCK_ENTER` / `BLOCK_EXIT` | `0.85` / `0.70` | BLOCK 히스테리시스 |

> **임계값은 Day 11에 비용 곡선(미탐:오탐 = 20:1 가정)으로 재산출한다.**
> 현재 값은 자리표시자다.

---

## 한계 (Limitations)

- 판별기가 아직 더미다. 성능 수치는 Day 5 이후에 채워진다.
- 정책 레이어의 가중치·임계값은 학습된 값이 아니라 **손으로 정한 규칙**이다 (설계 결정 3).
- TTD(Time-to-Detection)는 **측정하지 않는다.** 타임스탬프가 있는 실제 통화 데이터가
  존재하지 않아 자체 합성 시 순환 논리가 되기 때문이다. 폐기 사유는 검토서 C-4 참조.
- STT는 미통합(Day 10). 현재 텍스트 채널은 클라이언트가 전사를 넣어줘야 한다.

## 개인정보·윤리

오디오와 전사 텍스트는 **디스크에 쓰지 않는다.** 세션 스토어에 남는 것은
정책 스냅샷(위험도·등급·갱신 횟수) 세 개뿐이며 TTL로 만료된다.
Redis도 영속화를 끈 채로 뜬다. 상세는 [`docs/ETHICS.md`](docs/ETHICS.md).

## 문서

| 문서 | 내용 |
|---|---|
| [`docs/PROJECT_CHARTER.md`](docs/PROJECT_CHARTER.md) | 과제정의서 v2 |
| [`docs/RED_TEAM_REVIEW.md`](docs/RED_TEAM_REVIEW.md) | v1 설계를 폐기한 근거 |
| [`docs/ETHICS.md`](docs/ETHICS.md) | 데이터 출처·라이선스·설계 조치 |

## 참고 문헌

- arXiv [2506.06180](https://arxiv.org/abs/2506.06180) — *Detecting Voice Phishing with Precision: Fine-Tuning Small Language Models*
- arXiv [2504.12423](https://arxiv.org/abs/2504.12423) — *Benchmarking Audio Deepfake Detection Robustness in Real-world Communication Scenarios*
- Applied Sciences 15(20), 11170 (2025) — *A Multimodal Voice Phishing Detection System Integrating Text and Audio Analysis*
- Mathematics 11(14), 3217 (2023) — *Attention-Based 1D CNN-BiLSTM Hybrid Model with FastText for Korean Voice Phishing Detection*

## 라이선스

Apache-2.0
