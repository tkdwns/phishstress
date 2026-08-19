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

## 현재 상태 — Day 4 / 14

| Day | 산출물 | 상태 |
|---|---|---|
| 1 | Detector 플러그인 계약, 링버퍼, 정책 레이어, WebSocket 게이트웨이, Docker, CI | ✅ 완료 |
| 2 | CI 그린화, 저장소 설정 검증 테스트, 줄바꿈 정규화 | ✅ 완료 |
| 3 | 데이터 확보, 분할 고정, 평가 프로토콜, **데이터셋 감사** | ✅ 완료 |
| 4 | KO-VP-Stress 축 A(적대적 문구)·축 C(전사 오류)·**축 D(하드 네거티브)** | ✅ 완료 |
| 5–7 | 텍스트 판별기 학습, 붕괴 재현, 증강 학습으로 Gap 축소 | ⬜ |
| 8–10 | 축 B(코덱 열화), 음성 판별기, STT 통합 | ⬜ |
| 11–12 | 열화 인지 정책 튜닝, **강건성 회귀 CI 게이트**, 부하 테스트 | ⬜ |
| 13–14 | 데모 UI, 실험 리포트 | ⬜ |

> **학습 모델 성능표는 아직 비어 있다.** Day 1은 *모델이 하나도 없는 상태에서* 오디오가
> 들어가면 위험 등급이 돌아오는 전 구간을 완성했고, Day 3은 *모델을 만들기 전에* 채점할
> 자를 먼저 만들었다. 판별기는 `Detector` 계약 뒤의 플러그인이므로, 학습 모델이 들어와도
> 서빙 코드와 평가 코드는 바뀌지 않는다.

### 검증 결과

```
467 passed              # pytest — Python 3.10 / 3.11 / 3.12 전부
All checks passed!      # ruff check
47 files formatted      # ruff format --check
```

### Day 3에서 찾은 것 — 이 벤치마크는 생각보다 쉽다

모델을 학습시키기 전에 데이터셋을 감사했고, 결과가 계획을 바꿨다.

| 예측기 | 정확도 | F1 |
|---|---|---|
| 다수클래스 (전부 정상으로 찍기) | 0.763 | 0.000 |
| **글자 수 임계값 하나** | **0.963** | **0.919** |
| 선행 연구 보고값 (Mathematics 11(14), 3217) | 0.9932 | 0.9931 |

**F1 99.31%는 글자 수를 세는 것보다 7.4점 높다.** 정상 통화는 전부 1,740자 이상인데
피싱 전사문 중앙값은 534자다 — 두 클래스가 다른 출처에서 수집된 탓이다.

더 큰 문제는 정상 클래스의 정체다.

| 어휘군 | 피싱 | 정상 |
|---|---|---|
| 금융·수사 어휘 (계좌/이체/검찰…) | 0.886 | **0.149** |
| 일상 어휘 (여행/친구/반려동물…) | 0.145 | **0.949** |

**은행 통화가 아니라 잡담이다.** 이 벤치마크가 요구하는 과제는 사실
"은행 사기 대본과 반려동물 수다를 구별하라"이며, 배포 환경의 어려운 음성 샘플
— 진짜 은행 상담, 실제 대출 안내 — 은 데이터셋에 없다. 교차 확인한 KorCCViD v1.3도 같았다.

그래서 길이가 신호를 주지 못하는 **길이정합 슬라이스**를 만들어 함께 보고한다.

| 판별기 | test ROC-AUC | test+lenmatch ROC-AUC |
|---|---|---|
| 글자 수 베이스라인 | 0.9499 | **0.4961** (동전 던지기) |
| 키워드 베이스라인 | 0.8434 | **1.0000** |

원본 테스트셋은 잘못된 능력에 상을 준다. 전체 분석은
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

### Day 4에서 만든 것 — KO-VP-Stress 스트레스 스위트

Robustness Gap을 재려면 열화 조건이 코드로 있어야 한다. 세 축 12개 변환을 만들었다.

| 축 | 변환 | 무엇을 흔드는가 | 적용 대상 |
|---|---|---|---|
| A | 6개 | 사기범이 문구를 바꾼다 | **양성만** |
| C | 4개 | STT가 전사를 망가뜨린다 | 전체 |
| D | 2개 | 배포 환경의 어려운 음성 샘플 | 음성만 / 양성만 |

```bash
python -m phishstress.stress
```

첫 측정 결과 (test 분할, Recall@FPR 1%):

| 판별기 | 기준 | 최악 축 | 최악 Gap |
|---|---|---|---|
| 키워드 베이스라인 | 0.3667 | A1 동의어 치환 | **+0.3667 → 0.0000** |
| 키워드 베이스라인 | 0.3667 | D2 유도 제거 | **+0.3667 → 0.0000** |
| 글자 수 베이스라인 | 0.8750 | D2 유도 제거 | **+0.7833 → 0.0917** |

**동의어를 바꾸는 것만으로 키워드 판별기가 0이 된다.** 그리고 길이 베이스라인은
축 A/C에 전혀 흔들리지 않다가 하드 네거티브에서 무너진다 —
아티팩트에 기대는 모델은 아티팩트가 깨지는 순간 최악으로 실패한다.

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

# 데이터셋 감사 + 베이스라인 평가 (데이터는 자동으로 내려받아 체크섬 검증)
python -m phishstress.eval

# KO-VP-Stress — 축별 Robustness Gap 측정
python -m phishstress.stress
python -m phishstress.stress --axis D --detector dummy-text   # 하드 네거티브만
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

KorCCVi v2는 양성 692 / 음성 2,232건이다. **전부 정상으로 찍기만 해도 정확도 76.3%,
글자 수만 세면 96.3%가 나온다.** 이 프로젝트는 PR-AUC와 Recall@FPR=1%,
그리고 무엇보다 **Robustness Gap**을 본다.

### 설계 결정 5 — 성능표보다 데이터셋 감사를 먼저 출력한다

`python -m phishstress.eval`은 지표 표를 내기 전에 길이 아티팩트와 주제 분리를 먼저 보고한다.
숫자만 보고 판단하는 일이 없도록 기본 동작으로 두었다. `--skip-diagnosis`로 끌 수는 있지만
권장하지 않는다.

### 설계 결정 6 — 위협 모델을 코드에 박는다

`TransformMeta.applies_to`가 변환의 적용 대상을 명시한다. 축 A는 양성 샘플에만
적용된다 — **사기범은 탐지를 피하려 대본을 고치지만 정상 발신자는 그러지 않는다.**

처음에는 축 A를 모든 샘플에 적용했고, 그러자 스트레스를 걸었더니 정확도가
0.80 → 0.92로 **올라갔다.** 정상 통화의 "계좌"까지 "통장"으로 바뀌어 키워드
판별기가 오히려 정확해진 것이다. 존재하지 않는 상황을 시험하고 있었다.

### 설계 결정 7 — 원본 데이터를 저장소에 넣지 않는다

재현성은 파일을 커밋해서가 아니라 **URL + SHA-256 고정**으로 확보한다.
업스트림이 조용히 파일을 바꾸면 체크섬 검증이 실패하며 멈춘다.
평가 모듈은 numpy만 쓰므로 서빙 컨테이너에 pandas·scikit-learn이 들어가지 않는다.

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

- 판별기가 아직 더미다. 학습 모델 성능은 Day 5 이후에 채워진다.
- 길이정합 슬라이스가 32건으로 작다. 부트스트랩 CI를 함께 보고하지만,
  완전 분리 상황에서는 CI가 불확실성을 표현하지 못한다.
- 공개 데이터에 **하드 네거티브(진짜 은행 상담)가 없다.** 축 D가 대체재를 만들지만,
  D2의 라벨 0은 근사다 — 유도 문장을 제거한 통화가 합법적 금융 상담과 같지는 않다.
  진짜 콜센터 녹취를 확보하면 교체한다.
- 축 A의 문구 변형은 규칙 기반이다. LLM 패러프레이즈는 재현성 때문에 쓰지 않았고,
  따라서 실제 사기범의 창의성을 과소평가할 수 있다.
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
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | 실험 기록 — Day 3 데이터셋 감사 결과 |
| [`docs/ETHICS.md`](docs/ETHICS.md) | 데이터 출처·라이선스·설계 조치 |

## 참고 문헌

- arXiv [2506.06180](https://arxiv.org/abs/2506.06180) — *Detecting Voice Phishing with Precision: Fine-Tuning Small Language Models*
- arXiv [2504.12423](https://arxiv.org/abs/2504.12423) — *Benchmarking Audio Deepfake Detection Robustness in Real-world Communication Scenarios*
- Applied Sciences 15(20), 11170 (2025) — *A Multimodal Voice Phishing Detection System Integrating Text and Audio Analysis*
- Mathematics 11(14), 3217 (2023) — *Attention-Based 1D CNN-BiLSTM Hybrid Model with FastText for Korean Voice Phishing Detection*

## 라이선스

Apache-2.0
