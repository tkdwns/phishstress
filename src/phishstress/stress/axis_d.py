"""축 D — 하드 네거티브.

**왜 이 축이 생겼나.** Day 3 데이터셋 감사에서 공개 한국어 보이스피싱 데이터의
정상 클래스가 은행 통화가 아니라 **여행·반려동물·음식 잡담**임을 확인했다.

    일상 어휘 포함률   정상 0.949 / 피싱 0.145
    금융 어휘 포함률   정상 0.149 / 피싱 0.886

즉 이 벤치마크가 요구하는 과제는 사실 "은행 사기 대본과 반려동물 수다를 구별하라"이며,
배포 환경에서 진짜 어려운 음성 샘플 — **진짜 은행 상담, 실제 대출 안내, 실제 경찰 통화**
— 은 데이터셋에 아예 없다. 그런 데이터로 학습한 모델은 "금융 어휘가 있는가"만 보면
99%를 받는다. 그리고 실제 배포하면 정상 은행 통화를 전부 차단한다.

축 A(문구 변형)만으로는 이 취약점을 못 건드린다. 문구를 아무리 바꿔도 주제는 그대로이기
때문이다. **진짜 시험은 금융 어휘를 담고 있지만 사기가 아닌 통화를 정상으로 분류하는 것**이며,
축 D가 그 통화를 만든다.

## 두 가지 구성 방식

| 키 | 방식 | 라벨 |
|---|---|---|
| D1 | 잡담 정상 통화에 **무해한 금융 문장**을 섞는다 | 0 유지 |
| D2 | 피싱 전사문에서 **범죄 유도 부분을 제거**한다 | 1 → **0** |

D1은 "모델이 금융 어휘만 보는가"를 직접 묻는다.
D2는 "모델이 주제가 아니라 행위를 보는가"를 묻는다.

## 윤리 (docs/ETHICS.md 3절)

- D1이 삽입하는 문장은 **은행 업무를 설명하는 일상 문장**이다. 사기 요소가 없다.
- D2는 사기 문장을 **생성하지 않고 제거만** 한다. 방향이 반대다.
- D2 결과물은 잔여 유도 표현이 없는지 코드로 검증한다(`_assert_declawed`).
- 두 방식 모두 새로운 사기 시나리오를 만들지 않는다.

## 라벨 타당성에 대한 정직한 한계

D2의 라벨 0은 **근사**다. 범죄 유도 문장을 제거한 통화가 "합법적 금융 상담"과
완전히 같지는 않다. 이 슬라이스는 진짜 은행 녹취의 대체재이며, 그렇게 읽어야 한다.
진짜 콜센터 녹취를 확보하면 교체한다(docs/EXPERIMENTS.md 참조).
"""

from __future__ import annotations

import random
import re

from ..data.loaders import Example, normalize_text
from .base import Transform, TransformMeta

# ---------------------------------------------------------------------------
# D1 — 무해한 금융 문맥 삽입
# ---------------------------------------------------------------------------

# 금융 어휘를 담되 사기 요소가 전혀 없는 일상 문장. 은행 창구에서 오갈 법한 내용이며,
# 금융감독원·은행 공개 안내문 수준의 일반 표현만 사용한다.
#
# 제약: 아래 문장들은 COERCION_MARKERS를 **하나도 포함하지 않아야** 한다
# (`test_benign_sentences_have_no_coercion`). 우리가 D2에서 "유도 표현"이라고
# 정의한 어휘를 D1의 무해 문장에 넣으면 두 축의 라벨 정의가 서로 모순된다.
# 그래서 이체·입금·비밀번호 같은 단어는 여기서 쓰지 않는다 — 일상 금융 대화에서
# 흔한 말이지만, 자기 정의와의 일관성이 D1의 표현력보다 우선한다.
BENIGN_FINANCIAL_SENTENCES: tuple[str, ...] = (
    "어제 은행 지점에 들러서 적금 하나 새로 들었어요.",
    "요즘 은행 금리가 올라서 예금 이자가 좀 나아졌더라고요.",
    "月급 통장을 다른 은행으로 옮길까 고민 중이에요.",
    "카드 명세서 보니까 이번 달에 생각보다 많이 썼더라고요.",
    "주택청약 계좌는 오래 유지하는 게 좋다고 하더라고요.",
    "전세 대출 상담을 받아봤는데 조건이 생각보다 까다롭더라고요.",
    "체크카드 실적 채우려고 편의점에서도 그걸로 결제하고 있어요.",
    "인터넷뱅킹 공동인증서 갱신하는 걸 자꾸 까먹네요.",
    "친구한테 밥값 나눠 냈는데 카드로 금방 정산됐다고 하더라고요.",
    "연말정산 때 쓰려고 카드 사용 내역을 정리하고 있어요.",
    "은행 앱이 새로 바뀌어서 처음엔 좀 헤맸어요.",
    "은행 앱 로그인 정보를 오랜만에 갱신해 뒀어요.",
    "적금 만기가 다음 달이라 어느 은행에 넣을지 알아보는 중이에요.",
    "카드 포인트가 쌓여 있길래 현금으로 전환했어요.",
    "환전 수수료 우대받으려고 미리 은행 창구에 문의했어요.",
)

# 위 목록의 오타 교정 (한자 혼입 방지)
BENIGN_FINANCIAL_SENTENCES = tuple(s.replace("月급", "월급") for s in BENIGN_FINANCIAL_SENTENCES)


class FinancialContextInjection(Transform):
    """D1 — 잡담 정상 통화에 무해한 금융 문장을 섞는다.

    라벨은 0 그대로다. 사기가 아니라 그냥 돈 이야기를 하는 통화이기 때문이다.
    모델이 금융 어휘를 보고 위험도를 올린다면 여기서 오탐이 폭증한다.
    """

    meta = TransformMeta(
        axis="D",
        key="D1",
        name="무해한 금융 문맥 삽입",
        label_preserving=True,
        applies_to=(0,),
        drop_non_applicable=True,
        description="정상 통화에 사기 요소 없는 금융 문장을 섞는다. 어휘 의존 오탐을 드러낸다.",
    )

    def __init__(self, seed: int = 20260816, strength: float = 1.0, n_inject: int = 6) -> None:
        super().__init__(seed=seed, strength=strength)
        self.n_inject = max(1, n_inject)

    _SPLIT = re.compile(r"(?<=[.!?])\s+")

    def _apply(self, ex: Example, rng: random.Random) -> Example | None:
        if ex.label != 0:
            return None  # 정상 통화에만 적용한다
        parts = [p for p in self._SPLIT.split(ex.text) if p.strip()]
        if not parts:
            return None
        picks = rng.sample(
            BENIGN_FINANCIAL_SENTENCES,
            k=min(self.n_inject, len(BENIGN_FINANCIAL_SENTENCES)),
        )
        for sentence in picks:
            parts.insert(rng.randrange(len(parts) + 1), sentence)
        return Example(uid=ex.uid, text=normalize_text(" ".join(parts)), label=0)


# ---------------------------------------------------------------------------
# D2 — 범죄 유도 제거 (무해화)
# ---------------------------------------------------------------------------

# 이 표현들이 들어간 문장을 통째로 제거한다. 금융 어휘 자체는 남긴다 —
# 남기는 것이 목적이다.
COERCION_MARKERS: tuple[str, ...] = (
    # 자금 이동 요구
    "이체",
    "송금",
    "입금",
    "출금",
    "인출",
    "계좌로",
    "안전계좌",
    "보안계좌",
    # 기관 사칭
    "검찰",
    "경찰",
    "금융감독원",
    "수사관",
    "지검",
    "영장",
    "수사",
    "구속",
    # 긴급성·협박
    "즉시",
    "당장",
    "지금 바로",
    "체포",
    "처벌",
    "구속영장",
    "동결",
    "압류",
    # 정보 요구
    "비밀번호",
    "인증번호",
    "보안카드",
    "주민등록번호",
    "카드번호",
    # 앱·원격 설치
    "설치",
    "원격",
    "앱을",
    "어플",
)


class CoercionRemoval(Transform):
    """D2 — 피싱 전사문에서 범죄 유도 문장을 제거해 정상 금융 통화의 대체재를 만든다.

    라벨이 1 → 0으로 **바뀐다**. `label_preserving=False`인 유일한 변환이다.

    남은 텍스트에 유도 표현이 없는지 코드로 검증한다. 검증에 실패하면 그 샘플은
    슬라이스에서 제외한다 — 잘못 라벨링된 샘플을 넣느니 표본이 줄어드는 편이 낫다.
    """

    meta = TransformMeta(
        axis="D",
        key="D2",
        name="범죄 유도 제거 (무해화)",
        label_preserving=False,
        applies_to=(1,),
        drop_non_applicable=True,
        description=(
            "피싱 전사문에서 유도 문장을 제거하고 라벨을 0으로 바꾼다. 주제 의존 오탐을 드러낸다."
        ),
    )

    _SPLIT = re.compile(r"(?<=[.!?])\s+|(?<=니다)\s+|(?<=세요)\s+")

    def __init__(self, seed: int = 20260816, strength: float = 1.0, min_words: int = 30) -> None:
        super().__init__(seed=seed, strength=strength)
        self.min_words = min_words

    @staticmethod
    def _assert_declawed(text: str) -> bool:
        return not any(m in text for m in COERCION_MARKERS)

    def _apply(self, ex: Example, rng: random.Random) -> Example | None:
        if ex.label != 1:
            return None  # 피싱 전사문에만 적용한다
        parts = [p for p in self._SPLIT.split(ex.text) if p.strip()]
        kept = [p for p in parts if not any(m in p for m in COERCION_MARKERS)]

        # **제거가 실제로 일어났을 때만 라벨을 뒤집는다.**
        # 유도 표현이 애초에 하나도 없던 피싱 전사문은 무해화할 것이 없으므로
        # 텍스트가 그대로인 채 라벨만 1→0이 되어 버린다. 그건 라벨 오류다.
        # (이 조건이 없을 때 test 분할에서 34건이 그렇게 새어 들어왔다.)
        if len(kept) == len(parts):
            return None

        text = normalize_text(" ".join(kept))
        if len(text.split()) < self.min_words:
            return None  # 너무 많이 잘려 나가면 평가 가치가 없다
        if not self._assert_declawed(text):
            return None  # 잔여 유도 표현이 있으면 버린다
        return Example(uid=ex.uid, text=text, label=0)


AXIS_D: tuple[type[Transform], ...] = (FinancialContextInjection, CoercionRemoval)
