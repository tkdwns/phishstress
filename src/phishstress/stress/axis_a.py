"""축 A — 적대적 문구 변형.

**근거.** arXiv 2506.06180은 KoBERT가 정상 테스트셋에서 98.52%인데 적대적
테스트셋에서 56.25%로 떨어짐을 보고했다. 42.27%p 붕괴, 이진분류에서 56%는
사실상 무작위다. 이 축은 그 붕괴를 우리 손으로 재현 가능하게 만든다.

**윤리적 제약** (docs/ETHICS.md 3절):
- 새 사기 시나리오를 만들지 않는다. 이미 라벨이 붙은 공개 데이터의 **표면형만** 바꾼다.
- 자유 텍스트를 받아 "더 그럴듯한 피싱 문구"를 만들어주는 인터페이스는 없다.
- 여기 쓰인 치환어는 한국어 화자면 누구나 아는 일반 어휘 변형이며, 새로운 공격
  능력을 주지 않는다. 규칙을 공개하는 편이 방어 연구의 재현성에 이롭다고 판단했다.

**LLM을 쓰지 않는 이유.** 재현성이다. 오프라인 규칙 변환은 네트워크·모델 버전과
무관하게 같은 결과를 낸다. Gap 수치가 API 응답 변화로 흔들리면 안 된다.
"""

from __future__ import annotations

import random
import re

from ..data.loaders import Example, normalize_text
from .base import Transform, TransformMeta
from .hangul import VISUAL_CONFUSION, confuse_syllable, explode

# ---------------------------------------------------------------------------
# A1 — 동의어·완곡어 치환
# ---------------------------------------------------------------------------

# 일반 어휘 수준의 동의 관계. 사전을 찾으면 나오는 것들만 담았다.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "계좌": ("통장", "구좌", "어카운트"),
    "이체": ("송부", "옮김", "넘김"),
    "송금": ("부침", "보냄", "쏨"),
    "입금": ("넣기", "예치", "넣음"),
    "대출": ("융자", "차입", "빌림"),
    "금융": ("파이낸스", "재정", "자금"),
    "은행": ("뱅크", "지점", "금고"),
    "카드": ("플라스틱", "결제수단"),
    "검찰": ("지검", "수사기관", "공안"),
    "경찰": ("치안기관", "형사"),
    "수사": ("조사", "내사", "탐문"),
    "명의": ("이름", "성함", "네임"),
    "개인정보": ("신상자료", "인적사항", "프로필"),
    "확인": ("체크", "조회", "대조"),
    "고객님": ("선생님", "회원님", "그쪽"),
    "지금": ("당장", "곧바로", "이제"),
    "필요": ("요망", "소요", "긴요"),
}


class SynonymSubstitution(Transform):
    meta = TransformMeta(
        axis="A",
        key="A1",
        name="동의어·완곡어 치환",
        label_preserving=True,
        applies_to=(1,),
        description="핵심 어휘를 사전적 동의어로 바꾼다. 키워드 매칭 의존도를 드러낸다.",
    )

    def _apply(self, ex: Example, rng: random.Random) -> Example | None:
        text = ex.text
        for word, options in SYNONYMS.items():
            if word not in text:
                continue
            if rng.random() > self.strength:
                continue
            text = text.replace(word, options[rng.randrange(len(options))])
        return Example(uid=ex.uid, text=text, label=ex.label)


# ---------------------------------------------------------------------------
# A2 — 우회 표현 (직접 지시 → 간접 화행)
# ---------------------------------------------------------------------------

CIRCUMLOCUTION: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("하세요", ("하시면 됩니다", "하시는 게 좋겠습니다", "해 주시겠어요")),
    ("하십시오", ("부탁드립니다", "해 주시면 감사하겠습니다")),
    ("해야 합니다", ("하시는 편이 낫습니다", "권해 드립니다")),
    ("합니다", ("하게 됩니다", "하는 상황입니다")),
    ("입니다", ("이라고 보시면 됩니다", "인 셈입니다")),
    ("주세요", ("주시면 좋겠습니다", "주실 수 있을까요")),
)


class Circumlocution(Transform):
    meta = TransformMeta(
        axis="A",
        key="A2",
        name="우회 표현",
        label_preserving=True,
        applies_to=(1,),
        description="직접 지시를 간접 화행으로 바꾼다. 명령형 패턴 의존도를 드러낸다.",
    )

    def _apply(self, ex: Example, rng: random.Random) -> Example | None:
        text = ex.text
        for src, options in CIRCUMLOCUTION:
            if src not in text or rng.random() > self.strength:
                continue
            text = text.replace(src, options[rng.randrange(len(options))])
        return Example(uid=ex.uid, text=text, label=ex.label)


# ---------------------------------------------------------------------------
# A3 — 표기 교란
# ---------------------------------------------------------------------------


class OrthographicNoise(Transform):
    """띄어쓰기 교란, 자모 분리, 시각적 유사 자모 치환.

    사람은 읽는 데 지장이 없지만 토크나이저는 전혀 다른 토큰을 본다.
    실제 스팸에서 오래 쓰인 회피 기법이다.
    """

    meta = TransformMeta(
        axis="A",
        key="A3",
        name="표기 교란",
        label_preserving=True,
        applies_to=(1,),
        description="띄어쓰기 삽입/제거, 자모 분리, 유사 자모 치환. 토큰화 의존도를 드러낸다.",
    )

    def __init__(self, seed: int = 20260816, strength: float = 0.15) -> None:
        super().__init__(seed=seed, strength=strength)

    def _apply(self, ex: Example, rng: random.Random) -> Example | None:
        chars = list(ex.text)
        out: list[str] = []
        for ch in chars:
            if ch.isspace():
                # 공백 제거
                if rng.random() < self.strength * 0.5:
                    continue
                out.append(ch)
                continue
            r = rng.random()
            if r < self.strength * 0.25:
                # 자모 분리 — 완성형을 풀어쓴다
                out.append(explode(ch))
            elif r < self.strength * 0.5:
                # 시각적 유사 자모 치환
                out.append(confuse_syllable(ch, rng, (VISUAL_CONFUSION,)))
            elif r < self.strength * 0.65:
                # 어중 공백 삽입
                out.append(ch)
                out.append(" ")
            else:
                out.append(ch)
        return Example(uid=ex.uid, text="".join(out), label=ex.label)


# ---------------------------------------------------------------------------
# A4 — 숫자·금액 표기 변형
# ---------------------------------------------------------------------------

_SINO = ("영", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구")
_UNITS = ((100_000_000, "억"), (10_000, "만"), (1_000, "천"), (100, "백"), (10, "십"))


def sino_korean(n: int) -> str:
    """아라비아 숫자를 한글 수사로. 0~99,999,999 범위에서 동작한다."""
    if n == 0:
        return "영"
    if n < 0 or n >= 1_000_000_000:
        return str(n)
    out = []
    for value, unit in _UNITS:
        q, n = divmod(n, value)
        if q:
            out.append(("" if q == 1 and value < 10_000 else sino_korean(q)) + unit)
    if n:
        out.append(_SINO[n])
    return "".join(out)


class NumericReformat(Transform):
    meta = TransformMeta(
        axis="A",
        key="A4",
        name="숫자·금액 표기 변형",
        label_preserving=True,
        applies_to=(1,),
        description="아라비아 숫자 ↔ 한글 수사, 자릿점 삽입. 숫자 패턴 의존도를 드러낸다.",
    )

    _NUM = re.compile(r"\d{1,9}")

    def _apply(self, ex: Example, rng: random.Random) -> Example | None:
        def repl(m: re.Match[str]) -> str:
            if rng.random() > self.strength:
                return m.group(0)
            n = int(m.group(0))
            mode = rng.randrange(3)
            if mode == 0:
                return sino_korean(n)
            if mode == 1 and n >= 1000:
                return f"{n:,}"
            return " ".join(m.group(0))  # 자릿수 사이 공백

        return Example(uid=ex.uid, text=self._NUM.sub(repl, ex.text), label=ex.label)


# ---------------------------------------------------------------------------
# A5 — 화계 변환
# ---------------------------------------------------------------------------

REGISTER_SHIFT: tuple[tuple[str, str], ...] = (
    ("습니다", "어요"),
    ("십니다", "세요"),
    ("합니다", "해요"),
    ("입니다", "이에요"),
    ("드립니다", "드려요"),
    ("하십시오", "하세요"),
    ("됩니다", "돼요"),
    ("있습니다", "있어요"),
    ("없습니다", "없어요"),
)


class RegisterShift(Transform):
    meta = TransformMeta(
        axis="A",
        key="A5",
        name="화계 변환",
        label_preserving=True,
        applies_to=(1,),
        description="격식체(하십시오체) ↔ 비격식체(해요체). 문체 의존도를 드러낸다.",
    )

    def _apply(self, ex: Example, rng: random.Random) -> Example | None:
        text = ex.text
        for src, dst in REGISTER_SHIFT:
            if src not in text or rng.random() > self.strength:
                continue
            text = text.replace(src, dst)
        return Example(uid=ex.uid, text=text, label=ex.label)


# ---------------------------------------------------------------------------
# A6 — 어순 교란 (문장 단위)
# ---------------------------------------------------------------------------


class SentenceShuffle(Transform):
    """문장 순서를 국소적으로 섞는다.

    통화 내용의 집합은 그대로이고 순서만 바뀐다. 사기 여부라는 라벨은 유지된다
    (어떤 문장들이 오갔는지가 판단 근거이지 순서가 아니다).
    인접 교환만 허용해 담화 흐름이 완전히 무너지지 않게 한다.
    """

    meta = TransformMeta(
        axis="A",
        key="A6",
        name="어순 교란",
        label_preserving=True,
        applies_to=(1,),
        description="인접 문장을 국소적으로 교환한다. 위치·순서 의존도를 드러낸다.",
    )

    _SPLIT = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, seed: int = 20260816, strength: float = 0.3) -> None:
        super().__init__(seed=seed, strength=strength)

    def _apply(self, ex: Example, rng: random.Random) -> Example | None:
        parts = [p for p in self._SPLIT.split(ex.text) if p.strip()]
        if len(parts) < 2:
            # 문장 구분자가 없으면 어절 단위 국소 교환으로 대체
            parts = ex.text.split(" ")
            if len(parts) < 4:
                return None
        for i in range(len(parts) - 1):
            if rng.random() < self.strength:
                parts[i], parts[i + 1] = parts[i + 1], parts[i]
        return Example(uid=ex.uid, text=normalize_text(" ".join(parts)), label=ex.label)


AXIS_A: tuple[type[Transform], ...] = (
    SynonymSubstitution,
    Circumlocution,
    OrthographicNoise,
    NumericReformat,
    RegisterShift,
    SentenceShuffle,
)
