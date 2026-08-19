"""한글 자모 처리.

축 A3(표기 교란)과 축 C2(발음 혼동)가 음절을 자모 단위로 다뤄야 해서 필요하다.
외부 의존성 없이 유니코드 산술로 직접 구현한다 — 서빙 컨테이너에 형태소 분석기
같은 무거운 패키지를 넣지 않겠다는 원칙(설계 결정 6)의 연장이다.

    음절코드 = 0xAC00 + (초성 * 21 + 중성) * 28 + 종성
"""

from __future__ import annotations

SBASE = 0xAC00
LCOUNT, VCOUNT, TCOUNT = 19, 21, 28
NCOUNT = VCOUNT * TCOUNT  # 588
SCOUNT = LCOUNT * NCOUNT  # 11172

CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"


def is_syllable(ch: str) -> bool:
    return SBASE <= ord(ch) < SBASE + SCOUNT


def decompose(ch: str) -> tuple[str, str, str] | None:
    """완성형 음절을 (초성, 중성, 종성)으로 분해한다. 종성이 없으면 빈 문자열."""
    if not is_syllable(ch):
        return None
    idx = ord(ch) - SBASE
    cho, rem = divmod(idx, NCOUNT)
    jung, jong = divmod(rem, TCOUNT)
    return CHO[cho], JUNG[jung], ("" if jong == 0 else JONG[jong])


def compose(cho: str, jung: str, jong: str = "") -> str | None:
    """자모를 완성형 음절로 조합한다. 유효하지 않으면 None."""
    if cho not in CHO or jung not in JUNG:
        return None
    j = 0 if not jong else JONG.find(jong)
    if j < 0:
        return None
    return chr(SBASE + (CHO.index(cho) * VCOUNT + JUNG.index(jung)) * TCOUNT + j)


def explode(text: str) -> str:
    """모든 음절을 자모로 풀어쓴다. 완성형이 아닌 문자는 그대로 둔다."""
    out = []
    for ch in text:
        parts = decompose(ch)
        out.append("".join(parts) if parts else ch)
    return "".join(out)


# ---------------------------------------------------------------------------
# 혼동 쌍
# ---------------------------------------------------------------------------

# 한국어 STT가 실제로 자주 헷갈리는 모음. 발음이 거의 같거나 변별이 약한 짝이다.
VOWEL_CONFUSION: dict[str, tuple[str, ...]] = {
    "ㅐ": ("ㅔ",),
    "ㅔ": ("ㅐ",),
    "ㅒ": ("ㅖ",),
    "ㅖ": ("ㅒ",),
    "ㅚ": ("ㅙ", "ㅞ"),
    "ㅙ": ("ㅚ", "ㅞ"),
    "ㅞ": ("ㅚ", "ㅙ"),
    "ㅓ": ("ㅏ",),
    "ㅜ": ("ㅡ",),
    "ㅡ": ("ㅜ",),
}

# 조음 위치·방법이 가까워 혼동되는 자음.
CONSONANT_CONFUSION: dict[str, tuple[str, ...]] = {
    "ㄱ": ("ㅋ", "ㄲ"),
    "ㅋ": ("ㄱ",),
    "ㄲ": ("ㄱ",),
    "ㄷ": ("ㅌ", "ㄸ"),
    "ㅌ": ("ㄷ",),
    "ㄸ": ("ㄷ",),
    "ㅂ": ("ㅍ", "ㅃ"),
    "ㅍ": ("ㅂ",),
    "ㅃ": ("ㅂ",),
    "ㅈ": ("ㅊ", "ㅉ"),
    "ㅊ": ("ㅈ",),
    "ㅉ": ("ㅈ",),
    "ㅅ": ("ㅆ",),
    "ㅆ": ("ㅅ",),
}

# 종성(받침) 혼동 — 비음끼리, 그리고 받침 탈락.
FINAL_CONFUSION: dict[str, tuple[str, ...]] = {
    "ㄴ": ("ㅇ", "ㅁ"),
    "ㅇ": ("ㄴ", "ㅁ"),
    "ㅁ": ("ㄴ", "ㅇ"),
    "ㄱ": ("", "ㅋ"),
    "ㅂ": ("", "ㅍ"),
    "ㄷ": ("", "ㅅ"),
    "ㅅ": ("ㄷ", ""),
    "ㅆ": ("ㅅ",),
    "ㄹ": ("",),
}

# 시각적으로 비슷해 사람 눈은 속지 않지만 토크나이저는 갈리는 짝 (축 A3용).
VISUAL_CONFUSION: dict[str, tuple[str, ...]] = {
    "ㅗ": ("ㅜ",),
    "ㅜ": ("ㅗ",),
    "ㅏ": ("ㅑ",),
    "ㅓ": ("ㅕ",),
    "ㅡ": ("ㅣ",),
}


def _pick(rng, options: tuple[str, ...]) -> str:
    return options[rng.randrange(len(options))]


def confuse_syllable(ch: str, rng, tables: tuple[dict[str, tuple[str, ...]], ...]) -> str:
    """음절 하나를 혼동 표에 따라 살짝 바꾼다. 바꿀 수 없으면 원본을 돌려준다."""
    parts = decompose(ch)
    if parts is None:
        return ch
    cho, jung, jong = parts
    slots: list[tuple[int, str, tuple[str, ...]]] = []
    for table in tables:
        if table in (VOWEL_CONFUSION, VISUAL_CONFUSION) and jung in table:
            slots.append((1, jung, table[jung]))
        if table is CONSONANT_CONFUSION and cho in table:
            slots.append((0, cho, table[cho]))
        if table is FINAL_CONFUSION and jong and jong in table:
            slots.append((2, jong, table[jong]))
    if not slots:
        return ch
    slot, _, options = slots[rng.randrange(len(slots))]
    repl = _pick(rng, options)
    new = [cho, jung, jong]
    new[slot] = repl
    return compose(new[0], new[1], new[2]) or ch
