"""축 C — 전사(STT) 오류 주입.

**왜 필요한가.** 이 프로젝트의 파이프라인 한가운데에 train-serving skew가 있다.
텍스트 판별기는 **사람이 정제한 전사문**으로 학습하는데, 서빙 시 입력은
**faster-whisper가 8kHz 압축 통화에서 뽑은 전사문**이다. 조사가 빠지고, 숫자가
틀리고, 고유명사가 무너진다.

선행 연구는 이 축을 측정하지 않았다. 정제된 전사문에서 F1 99.31%를 보고할 뿐이다.
**축 C의 Gap은 본 프로젝트의 독자적 기여다.**

Day 10에 실제 STT를 붙이면 실측 WER로 강도를 보정한다. 그때까지는 한국어 STT의
알려진 오류 유형을 규칙으로 모사한다.
"""

from __future__ import annotations

import random
import re

from ..data.loaders import Example, normalize_text
from .base import Transform, TransformMeta
from .hangul import CONSONANT_CONFUSION, FINAL_CONFUSION, VOWEL_CONFUSION, confuse_syllable

# 한국어 조사 — STT가 가장 자주 흘리는 요소.
PARTICLES: tuple[str, ...] = (
    "으로",
    "에서",
    "부터",
    "까지",
    "에게",
    "한테",
    "이라고",
    "라고",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "에",
    "로",
    "의",
    "와",
    "과",
    "도",
    "만",
)


class WordDropout(Transform):
    """C1 — 어절 삭제.

    strength가 목표 WER에 대응한다. 실제 STT의 삭제 오류(deletion)를 모사한다.
    """

    meta = TransformMeta(
        axis="C",
        key="C1",
        name="어절 삭제 (WER 모사)",
        label_preserving=True,
        description="strength 비율의 어절을 삭제한다. STT 삭제 오류에 해당한다.",
    )

    def __init__(self, seed: int = 20260816, strength: float = 0.10) -> None:
        super().__init__(seed=seed, strength=strength)

    def _apply(self, ex: Example, rng: random.Random) -> Example | None:
        words = ex.text.split()
        if len(words) < 4:
            return None
        kept = [w for w in words if rng.random() >= self.strength]
        if not kept:
            return None
        return Example(uid=ex.uid, text=" ".join(kept), label=ex.label)


class PhoneticConfusion(Transform):
    """C2 — 발음 혼동 치환.

    STT의 치환 오류(substitution)를 모사한다. 조음이 가까운 자음, 변별이 약한 모음,
    비음 받침을 서로 바꾼다.
    """

    meta = TransformMeta(
        axis="C",
        key="C2",
        name="발음 혼동 치환",
        label_preserving=True,
        description="조음이 가까운 자모를 치환한다. STT 치환 오류에 해당한다.",
    )

    def __init__(self, seed: int = 20260816, strength: float = 0.10) -> None:
        super().__init__(seed=seed, strength=strength)

    _TABLES = (VOWEL_CONFUSION, CONSONANT_CONFUSION, FINAL_CONFUSION)

    def _apply(self, ex: Example, rng: random.Random) -> Example | None:
        out = []
        for ch in ex.text:
            if rng.random() < self.strength:
                out.append(confuse_syllable(ch, rng, self._TABLES))
            else:
                out.append(ch)
        return Example(uid=ex.uid, text="".join(out), label=ex.label)


class ParticleDropout(Transform):
    """C3 — 조사 손실.

    한국어 STT의 대표 오류다. 조사는 짧고 약하게 발음되어 자주 누락되며,
    누락되면 문장 구조가 흔들려 형태소 기반 특징이 무너진다.
    """

    meta = TransformMeta(
        axis="C",
        key="C3",
        name="조사 손실",
        label_preserving=True,
        description="어절 끝의 조사를 떨어뜨린다. 한국어 STT의 대표 오류 유형.",
    )

    def __init__(self, seed: int = 20260816, strength: float = 0.35) -> None:
        super().__init__(seed=seed, strength=strength)

    def _apply(self, ex: Example, rng: random.Random) -> Example | None:
        words = ex.text.split()
        if len(words) < 4:
            return None
        out = []
        for w in words:
            if rng.random() < self.strength:
                for p in PARTICLES:  # 긴 조사부터 검사되도록 정렬되어 있다
                    if len(w) > len(p) and w.endswith(p):
                        w = w[: -len(p)]
                        break
            out.append(w)
        text = normalize_text(" ".join(out))
        return Example(uid=ex.uid, text=text, label=ex.label) if text else None


class NumericCorruption(Transform):
    """C4 — 숫자 오인식.

    계좌번호·금액은 보이스피싱 판별에서 의미가 큰데 STT가 가장 자주 틀리는 부분이다.
    자릿수 변경, 숫자 치환, 자릿수 누락을 모사한다.
    """

    meta = TransformMeta(
        axis="C",
        key="C4",
        name="숫자 오인식",
        label_preserving=True,
        description="숫자 치환·자릿수 누락. 계좌번호·금액 전사 오류를 모사한다.",
    )

    _NUM = re.compile(r"\d+")

    def __init__(self, seed: int = 20260816, strength: float = 0.5) -> None:
        super().__init__(seed=seed, strength=strength)

    def _apply(self, ex: Example, rng: random.Random) -> Example | None:
        def repl(m: re.Match[str]) -> str:
            s = m.group(0)
            if rng.random() > self.strength:
                return s
            digits = list(s)
            mode = rng.randrange(3)
            if mode == 0 and len(digits) > 1:
                del digits[rng.randrange(len(digits))]  # 자릿수 누락
            elif mode == 1:
                i = rng.randrange(len(digits))
                digits[i] = str(rng.randrange(10))  # 숫자 치환
            else:
                digits.insert(rng.randrange(len(digits) + 1), str(rng.randrange(10)))
            return "".join(digits)

        return Example(uid=ex.uid, text=self._NUM.sub(repl, ex.text), label=ex.label)


AXIS_C: tuple[type[Transform], ...] = (
    WordDropout,
    PhoneticConfusion,
    ParticleDropout,
    NumericCorruption,
)
