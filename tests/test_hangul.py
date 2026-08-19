"""한글 자모 처리 검증.

축 A3(표기 교란)과 축 C2(발음 혼동)의 토대다. 여기가 틀리면 스트레스 슬라이스가
조용히 깨진 텍스트를 만들어내고, 그 위에서 잰 Gap은 무의미해진다.
"""

from __future__ import annotations

import pytest

from phishstress.stress.hangul import (
    CHO,
    CONSONANT_CONFUSION,
    FINAL_CONFUSION,
    JONG,
    JUNG,
    SBASE,
    SCOUNT,
    VOWEL_CONFUSION,
    compose,
    confuse_syllable,
    decompose,
    explode,
    is_syllable,
)


class TestRoundTrip:
    def test_all_syllables_round_trip(self):
        """완성형 음절 11,172개 전부 분해→조합이 원본과 같아야 한다."""
        bad = []
        for code in range(SBASE, SBASE + SCOUNT):
            ch = chr(code)
            parts = decompose(ch)
            assert parts is not None
            if compose(*parts) != ch:
                bad.append(ch)
        assert not bad, f"왕복 실패 {len(bad)}건: {bad[:5]}"

    def test_jamo_table_sizes(self):
        assert (len(CHO), len(JUNG), len(JONG)) == (19, 21, 28)
        assert SCOUNT == 19 * 21 * 28

    @pytest.mark.parametrize(
        ("ch", "expected"),
        [("가", ("ㄱ", "ㅏ", "")), ("값", ("ㄱ", "ㅏ", "ㅄ")), ("헐", ("ㅎ", "ㅓ", "ㄹ"))],
    )
    def test_known_decompositions(self, ch, expected):
        assert decompose(ch) == expected


class TestNonSyllable:
    @pytest.mark.parametrize("ch", ["A", "1", " ", "ㄱ", "。", "😀"])
    def test_returns_none(self, ch):
        assert decompose(ch) is None
        assert not is_syllable(ch)

    def test_explode_leaves_non_syllables_alone(self):
        assert explode("ABC 123 !@#") == "ABC 123 !@#"

    def test_explode_mixed(self):
        assert explode("가A") == "ㄱㅏA"


class TestCompose:
    def test_rejects_invalid_jamo(self):
        assert compose("A", "ㅏ") is None
        assert compose("ㄱ", "Z") is None
        assert compose("ㄱ", "ㅏ", "Z") is None

    def test_no_final_by_default(self):
        assert compose("ㄱ", "ㅏ") == "가"


class TestConfuseSyllable:
    def _rng(self, seed=0):
        import random

        return random.Random(seed)

    def test_output_is_single_syllable(self):
        for code in range(SBASE, SBASE + SCOUNT, 97):
            ch = chr(code)
            out = confuse_syllable(ch, self._rng(code), (VOWEL_CONFUSION, CONSONANT_CONFUSION))
            assert len(out) == 1
            assert is_syllable(out), f"{ch} → {out} 가 완성형이 아님"

    def test_leaves_non_syllable_untouched(self):
        assert confuse_syllable("A", self._rng(), (VOWEL_CONFUSION,)) == "A"

    def test_actually_changes_something_across_corpus(self):
        """혼동 표가 실제로 적용되는지 — 전부 원본 그대로면 축 C2가 무의미하다."""
        text = "고객님의 계좌가 범죄에 이용되었습니다"
        changed = sum(
            1
            for i, ch in enumerate(text)
            if confuse_syllable(ch, self._rng(i), (VOWEL_CONFUSION, CONSONANT_CONFUSION)) != ch
        )
        assert changed > 0

    def test_final_confusion_can_drop_batchim(self):
        """받침 탈락도 표현되어야 한다 (STT의 흔한 오류)."""
        seen = {confuse_syllable("각", self._rng(s), (FINAL_CONFUSION,)) for s in range(40)}
        assert len(seen) > 1, "받침 혼동이 전혀 적용되지 않았다"

    def test_deterministic_for_same_seed(self):
        a = confuse_syllable("계", self._rng(7), (VOWEL_CONFUSION,))
        b = confuse_syllable("계", self._rng(7), (VOWEL_CONFUSION,))
        assert a == b


class TestConfusionTables:
    @pytest.mark.parametrize("table", [VOWEL_CONFUSION, CONSONANT_CONFUSION, FINAL_CONFUSION])
    def test_entries_are_valid_jamo(self, table):
        for src, options in table.items():
            assert isinstance(options, tuple) and options
            for opt in options:
                assert opt == "" or opt in CHO or opt in JUNG or opt in JONG.strip()
            assert src not in options, f"{src}가 자기 자신으로 치환됨"
