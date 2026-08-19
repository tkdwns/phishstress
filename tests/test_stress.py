"""KO-VP-Stress 변환 검증.

변환의 **불변식**을 성질 테스트로 못 박는다. 개별 변환의 출력이 예쁜지가 아니라,
어떤 입력이 와도 깨지지 않는지가 중요하다.

핵심 불변식 네 가지:
  1. 결정성 — 같은 (변환, 샘플, seed)는 항상 같은 결과
  2. 라벨 규약 — 축 A/C는 라벨 보존, 축 D2만 1→0
  3. 실효성 — 변환이 실제로 텍스트를 바꾼다 (조용한 무동작 방지)
  4. 안전성 — 어떤 입력에도 예외를 던지거나 빈 문자열을 내지 않는다
"""

from __future__ import annotations

import pytest

from phishstress.data.loaders import Example
from phishstress.eval.diagnostics import FINANCIAL_VOCAB
from phishstress.stress import ALL_TRANSFORMS, build_slice, build_suite, get_transform
from phishstress.stress.axis_a import AXIS_A, sino_korean
from phishstress.stress.axis_c import AXIS_C
from phishstress.stress.axis_d import (
    BENIGN_FINANCIAL_SENTENCES,
    COERCION_MARKERS,
    CoercionRemoval,
    FinancialContextInjection,
)

# 실제 전사문은 평균 4천 자다. 유도 문장과 일상 문장이 섞여 있어야
# D2(유도 제거)가 의미 있는 잔여 텍스트를 남긴다.
PHISH = (
    "여보세요 고객님 안녕하세요. 통화 가능하신 시간이실까요. "
    "서울중앙지검 수사관입니다. 고객님 명의의 계좌가 범죄에 이용되었습니다. "
    "지금 즉시 안전계좌 1234567890번으로 5000000원을 이체하셔야 합니다. "
    "협조하지 않으시면 체포영장이 발부됩니다. 개인정보 확인을 위해 주민등록번호를 알려주세요. "
    "보안을 위해 앱을 설치해 주십시오. 절대 다른 곳에 말하지 마세요. "
    "네 잠시만 기다려 주시겠어요. 담당자에게 연결해 드리겠습니다. "
    "본인 확인 절차가 남아 있어서요. 성함과 생년월일을 말씀해 주시면 됩니다. "
    "네 맞습니다. 그렇게 진행하시면 되고요. 오래 걸리지는 않습니다. "
    "혹시 더 궁금하신 점 있으실까요. 없으시면 이대로 마무리하겠습니다."
)
NORMAL = (
    "저는 여행 다니는 걸 정말 좋아해요. 작년에는 스페인에 다녀왔는데 음식이 너무 맛있었어요. "
    "친구랑 같이 갔는데 사진도 많이 찍었고요. 다음에는 학교 방학 때 유럽 쪽으로 또 가보려고요. "
    "반려동물 키우시나요? 저는 고양이를 한 마리 키우는데 정말 사랑스러워요. "
    "요즘은 날씨가 좋아서 주말마다 운동도 하고 있어요."
)


# 금융 어휘를 담았지만 사기가 아닌 통화. 실제 배포 환경의 어려운 음성 샘플이며,
# Day 3 감사에서 공개 데이터에 이런 샘플이 없다는 것이 이 프로젝트의 출발점이었다.
BENIGN_FINANCIAL = (
    "네 안녕하세요 대출 상담 문의드리려고 연락드렸어요. "
    "지금 쓰고 있는 계좌가 어느 은행인지 확인해 주시겠어요. "
    "금리가 어떻게 되는지 알고 싶어서요. 카드 실적도 같이 봐주시면 좋겠고요. "
    "개인정보 동의서는 어디에 서명하면 되나요. 네 알겠습니다 감사합니다."
)


# 피싱 전사문의 어휘 구성을 샘플마다 다르게 만드는 조각들.
# 실제 전사문은 이질적이다 — 어떤 통화는 검찰을 사칭하고 어떤 통화는 대출을 미끼로 쓴다.
# 모든 샘플이 동일하면 변환이 전부를 균일하게 바꿔 순위가 흔들리지 않고,
# 순위 기반 지표(AUC)로는 열화를 관측할 수 없다.
PHISH_VARIANTS = (
    "저희는 서울중앙지검 수사관입니다. 명의 도용 수사가 진행 중입니다.",
    "고객님 계좌가 대포통장으로 신고되었습니다. 안전계좌로 옮기셔야 합니다.",
    "저금리 대출 상품 안내드립니다. 기존 대출을 먼저 상환하셔야 합니다.",
    "금융감독원입니다. 개인정보 유출이 확인되어 연락드렸습니다.",
    "체포영장이 발부된 상태입니다. 지금 즉시 협조해 주셔야 합니다.",
    "본인 확인을 위해 인증번호를 불러주시면 됩니다.",
    "보안 앱을 설치하시고 원격 지원에 동의해 주세요.",
    "송금 절차를 안내드리겠습니다. 이체는 지금 진행하셔야 합니다.",
)


def corpus() -> list[Example]:
    """피싱 / 잡담 정상 / 금융 정상 세 종류를 섞는다.

    피싱 샘플은 어휘 구성을 서로 다르게 준다(PHISH_VARIANTS 부분집합).
    """
    out = []
    for i in range(40):
        # 샘플마다 다른 변형 조각 2~4개를 붙인다
        picks = [PHISH_VARIANTS[(i + k) % len(PHISH_VARIANTS)] for k in range(2 + i % 3)]
        out.append(Example(f"p{i}", PHISH + " " + " ".join(picks) + f" 통화 번호 {i}.", 1))
        out.append(Example(f"n{i}", NORMAL + f" 그리고 {i}번째 이야기예요.", 0))
    for i in range(20):
        out.append(Example(f"h{i}", BENIGN_FINANCIAL + f" 접수번호 {i}번입니다.", 0))
    return out


ALL_KEYS = [c.meta.key for c in ALL_TRANSFORMS]


# ------------------------------------------------------------- 계약·불변식


class TestContract:
    def test_keys_are_unique(self):
        assert len(ALL_KEYS) == len(set(ALL_KEYS))

    def test_axis_membership(self):
        assert [c.meta.key for c in AXIS_A] == ["A1", "A2", "A3", "A4", "A5", "A6"]
        assert [c.meta.key for c in AXIS_C] == ["C1", "C2", "C3", "C4"]
        assert {c.meta.axis for c in AXIS_A} == {"A"}
        assert {c.meta.axis for c in AXIS_C} == {"C"}

    def test_only_d2_changes_labels(self):
        non_preserving = [c.meta.key for c in ALL_TRANSFORMS if not c.meta.label_preserving]
        assert non_preserving == ["D2"]

    def test_rejects_bad_strength(self):
        with pytest.raises(ValueError, match="strength"):
            get_transform("A1").__class__(strength=1.5)

    def test_get_transform_unknown(self):
        with pytest.raises(KeyError, match="알 수 없는 변환"):
            get_transform("Z9")

    @pytest.mark.parametrize("key", ALL_KEYS)
    def test_describe_is_json_safe(self, key):
        import json

        json.dumps(get_transform(key).describe())


class TestDeterminism:
    @pytest.mark.parametrize("key", ALL_KEYS)
    def test_same_seed_same_output(self, key):
        src = Example("x1", PHISH, 1)
        a = get_transform(key, seed=7).apply(src)
        b = get_transform(key, seed=7).apply(src)
        assert (a is None) == (b is None)
        if a is not None:
            assert a.text == b.text and a.label == b.label

    @pytest.mark.parametrize("key", ["A1", "A3", "C1", "C2"])
    def test_different_seed_different_output(self, key):
        src = Example("x1", PHISH, 1)
        a = get_transform(key, seed=1).apply(src)
        b = get_transform(key, seed=999).apply(src)
        assert a is not None and b is not None
        assert a.text != b.text, "seed가 달라도 같은 결과면 결정성 구현이 잘못됐다"

    @pytest.mark.parametrize("key", ALL_KEYS)
    def test_uid_is_preserved(self, key):
        for src in (Example("keepme", PHISH, 1), Example("keepme", NORMAL, 0)):
            out = get_transform(key).apply(src)
            if out is not None:
                assert out.uid == "keepme"


class TestLabelContract:
    @pytest.mark.parametrize("key", [k for k in ALL_KEYS if k != "D2"])
    def test_label_preserved(self, key):
        t = get_transform(key)
        for src in (Example("p", PHISH, 1), Example("n", NORMAL, 0)):
            out = t.apply(src)
            if out is not None:
                assert out.label == src.label

    def test_d2_flips_label_to_zero(self):
        out = CoercionRemoval(min_words=10).apply(Example("p", PHISH, 1))
        assert out is not None and out.label == 0

    def test_d2_rejects_negatives(self):
        assert get_transform("D2").apply(Example("n", NORMAL, 0)) is None

    def test_d1_rejects_positives(self):
        assert get_transform("D1").apply(Example("p", PHISH, 1)) is None


# 위협 모델이 적용 범위를 정한다(TransformMeta.applies_to).
#  - 축 A는 "사기범이 자기 대본을 고친다" → 양성 샘플에만 적용된다.
#  - 축 C는 "STT가 전사를 망가뜨린다" → 통화 종류를 가리지 않는다.
ATTACKER_KEYS = ["A1", "A2", "A3", "A4", "A5", "A6"]
BROAD_KEYS = ["C1", "C2", "C3"]


class TestEffectiveness:
    """변환이 조용히 아무것도 안 하면 Gap이 0으로 나온다.
    그건 '모델이 강건하다'가 아니라 '시험을 안 했다'이다."""

    @pytest.mark.parametrize("key", BROAD_KEYS)
    def test_broad_transforms_change_most_samples(self, key):
        s = build_slice(get_transform(key), corpus())
        assert len(s) > 0, f"{key}가 아무 샘플도 통과시키지 못했다"
        assert s.change_rate > 0.8, f"{key} 변경률이 {s.change_rate:.3f}로 너무 낮다"

    @pytest.mark.parametrize("key", ATTACKER_KEYS)
    def test_attacker_transforms_change_phishing_text(self, key):
        positives = [e for e in corpus() if e.label == 1]
        s = build_slice(get_transform(key), positives)
        assert s.change_rate > 0.8, f"{key}가 피싱 전사문을 못 바꿨다: {s.change_rate:.3f}"

    @pytest.mark.parametrize("key", ATTACKER_KEYS)
    def test_attacker_transforms_never_touch_normal_calls(self, key):
        """위협 모델: 정상 발신자는 탐지를 회피하려 문장을 고치지 않는다.
        축 A를 정상 통화에도 적용하면 존재하지 않는 상황을 시험하게 된다."""
        negatives = [e for e in corpus() if e.label == 0]
        s = build_slice(get_transform(key), negatives)
        assert s.change_rate == 0.0, f"{key}가 정상 통화를 건드렸다"
        assert len(s) == len(negatives), "정상 통화는 원본 그대로 남아야 한다"

    def test_c4_changes_texts_containing_numbers(self):
        """C4는 숫자에만 작용한다. 숫자가 드문 텍스트에서 변경률이 낮은 건 정상이다."""
        numeric = [
            Example(
                f"x{i}",
                f"계좌번호 {i}12345678 로 {i}50000원 요청 {i}9876 건 {i}4321 번 {i}777",
                1,
            )
            for i in range(40)
        ]
        assert build_slice(get_transform("C4"), numeric).change_rate > 0.9

    def test_d1_changes_all_negatives(self):
        negatives = [e for e in corpus() if e.label == 0]
        assert build_slice(get_transform("D1"), negatives).change_rate == 1.0

    def test_d2_changes_all_it_accepts(self):
        s = build_slice(get_transform("D2"), corpus())
        assert len(s) > 0
        assert s.change_rate == 1.0, "D2는 제거가 일어난 샘플만 통과시켜야 한다"

    def test_synonyms_do_not_contain_their_source(self):
        """검찰→검찰청 같은 치환은 부분문자열 매칭을 전혀 회피하지 못한다.
        축 A1이 조용히 무력해지는 버그였다."""
        from phishstress.stress.axis_a import SYNONYMS

        bad = [(k, o) for k, opts in SYNONYMS.items() for o in opts if k in o]
        assert not bad, f"원본을 포함하는 동의어: {bad}"

    def test_a1_removes_target_keywords(self):
        out = get_transform("A1", seed=3).apply(Example("p", PHISH, 1))
        assert "계좌" not in out.text or "통장" in out.text or "계정" in out.text

    def test_a3_changes_tokenization(self):
        out = get_transform("A3").apply(Example("p", PHISH, 1))
        assert out.text != PHISH

    def test_a4_converts_numbers(self):
        out = get_transform("A4", seed=11).apply(Example("p", PHISH, 1))
        assert "1234567890" not in out.text

    def test_c3_drops_particles(self):
        src = Example("p", "고객님의 계좌가 범죄에 이용되었습니다 지금 이체를 하세요", 1)
        out = get_transform("C3", seed=5).apply(src)
        assert out is not None and out.text != src.text


class TestSafety:
    """어떤 입력에도 터지지 않아야 한다."""

    EDGE_TEXTS = [
        "가",
        "a",
        "1",
        " ".join(["단어"] * 200),
        "!@#$%^&*()",
        "숫자 999999999999 아주 큰 값",
        "이모지 😀 섞임",
        "English mixed 한글 text",
        "\t\n 공백만 아닌 텍스트 \t",
    ]

    @pytest.mark.parametrize("key", ALL_KEYS)
    @pytest.mark.parametrize("text", EDGE_TEXTS)
    def test_no_exception_and_no_empty(self, key, text):
        t = get_transform(key)
        for label in (0, 1):
            out = t.apply(Example("e", text, label))
            if out is not None:
                assert out.text.strip(), f"{key}가 빈 텍스트를 냈다: {text!r}"

    @pytest.mark.parametrize("key", ALL_KEYS)
    def test_handles_very_short_text(self, key):
        get_transform(key).apply(Example("e", "가", 1))  # 예외만 안 나면 된다


class TestSinoKorean:
    @pytest.mark.parametrize(
        ("n", "expected"),
        [(0, "영"), (7, "칠"), (10, "십"), (15, "십오"), (100, "백"), (1234, "천이백삼십사")],
    )
    def test_known_values(self, n, expected):
        assert sino_korean(n) == expected

    def test_large_numbers(self):
        assert sino_korean(50000) == "오만"
        assert sino_korean(120000000) == "일억이천만"

    def test_out_of_range_falls_back_to_digits(self):
        assert sino_korean(10**10) == str(10**10)


# ------------------------------------------------------------------- 축 D


class TestAxisD:
    def test_benign_sentences_contain_financial_vocab(self):
        """축 D1의 요점은 금융 어휘를 넣는 것이다."""
        hits = sum(1 for s in BENIGN_FINANCIAL_SENTENCES if any(w in s for w in FINANCIAL_VOCAB))
        assert hits >= len(BENIGN_FINANCIAL_SENTENCES) * 0.7

    def test_benign_sentences_have_no_coercion(self):
        """삽입 문장에 유도 표현이 섞이면 라벨 0이 거짓이 된다."""
        bad = [(s, m) for s in BENIGN_FINANCIAL_SENTENCES for m in COERCION_MARKERS if m in s]
        assert not bad, f"무해 문장에 유도 표현이 있다: {bad}"

    def test_benign_sentences_have_no_hanja(self):
        for s in BENIGN_FINANCIAL_SENTENCES:
            assert not any("一" <= c <= "鿿" for c in s), f"한자 혼입: {s}"

    def test_d1_injects_financial_vocab_into_negatives(self):
        src = Example("n", NORMAL, 0)
        out = FinancialContextInjection().apply(src)
        assert out is not None and out.label == 0
        assert any(w in out.text for w in FINANCIAL_VOCAB), "금융 어휘가 들어가지 않았다"
        assert "여행" in out.text, "원래 내용이 보존되어야 한다"

    def test_d2_removes_all_coercion_markers(self):
        out = CoercionRemoval(min_words=10).apply(Example("p", PHISH, 1))
        assert out is not None
        remaining = [m for m in COERCION_MARKERS if m in out.text]
        assert not remaining, f"잔여 유도 표현: {remaining}"

    def test_d2_keeps_some_content(self):
        out = CoercionRemoval(min_words=3).apply(Example("p", PHISH, 1))
        assert out is not None and len(out.text.split()) >= 3

    def test_d2_rejects_when_nothing_removed(self):
        """유도 표현이 없던 피싱 전사문은 무해화할 것이 없다.
        텍스트가 그대로인 채 라벨만 1→0이 되면 라벨 오류다."""
        clean = Example("p", "오늘 날씨가 좋네요. " * 20, 1)
        assert CoercionRemoval(min_words=3).apply(clean) is None

    def test_d2_rejects_when_too_short_after_removal(self):
        src = Example("p", "지금 즉시 이체하세요.", 1)
        assert CoercionRemoval(min_words=30).apply(src) is None


# ------------------------------------------------------------------ 스위트


class TestSuite:
    def test_builds_all_axes(self):
        suite = build_suite(corpus(), "test", min_slice_size=1)
        assert {s.axis for s in suite.slices} == {"A", "C", "D"}
        assert len(suite.slices) == len(ALL_TRANSFORMS)

    def test_axis_filter(self):
        suite = build_suite(corpus(), "test", axes=("A",), min_slice_size=1)
        assert {s.axis for s in suite.slices} == {"A"}

    def test_axis_d_slices_are_two_class(self):
        """단일 클래스면 AUC가 정의되지 않는다 — 원본 양성을 섞어야 한다."""
        suite = build_suite(corpus(), "test", axes=("D",), min_slice_size=1)
        for s in suite.slices:
            labels = {e.label for e in s.examples}
            assert labels == {0, 1}, f"{s.key}가 단일 클래스다"
            assert s.mixed_base_positives > 0

    def test_axis_a_slices_keep_both_classes(self):
        """축 A는 양성만 바꾸지만 음성은 통과시켜야 한다 — 안 그러면 단일 클래스가 된다."""
        suite = build_suite(corpus(), "test", axes=("A",), min_slice_size=1)
        for s in suite.slices:
            assert {e.label for e in s.examples} == {0, 1}, f"{s.key}가 단일 클래스다"

    def test_axis_ac_slices_not_mixed(self):
        suite = build_suite(corpus(), "test", axes=("A", "C"), min_slice_size=1)
        assert all(s.mixed_base_positives == 0 for s in suite.slices)

    def test_min_slice_size_filters(self):
        suite = build_suite(corpus(), "test", min_slice_size=10**6)
        assert suite.slices == []

    def test_eval_slices_include_base(self):
        suite = build_suite(corpus(), "test", min_slice_size=1)
        slices = suite.as_eval_slices()
        assert "test" in slices
        assert "test/A1" in slices and "test/D1" in slices

    def test_to_dict_is_json_safe(self):
        import json

        json.dumps(build_suite(corpus(), "test", min_slice_size=1).to_dict())

    def test_by_axis(self):
        suite = build_suite(corpus(), "test", min_slice_size=1)
        assert len(suite.by_axis("A")) == 6
        assert len(suite.by_axis("C")) == 4


class TestGapDirection:
    """스트레스가 실제로 판별기를 흔드는지 — 스위트의 존재 이유.

    순위 기반 지표(AUC)로 검증하지 않는다. 키워드 판별기는 마커 4개에서 포화하므로,
    마커가 6개에서 5개로 줄어도 점수가 1.0 그대로다. 합성 데이터에서 AUC를 억지로
    흔들려 하면 테스트가 실데이터를 흉내 내는 데만 매달리게 된다.

    대신 **메커니즘을 직접 잰다.** 축 A1은 마커를 지우는가, 축 D1은 정상 통화의
    점수를 올리는가. 이게 Gap이 생기는 인과이고, 실데이터에서의 크기는
    docs/EXPERIMENTS.md가 기록한다.
    """

    @staticmethod
    def _marker_count(text: str) -> int:
        from phishstress.detectors.dummy import KeywordTextDetector

        return sum(1 for m in KeywordTextDetector.DEFAULT_MARKERS if m in text)

    def test_a1_strips_markers_from_phishing_text(self):
        """축 A1의 인과: 동의어로 바꾸면 키워드 매칭이 놓친다."""
        positives = [e for e in corpus() if e.label == 1]
        before = sum(self._marker_count(e.text) for e in positives)
        after = sum(
            self._marker_count(e.text) for e in build_slice(get_transform("A1"), positives).examples
        )
        assert after < before, f"마커가 줄지 않았다: {before} → {after}"
        assert after < before * 0.7, f"감소폭이 너무 작다: {before} → {after}"

    def test_a3_breaks_marker_matching(self):
        """축 A3의 인과: 표기를 흐트러뜨리면 부분문자열 매칭이 깨진다."""
        positives = [e for e in corpus() if e.label == 1]
        before = sum(self._marker_count(e.text) for e in positives)
        after = sum(
            self._marker_count(e.text) for e in build_slice(get_transform("A3"), positives).examples
        )
        assert after < before

    def test_d1_raises_scores_on_normal_calls(self):
        """축 D1의 인과: 정상 통화에 금융 어휘가 들어가면 오탐이 생긴다."""
        from phishstress.detectors.dummy import KeywordTextDetector

        det = KeywordTextDetector()
        chitchat = [e for e in corpus() if e.label == 0 and e.uid.startswith("n")]
        before = sum(det.predict(e.text).score for e in chitchat) / len(chitchat)
        injected = build_slice(get_transform("D1"), chitchat).examples
        after = sum(det.predict(e.text).score for e in injected) / len(injected)
        assert after > before, f"오탐 점수가 오르지 않았다: {before:.3f} → {after:.3f}"

    def test_d2_keeps_financial_topic_while_dropping_label(self):
        """축 D2의 인과: 주제는 금융인데 라벨은 정상이다."""
        from phishstress.eval.diagnostics import FINANCIAL_VOCAB

        out = [e for e in build_slice(get_transform("D2"), corpus()).examples if e.label == 0]
        assert out, "D2가 아무것도 만들지 못했다"
        with_finance = sum(1 for e in out if any(w in e.text for w in FINANCIAL_VOCAB))
        assert with_finance / len(out) > 0.3, "금융 주제가 남아 있지 않으면 하드 네거티브가 아니다"

    def test_stress_degrades_recall(self):
        """축 A는 양성만 바꾼다. 따라서 재현율(놓치는 피싱)이 나빠져야 하고,
        오탐(정상 통화)은 그대로여야 한다. 정확도 하나로 보면 두 효과가 섞인다."""
        from phishstress.detectors.dummy import KeywordTextDetector

        det = KeywordTextDetector()
        positives = [e for e in corpus() if e.label == 1]
        before = sum(1 for e in positives if det.predict(e.text).score >= 0.5)
        stressed = build_slice(get_transform("A1"), positives).examples
        after = sum(1 for e in stressed if det.predict(e.text).score >= 0.5)
        assert after < before, f"놓친 피싱이 늘지 않았다: {before} → {after}"
