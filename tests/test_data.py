"""데이터 적재와 분할 검증.

네트워크 없이 도는 테스트다. 28MB 원본을 CI에서 매번 받게 하면 느리고 불안정하다.
실제 데이터가 있는 환경에서만 도는 테스트는 `needs_dataset` 마커로 분리한다.
"""

from __future__ import annotations

import csv
import hashlib

import pytest

from phishstress.data.loaders import (
    Example,
    class_counts,
    load,
    normalize_text,
    sha256_of,
)
from phishstress.data.registry import DatasetSpec, get_spec
from phishstress.data.splits import (
    SplitConfig,
    length_matched_subset,
    length_signal_auc,
    make_splits,
    manifest,
    save_manifest,
)


def make_examples(n_pos=100, n_neg=300, pos_len=500, neg_len=5000) -> list[Example]:
    out = []
    for i in range(n_pos):
        out.append(Example(uid=f"p{i}", text="가" * (pos_len + i), label=1))
    for i in range(n_neg):
        out.append(Example(uid=f"n{i}", text="나" * (neg_len + i), label=0))
    return out


# ------------------------------------------------------------------ loaders


class TestNormalize:
    def test_collapses_whitespace(self):
        assert normalize_text("  가  나\n\n다\t라 ") == "가 나 다 라"

    def test_handles_non_string(self):
        assert normalize_text(12345) == "12345"

    def test_empty(self):
        assert normalize_text("   \n ") == ""


class TestLoad:
    @pytest.fixture
    def fake_dataset(self, tmp_path):
        spec = DatasetSpec(
            key="fake",
            url="file:///dev/null",
            sha256="",
            text_column="transcript",
            label_column="label",
            positive_label=1,
            description="테스트용",
            citation="-",
        )
        path = tmp_path / spec.filename
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "transcript", "label"])
            w.writerow([0, "안녕하세요  반갑습니다", 0])
            w.writerow([1, "계좌로 송금 바랍니다", 1])
            w.writerow([2, "안녕하세요 반갑습니다", 0])  # 정규화 시 0번과 중복
            w.writerow([3, "  ", 0])  # 빈 텍스트
            w.writerow([4, "여러 줄이\n들어간 전사문", 1])
            w.writerow([5, "라벨 없음", ""])
        return spec, tmp_path

    def test_reads_and_normalizes(self, fake_dataset):
        spec, d = fake_dataset
        ex = load(spec, data_dir=d)
        texts = [e.text for e in ex]
        assert "안녕하세요 반갑습니다" in texts
        assert "여러 줄이 들어간 전사문" in texts, "CSV 안의 개행이 처리되어야 한다"

    def test_drops_duplicates_by_default(self, fake_dataset):
        spec, d = fake_dataset
        assert len(load(spec, data_dir=d)) == 3  # 중복/빈값/라벨없음 제거

    def test_can_keep_duplicates(self, fake_dataset):
        spec, d = fake_dataset
        assert len(load(spec, data_dir=d, drop_duplicates=False)) == 4

    def test_labels_are_binarized(self, fake_dataset):
        spec, d = fake_dataset
        assert {e.label for e in load(spec, data_dir=d)} == {0, 1}

    def test_checksum_mismatch_raises(self, fake_dataset, tmp_path):
        spec, d = fake_dataset
        bad = DatasetSpec(**{**spec.__dict__, "sha256": "0" * 64})
        with pytest.raises(RuntimeError, match="체크섬 불일치"):
            load(bad, data_dir=d)

    def test_sha256_matches_hashlib(self, tmp_path):
        p = tmp_path / "x.bin"
        p.write_bytes(b"phishstress")
        assert sha256_of(p) == hashlib.sha256(b"phishstress").hexdigest()


def test_registry_rejects_unknown():
    with pytest.raises(KeyError, match="알 수 없는 데이터셋"):
        get_spec("nope")


def test_class_counts():
    assert class_counts(make_examples(10, 20)) == {0: 20, 1: 10}


# ------------------------------------------------------------------- splits


class TestSplitConfig:
    @pytest.mark.parametrize("kw", [{"train_pct": 0}, {"train_pct": 100}, {"val_pct": 0}])
    def test_rejects_bad_ratios(self, kw):
        with pytest.raises(ValueError):
            SplitConfig(**kw)

    def test_rejects_no_room_for_test(self):
        with pytest.raises(ValueError, match="test가 남습니다"):
            SplitConfig(train_pct=80, val_pct=20)

    def test_test_pct_is_remainder(self):
        assert SplitConfig(train_pct=70, val_pct=15).test_pct == 15


class TestMakeSplits:
    def test_partitions_without_overlap_or_loss(self):
        ex = make_examples()
        s = make_splits(ex)
        uids = [e.uid for _, part in s for e in part]
        assert len(uids) == len(ex), "샘플이 유실되면 안 된다"
        assert len(set(uids)) == len(uids), "분할 간 중복이 있으면 안 된다"

    def test_deterministic_across_runs(self):
        ex = make_examples()
        a, b = make_splits(ex), make_splits(ex)
        for (_, pa), (_, pb) in zip(a, b, strict=True):
            assert [e.uid for e in pa] == [e.uid for e in pb]

    def test_different_seed_gives_different_split(self):
        ex = make_examples()
        a = make_splits(ex, SplitConfig(seed=1))
        b = make_splits(ex, SplitConfig(seed=2))
        assert [e.uid for e in a.test] != [e.uid for e in b.test]

    def test_stratified_positive_rate_is_preserved(self):
        ex = make_examples(n_pos=200, n_neg=600)
        s = make_splits(ex)
        overall = 200 / 800
        for name, part in s:
            rate = sum(e.label for e in part) / len(part)
            assert rate == pytest.approx(overall, abs=0.06), f"{name} 층화 실패"

    def test_stable_under_data_growth(self):
        """해시 버킷팅을 쓴 이유 — 데이터가 늘어도 기존 배정이 유지되어야 한다."""
        base = make_examples(n_pos=50, n_neg=150)
        grown = (
            base
            + make_examples(n_pos=10, n_neg=30)[:0]
            + [Example(uid=f"extra{i}", text="다" * 100, label=i % 2) for i in range(40)]
        )
        a = make_splits(base)
        b = make_splits(grown)
        before = {e.uid for e in a.test}
        after = {e.uid for e in b.test if not e.uid.startswith("extra")}
        assert before == after, "새 데이터 추가가 기존 분할을 흔들면 안 된다"

    def test_unknown_split_name(self):
        with pytest.raises(KeyError, match="알 수 없는 분할"):
            make_splits(make_examples()).get("holdout")


class TestLengthMatched:
    def test_is_balanced(self):
        ex = make_examples(n_pos=100, n_neg=100, pos_len=1000, neg_len=1000)
        sub = length_matched_subset(ex)
        counts = class_counts(sub)
        assert counts[0] == counts[1] > 0, "1:1로 짝지어야 한다"

    def test_neutralizes_length_signal(self):
        """길이정합 슬라이스에서는 길이 단독 AUC가 0.5 근처여야 한다."""
        # 길이 분포가 겹치는 데이터 — 양성 1000~1399, 음성 1200~1599
        ex = make_examples(n_pos=200, n_neg=200, pos_len=1000, neg_len=1200)
        ex += [Example(uid=f"px{i}", text="가" * (1200 + i * 2), label=1) for i in range(200)]
        sub = length_matched_subset(ex, tolerance=0.2)
        assert len(sub) >= 20
        auc = length_signal_auc(sub)
        assert abs(auc - 0.5) <= 0.10, f"길이 신호가 남아 있다: AUC={auc:.3f}"

    def test_returns_empty_when_distributions_are_disjoint(self):
        """겹치지 않으면 어떤 짝짓기도 길이 신호를 못 없앤다 → 빈 결과가 정답이다.

        중화되지 않은 슬라이스를 '길이정합'이라 내놓는 것이 더 나쁘다."""
        ex = make_examples(n_pos=200, n_neg=200, pos_len=1000, neg_len=5000)
        assert length_matched_subset(ex) == []

    def test_empty_when_no_overlap(self):
        ex = make_examples(n_pos=50, n_neg=50, pos_len=10, neg_len=100000)
        assert length_matched_subset(ex, tolerance=0.05) == []

    def test_rejects_bad_residual_bound(self):
        with pytest.raises(ValueError, match="max_residual_auc"):
            length_matched_subset(make_examples(), max_residual_auc=0.5)

    def test_no_reuse_of_negatives(self):
        ex = make_examples(n_pos=100, n_neg=100, pos_len=1000, neg_len=1000)
        sub = length_matched_subset(ex)
        negs = [e.uid for e in sub if e.label == 0]
        assert len(negs) == len(set(negs)), "같은 음성 샘플을 재사용하면 안 된다"

    def test_rejects_bad_tolerance(self):
        with pytest.raises(ValueError, match="tolerance"):
            length_matched_subset(make_examples(), tolerance=0)


class TestManifest:
    def test_records_counts_and_fingerprints(self):
        ex = make_examples()
        cfg = SplitConfig()
        m = manifest(make_splits(ex, cfg), cfg, "fake")
        assert m["seed"] == cfg.seed
        assert set(m["splits"]) == {"train", "val", "test"}
        total = sum(v["n"] for v in m["splits"].values())
        assert total == len(ex)
        for v in m["splits"].values():
            assert v["positive"] + v["negative"] == v["n"]

    def test_fingerprint_detects_change(self):
        ex = make_examples()
        cfg = SplitConfig()
        m1 = manifest(make_splits(ex, cfg), cfg, "fake")
        m2 = manifest(make_splits(ex[:-5], cfg), cfg, "fake")
        assert m1["splits"]["train"]["fingerprint"] != m2["splits"]["train"]["fingerprint"]

    def test_saves_readable_json(self, tmp_path):
        import json

        ex = make_examples()
        cfg = SplitConfig()
        p = save_manifest(manifest(make_splits(ex, cfg), cfg, "fake"), tmp_path / "m.json")
        assert json.loads(p.read_text(encoding="utf-8"))["dataset"] == "fake"
