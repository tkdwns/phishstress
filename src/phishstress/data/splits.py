"""데이터 분할.

Day 3에 분할을 고정하는 이유: 나중에 분할을 바꾸면 그때까지 산출한 모든 수치가
비교 불가능해진다. 한 번 정하면 프로젝트가 끝날 때까지 건드리지 않는다.

설계 결정 — **난수 셔플이 아니라 해시 버킷팅**을 쓴다.
`random.shuffle(seed)`는 데이터가 한 건이라도 늘면 전체 배정이 뒤집힌다.
해시 버킷팅은 각 샘플의 배정이 그 샘플의 uid에만 의존하므로, 나중에 데이터를
추가해도 기존 샘플은 원래 분할에 그대로 남는다.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from .loaders import Example

SPLIT_NAMES = ("train", "val", "test")


def _bucket(uid: str, seed: int) -> int:
    """uid를 [0, 100) 정수로 안정적으로 사상한다."""
    digest = hashlib.sha256(f"{seed}:{uid}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % 100


@dataclass(frozen=True)
class SplitConfig:
    seed: int = 20260816
    train_pct: int = 70
    val_pct: int = 15
    # test_pct는 나머지

    def __post_init__(self) -> None:
        if not 0 < self.train_pct < 100:
            raise ValueError("train_pct는 (0,100) 범위여야 합니다.")
        if not 0 < self.val_pct < 100:
            raise ValueError("val_pct는 (0,100) 범위여야 합니다.")
        if self.train_pct + self.val_pct >= 100:
            raise ValueError("train_pct + val_pct < 100 이어야 test가 남습니다.")

    @property
    def test_pct(self) -> int:
        return 100 - self.train_pct - self.val_pct


@dataclass
class Splits:
    train: list[Example] = field(default_factory=list)
    val: list[Example] = field(default_factory=list)
    test: list[Example] = field(default_factory=list)

    def get(self, name: str) -> list[Example]:
        if name not in SPLIT_NAMES:
            raise KeyError(f"알 수 없는 분할: {name}. 사용 가능: {SPLIT_NAMES}")
        return getattr(self, name)

    def __iter__(self):
        for name in SPLIT_NAMES:
            yield name, getattr(self, name)


def make_splits(examples: list[Example], config: SplitConfig | None = None) -> Splits:
    """라벨별로 층화하여 해시 버킷팅으로 분할한다."""
    cfg = config or SplitConfig()
    splits = Splits()
    # 층화: 클래스별로 따로 버킷팅해야 소수 클래스가 한쪽에 쏠리지 않는다.
    for label in (0, 1):
        for ex in (e for e in examples if e.label == label):
            b = _bucket(ex.uid, cfg.seed)
            if b < cfg.train_pct:
                splits.train.append(ex)
            elif b < cfg.train_pct + cfg.val_pct:
                splits.val.append(ex)
            else:
                splits.test.append(ex)
    for _, part in splits:
        part.sort(key=lambda e: e.uid)
    return splits


def length_signal_auc(examples: list[Example]) -> float:
    """길이만으로 라벨을 얼마나 맞힐 수 있는지. 0.5면 길이가 아무 정보도 주지 않는다.

    (0.5에서 얼마나 떨어졌는지가 중요하므로 방향은 무시하고 절대편차로 본다.)
    """
    pos = [e.length for e in examples if e.label == 1]
    neg = [e.length for e in examples if e.label == 0]
    if not pos or not neg:
        return float("nan")
    # Mann-Whitney U 통계 = ROC-AUC (동점은 0.5로 처리)
    wins = ties = 0
    # O(n log n) 대신 명료한 O(n^2)이면 충분한 크기(수백 건)에서만 쓴다
    for pv in pos:
        for nv in neg:
            if pv > nv:
                wins += 1
            elif pv == nv:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def _greedy_caliper_match(
    pos: list[Example], neg: list[Example], tolerance: float
) -> list[Example]:
    import bisect

    neg_lengths = [e.length for e in neg]
    used: set[int] = set()
    out: list[Example] = []
    for p in pos:
        i = bisect.bisect_left(neg_lengths, p.length)
        best_idx, best_diff = None, None
        for j in range(max(0, i - 100), min(len(neg), i + 100)):
            if j in used:
                continue
            diff = abs(neg[j].length - p.length)
            if best_diff is None or diff < best_diff:
                best_idx, best_diff = j, diff
        if best_idx is None:
            continue
        if best_diff / max(p.length, neg[best_idx].length, 1) > tolerance:
            continue
        used.add(best_idx)
        out.append(p)
        out.append(neg[best_idx])
    return out


def length_matched_subset(
    examples: list[Example],
    tolerance: float = 0.15,
    max_residual_auc: float = 0.60,
) -> list[Example]:
    """길이 아티팩트를 중화한 1:1 부분집합을 만든다.

    KorCCVi v2에서는 **글자 수만 세도 정확도 96.3%가 나온다** (docs/EXPERIMENTS.md).
    두 클래스가 서로 다른 출처에서 수집되어 길이 분포가 갈라진 탓이다.
    그 위에서 잰 성능은 판별 능력이 아니라 수집 아티팩트를 반영한다.

    알고리즘은 세 단계다.

    1. **겹침 구간으로 제한.** 두 클래스의 길이 범위가 겹치는 구간 밖에서는 어떤
       짝짓기를 해도 길이가 라벨을 그대로 알려준다. 겹치지 않으면 빈 리스트를 낸다.
    2. **캘리퍼 매칭.** 각 양성에 길이가 가장 가까운 미사용 음성을 상대오차
       tolerance 이내에서 짝짓는다.
    3. **잔차 검증.** 만들어진 부분집합에서 길이 단독 AUC를 다시 재고,
       0.5에서 max_residual_auc 이상 벗어나면 tolerance를 좁혀 다시 시도한다.
       끝까지 만족하지 못하면 **빈 리스트를 낸다** — 중화되지 않은 슬라이스를
       "길이정합"이라고 내놓는 것이 아무것도 안 내놓는 것보다 나쁘다.
    """
    if not 0 < tolerance < 1:
        raise ValueError("tolerance는 (0,1) 범위여야 합니다.")
    if not 0.5 < max_residual_auc < 1:
        raise ValueError("max_residual_auc는 (0.5,1) 범위여야 합니다.")

    pos_all = [e for e in examples if e.label == 1]
    neg_all = [e for e in examples if e.label == 0]
    if not pos_all or not neg_all:
        return []

    # 1단계 — 겹침 구간
    lo = max(min(e.length for e in pos_all), min(e.length for e in neg_all))
    hi = min(max(e.length for e in pos_all), max(e.length for e in neg_all))
    if lo > hi:
        return []
    pos = sorted((e for e in pos_all if lo <= e.length <= hi), key=lambda e: e.length)
    neg = sorted((e for e in neg_all if lo <= e.length <= hi), key=lambda e: e.length)
    if not pos or not neg:
        return []

    # 2~3단계 — 매칭 후 잔차 검증, 실패 시 tolerance 축소
    tol = tolerance
    for _ in range(6):
        sub = _greedy_caliper_match(pos, neg, tol)
        if len(sub) >= 4:
            auc = length_signal_auc(sub)
            if auc == auc and abs(auc - 0.5) <= (max_residual_auc - 0.5):
                sub.sort(key=lambda e: e.uid)
                return sub
        tol /= 2.0
    return []


def manifest(splits: Splits, config: SplitConfig, dataset_key: str) -> dict:
    """분할이 나중에 조용히 바뀌지 않았는지 확인할 수 있는 지문."""

    def fingerprint(part: list[Example]) -> str:
        h = hashlib.sha256()
        for e in part:
            h.update(e.uid.encode())
            h.update(b"\x00")
        return h.hexdigest()[:16]

    out = {
        "dataset": dataset_key,
        "seed": config.seed,
        "ratios": {
            "train": config.train_pct,
            "val": config.val_pct,
            "test": config.test_pct,
        },
        "splits": {},
    }
    for name, part in splits:
        pos = sum(e.label for e in part)
        out["splits"][name] = {
            "n": len(part),
            "positive": pos,
            "negative": len(part) - pos,
            "positive_rate": round(pos / len(part), 4) if part else 0.0,
            "fingerprint": fingerprint(part),
        }
    return out


def save_manifest(data: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
