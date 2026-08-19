"""평가 CLI.

    python -m phishstress.eval                      # 기본 베이스라인 전부
    python -m phishstress.eval --detector dummy-text
    python -m phishstress.eval --split test --json out.json

Day 3의 완료 기준이 이 명령이다. 학습 모델이 없어도 지표 표가 나와야 한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..data import (
    SplitConfig,
    class_counts,
    length_matched_subset,
    load,
    make_splits,
    manifest,
    save_manifest,
)
from ..detectors.base import Detector
from ..detectors.dummy import KeywordTextDetector
from .baselines import ConstantScoreDetector, LengthDetector, OracleDetector, RandomDetector
from .diagnostics import diagnose
from .runner import evaluate

_HEADERS = [
    ("slice", 18),
    ("n", 6),
    ("pos%", 7),
    ("PR-AUC", 8),
    ("ROC-AUC", 8),
    ("R@FPR1%", 9),
    ("Acc", 7),
    ("F1", 7),
    ("다수클래스", 10),
]


def build_detectors(names: list[str], train_texts, train_labels, all_labels) -> list[Detector]:
    available: dict[str, Detector] = {
        "baseline-constant": ConstantScoreDetector(score=0.0),
        "baseline-random": RandomDetector(seed=20260816),
        "baseline-length": LengthDetector().fit(train_texts, train_labels),
        "baseline-oracle": OracleDetector(labels=all_labels),
        "dummy-text": KeywordTextDetector(),
    }
    if not names:
        # oracle은 정답을 보므로 기본 목록에서 뺀다. 필요하면 명시적으로 부른다.
        names = [k for k in available if k != "baseline-oracle"]
    out = []
    for n in names:
        if n not in available:
            raise SystemExit(f"알 수 없는 판별기: {n}\n사용 가능: {sorted(available)}")
        out.append(available[n])
    return out


def print_table(rows: list[dict]) -> None:
    line = " ".join(h.ljust(w) for h, w in _HEADERS)
    print(line)
    print("-" * len(line))
    for r in rows:
        cells = [
            str(r["slice"]).ljust(_HEADERS[0][1]),
            str(r["n"]).ljust(_HEADERS[1][1]),
            f"{r['positive_rate'] * 100:.1f}".ljust(_HEADERS[2][1]),
            f"{r['pr_auc']:.4f}".ljust(_HEADERS[3][1]),
            f"{r['roc_auc']:.4f}".ljust(_HEADERS[4][1]),
            f"{r['recall_at_fpr01']:.4f}".ljust(_HEADERS[5][1]),
            f"{r['accuracy']:.4f}".ljust(_HEADERS[6][1]),
            f"{r['f1']:.4f}".ljust(_HEADERS[7][1]),
            f"{r['majority_accuracy']:.4f}".ljust(_HEADERS[8][1]),
        ]
        print(" ".join(cells))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m phishstress.eval")
    ap.add_argument("--dataset", default="korccvi_v2")
    ap.add_argument("--detector", action="append", default=[], help="반복 지정 가능")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--seed", type=int, default=SplitConfig().seed)
    ap.add_argument("--json", type=Path, default=None, help="결과를 JSON으로 저장")
    ap.add_argument("--write-manifest", type=Path, default=None)
    ap.add_argument("--no-length-matched", action="store_true")
    ap.add_argument(
        "--skip-diagnosis", action="store_true", help="데이터셋 감사를 건너뛴다(권장하지 않음)"
    )
    args = ap.parse_args(argv)

    print(f"데이터셋 적재 중: {args.dataset} …", file=sys.stderr)
    examples = load(args.dataset)
    counts = class_counts(examples)
    print(
        f"  전체 {len(examples)}건 | 양성 {counts[1]} / 음성 {counts[0]} "
        f"(양성비율 {counts[1] / len(examples):.3f})",
        file=sys.stderr,
    )

    if not args.skip_diagnosis:
        d = diagnose(examples)
        print("\n── 데이터셋 감사 ──────────────────────────────────────────")
        print(
            f"  길이 단독 AUC        {d.length_auc:.4f}   "
            f"(0.5면 무정보 | 분리도 {d.length_separability:.3f})"
        )
        print(
            f"  길이 임계값 정확도   {d.length_ceiling_accuracy:.4f}   "
            f"(다수클래스 {max(d.positive_rate, 1 - d.positive_rate):.4f})"
        )
        print(
            f"  금융 어휘 포함률     피싱 {d.financial_hit_positive:.3f} / "
            f"정상 {d.financial_hit_negative:.3f}"
        )
        print(
            f"  일상 어휘 포함률     피싱 {d.casual_hit_positive:.3f} / "
            f"정상 {d.casual_hit_negative:.3f}"
        )
        for w in d.warnings:
            print(f"  ⚠ {w}")
        print("─────────────────────────────────────────────────────────\n")

    cfg = SplitConfig(seed=args.seed)
    splits = make_splits(examples, cfg)
    mf = manifest(splits, cfg, args.dataset)
    for name, info in mf["splits"].items():
        print(
            f"  {name:5s} n={info['n']:5d} 양성={info['positive']:4d} "
            f"({info['positive_rate']:.3f}) fp={info['fingerprint']}",
            file=sys.stderr,
        )
    if args.write_manifest:
        save_manifest(mf, args.write_manifest)
        print(f"  매니페스트 저장: {args.write_manifest}", file=sys.stderr)

    target = splits.get(args.split)
    slices = {args.split: target}
    if not args.no_length_matched:
        lm = length_matched_subset(target)
        if lm:
            slices[f"{args.split}+lenmatch"] = lm
            print(f"  길이정합 슬라이스: {len(lm)}건", file=sys.stderr)

    detectors = build_detectors(
        args.detector,
        [e.text for e in splits.train],
        [e.label for e in splits.train],
        {e.text: e.label for e in examples},
    )

    payload = {"dataset": args.dataset, "manifest": mf, "reports": []}
    if not args.skip_diagnosis:
        payload["diagnosis"] = diagnose(examples).to_dict()
    for det in detectors:
        report = evaluate(det, slices)
        print(f"\n■ {det.name}")
        print_table([r.to_dict() for r in report.results])
        if len(report.results) == 2:
            gap = report.gap(report.results[0].slice_name, report.results[1].slice_name)
            if gap == gap:
                print(f"  → 길이정합 Gap (R@FPR1%): {gap:+.4f}")
        payload["reports"].append(report.to_dict())

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nJSON 저장: {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
