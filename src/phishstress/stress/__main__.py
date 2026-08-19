"""KO-VP-Stress CLI — 축별 Robustness Gap 보고.

    python -m phishstress.stress
    python -m phishstress.stress --detector dummy-text --axis A --axis D
    python -m phishstress.stress --json data/day4_stress.json

기준선 (선행 연구):
    텍스트 적대적 Gap  42.27%p  (KoBERT 98.52% → 56.25%, arXiv 2506.06180)
    본 프로젝트 목표    15%p 이내
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..data import SplitConfig, load, make_splits
from ..detectors.base import Detector
from ..detectors.dummy import KeywordTextDetector
from ..eval.baselines import ConstantScoreDetector, LengthDetector, RandomDetector
from ..eval.metrics import bootstrap_ci, recall_at_fpr
from ..eval.runner import evaluate
from .suite import build_suite


def build_detectors(names: list[str], train_texts, train_labels) -> list[Detector]:
    available: dict[str, Detector] = {
        "baseline-constant": ConstantScoreDetector(score=0.0),
        "baseline-random": RandomDetector(seed=20260816),
        "baseline-length": LengthDetector().fit(train_texts, train_labels),
        "dummy-text": KeywordTextDetector(),
    }
    if not names:
        names = ["baseline-length", "dummy-text"]
    missing = [n for n in names if n not in available]
    if missing:
        raise SystemExit(f"알 수 없는 판별기: {missing}\n사용 가능: {sorted(available)}")
    return [available[n] for n in names]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m phishstress.stress")
    ap.add_argument("--dataset", default="korccvi_v2")
    ap.add_argument("--detector", action="append", default=[])
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--axis", action="append", default=[], choices=["A", "C", "D"])
    ap.add_argument("--seed", type=int, default=SplitConfig().seed)
    ap.add_argument("--metric", default="recall_at_fpr01")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--ci", action="store_true", help="부트스트랩 신뢰구간을 함께 낸다(느림)")
    args = ap.parse_args(argv)

    axes = tuple(args.axis) if args.axis else ("A", "C", "D")

    print(f"데이터셋 적재: {args.dataset}", file=sys.stderr)
    examples = load(args.dataset)
    splits = make_splits(examples, SplitConfig(seed=args.seed))
    base = splits.get(args.split)

    print(f"스트레스 스위트 생성 (축 {'/'.join(axes)}) …", file=sys.stderr)
    suite = build_suite(base, args.split, seed=args.seed, axes=axes)
    print(f"  기준 {len(base)}건 → 슬라이스 {len(suite.slices)}개\n", file=sys.stderr)

    detectors = build_detectors(
        args.detector, [e.text for e in splits.train], [e.label for e in splits.train]
    )

    payload = {
        "dataset": args.dataset,
        "split": args.split,
        "metric": args.metric,
        "suite": suite.to_dict(),
        "reports": [],
    }

    for det in detectors:
        report = evaluate(det, suite.as_eval_slices())
        baseline = report.by_slice(args.split)
        assert baseline is not None
        base_value = getattr(baseline.metrics, args.metric)

        print(f"■ {det.name}")
        print(f"  기준 슬라이스 {args.split}: {args.metric} = {base_value:.4f}")
        print()
        header = (
            f"  {'키':4s} {'축':3s} {'이름':22s} {'n':>5s} "
            f"{args.metric:>10s} {'Gap':>9s} {'변경률':>7s}"
        )
        print(header)
        print("  " + "-" * 66)

        rows = []
        for s in suite.slices:
            r = report.by_slice(f"{args.split}/{s.key}")
            if r is None:
                continue
            value = getattr(r.metrics, args.metric)
            gap = base_value - value if value == value else float("nan")
            rows.append({**s.to_dict(), args.metric: value, "gap": gap})
            flag = "  ⚠" if gap == gap and gap > 0.15 else ""
            print(
                f"  {s.key:4s} {s.axis:3s} {s.name:22s} {r.metrics.n:5d} "
                f"{value:10.4f} {gap:+9.4f} {s.change_rate:7.3f}{flag}"
            )

        # 축별 요약
        print()
        for axis in axes:
            in_axis = [r for r in rows if r["axis"] == axis and r["gap"] == r["gap"]]
            if not in_axis:
                continue
            worst = max(in_axis, key=lambda r: r["gap"])
            mean = sum(r["gap"] for r in in_axis) / len(in_axis)
            print(
                f"  축 {axis}: 평균 Gap {mean:+.4f} | "
                f"최대 Gap {worst['gap']:+.4f} ({worst['key']} {worst['name']})"
            )
        print()

        entry = {"detector": det.name, "base": base_value, "slices": rows}
        if args.ci:
            entry["base_ci"] = list(
                bootstrap_ci(
                    [e.label for e in base],
                    [det.predict(e.text).score for e in base],
                    lambda a, b: recall_at_fpr(a, b, 0.01),
                    n_resamples=500,
                )
            )
        payload["reports"].append(entry)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"JSON 저장: {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
