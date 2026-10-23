"""Evaluate trained Stage B checkpoints on a benchmark partition.

Writes per-seed metrics, the per-source table and mean +- s.d. of F1 over the
checkpoints given (one per seed).

    python scripts/eval_stage_b.py --ckpt runs/main/seed4*/model.pt --out runs/main/test
    python scripts/eval_stage_b.py --ckpt runs/lodo/hockey/seed*/model.pt --holdout hockey --out runs/lodo/hockey/test
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage_a.config import display_path  # noqa: E402
from stage_b.data import read_benchmark, select  # noqa: E402
from stage_b.engine import build_model, make_loader, predict  # noqa: E402
from stage_b.metrics import binary_metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--holdout")
    ap.add_argument("--benchmark", help="override the benchmark CSV stored in the checkpoint")
    ap.add_argument("--clips", help="override the clip directory stored in the checkpoint")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths = sorted(p for pattern in args.ckpt for p in glob.glob(pattern))
    if not paths:
        sys.exit(f"no checkpoint matches {args.ckpt}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    runs = {}
    for path in paths:
        ckpt = torch.load(path, map_location="cpu")
        cfg = ckpt["config"]
        if args.benchmark:
            cfg["data"]["benchmark"] = args.benchmark
        if args.clips:
            cfg["data"]["clips"] = args.clips
        rows = select(read_benchmark(cfg["data"]["benchmark"]), args.split, holdout=args.holdout)
        model = build_model(cfg, pretrained=False)
        model.load_state_dict(ckpt["model"])
        model.freeze_backbone()
        model.to(device)
        probs, labels = predict(model, make_loader(cfg, rows, False, cfg["train"]["batch_size"]), device)
        thr = cfg["train"]["threshold"]

        result = {"pooled": binary_metrics(labels, probs, thr)}
        sources = np.array([r["source"] for r in rows])
        for s in sorted(set(sources)):
            result[s] = binary_metrics(labels[sources == s], probs[sources == s], thr)
        runs[display_path(path)] = result
        np.savetxt(Path(path).with_name(f"{args.split}_probs.csv"),
                   np.c_[labels, probs], fmt="%.6f", delimiter=",", header="label,prob")
        print(f"{path}: F1 {result['pooled']['f1']:.4f}  acc {result['pooled']['acc']:.4f}")

    keys = ["pooled"] + sorted(set.intersection(*(set(r) for r in runs.values())) - {"pooled"})
    first = runs[display_path(paths[0])]
    lines = ["| Source | Clips | Acc | Prec | Recall | F1 (mean ± s.d.) | FP rate |",
             "|---|---|---|---|---|---|---|"]
    for k in keys:
        f1 = np.array([r[k]["f1"] for r in runs.values()]) * 100
        m = first[k]
        sd = f"{f1.std(ddof=1):.1f}" if len(f1) > 1 else "–"
        lines.append(f"| {k} | {m['n']} | {m['acc'] * 100:.1f} | {m['precision'] * 100:.1f} | "
                     f"{m['recall'] * 100:.1f} | {f1.mean():.1f} ± {sd} | {m['fpr'] * 100:.1f} |")
    table = "\n".join(lines)
    print(table)
    header = f"{len(paths)} checkpoint(s); Acc/Prec/Recall/FP rate from {display_path(paths[0])}.\n\n"
    (out / "per_source.md").write_text(header + table + "\n", encoding="utf-8")
    with open(out / "metrics.json", "w") as f:
        json.dump(runs, f, indent=2)


if __name__ == "__main__":
    main()
