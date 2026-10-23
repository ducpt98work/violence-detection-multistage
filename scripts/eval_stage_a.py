#!/usr/bin/env python
"""Evaluate the Stage A gate on a benchmark partition (Sec. IV-C, Table "Stage A decomposed by source").

Window level, per source and pooled:
  Rej. phys / Rej. SVM / Fwd.   share of candidate windows removed by each gate / forwarded
  Rec. / Prec.                  violent window recall and precision of the gate as a whole
Clip level:
  Cov.                          violent clips with at least one forwarded window
  Clips/vid.                    Stage B invocations per video after merging (Algorithm 1)
  no group / gated out          violent clips that never form a group / form one but never pass
  Stage A only                  Stage A used as a standalone classifier (clip violent if any window
                                passes), at the SVM's default threshold 0 and at the operating one
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from stage_a.clips import count_invocations  # noqa: E402
from stage_a.config import display_path, load_config  # noqa: E402
from stage_a.descriptor import FEATURE_NAMES  # noqa: E402
from stage_a.gate import StageAGate  # noqa: E402
from stage_a.paths import load_benchmark, load_windows, models_dir  # noqa: E402

log = logging.getLogger("eval_stage_a")


def clip_metrics(y, pred) -> dict:
    y, pred = np.asarray(y, bool), np.asarray(pred, bool)
    tp, fp = int((y & pred).sum()), int((~y & pred).sum())
    fn, tn = int((y & ~pred).sum()), int((~y & ~pred).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"accuracy": (tp + tn) / max(len(y), 1), "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "fp_rate": fp / (fp + tn) if fp + tn else 0.0, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--model", default=None, help="gate bundle (default <models_dir>/stage_a_gate.joblib)")
    ap.add_argument("--split", default="test", help="split column value: test | train")
    ap.add_argument("--column", default="split", help="benchmark column to filter on (split, stageb_split, svm_split)")
    ap.add_argument("--threshold", type=float, default=None, help="override the operating threshold")
    ap.add_argument("--out-dir", default=None, help="default <models_dir>/eval")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config, args.set)
    model = Path(args.model) if args.model else models_dir(cfg) / "stage_a_gate.joblib"
    gate = StageAGate.load(model)
    thr = gate.threshold if args.threshold is None else args.threshold
    out_dir = Path(args.out_dir) if args.out_dir else models_dir(cfg) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    bench = load_benchmark(cfg)
    clips = bench[bench[args.column] == args.split][["clip_id", "source", "label"]].reset_index(drop=True)
    if clips.empty:
        raise SystemExit(f"no clips with {args.column} == {args.split!r}")
    win = load_windows(cfg, clips.clip_id).merge(clips[["clip_id", "label", "source"]], on="clip_id")
    X = win[list(FEATURE_NAMES)].to_numpy(np.float64)
    dec = gate.decide(X, thr)
    win["physics"] = dec.physics
    win["score"] = dec.score
    win["forward"] = dec.forward
    win["pass_default"] = dec.physics & (np.nan_to_num(dec.score, nan=-np.inf) >= 0.0)
    log.info("%d clips, %d windows, threshold %.4f", len(clips), len(win), thr)

    rc = cfg.roi_clip
    per_clip = []
    for clip_id, g in win.groupby("clip_id"):
        f = g[g.forward]
        triggers = [(int(a), [int(m) for m in str(ms).split(";")]) for a, ms in zip(f.aidx, f.members)]
        per_clip.append({"clip_id": clip_id, "n_windows": len(g), "n_physics": int(g.physics.sum()),
                         "n_forwarded": len(f), "stage_b_invocations": count_invocations(triggers, rc.frames_before, rc.frames_after),
                         "any_forward": bool(len(f)), "any_default": bool(g.pass_default.any()),
                         "max_score": float(np.nanmax(g.score)) if g.physics.any() else float("nan")})
    pc = clips.merge(pd.DataFrame(per_clip), on="clip_id", how="left")
    pc[["n_windows", "n_physics", "n_forwarded", "stage_b_invocations"]] = \
        pc[["n_windows", "n_physics", "n_forwarded", "stage_b_invocations"]].fillna(0).astype(int)
    for col in ("any_forward", "any_default"):
        pc[col] = pc[col].astype("boolean").fillna(False).astype(bool)
    pc["status"] = np.select([pc.n_windows == 0, ~pc.any_forward], ["no_group", "gated_out"], "forwarded")

    rows = []
    for src, w, c in [(s, win[win.source == s], pc[pc.source == s]) for s in sorted(clips.source.unique())] + \
                     [("pooled", win, pc)]:
        viol = w.label == 1
        cv = c[c.label == 1]
        rows.append({
            "source": src, "clips": len(c), "windows": len(w),
            "rej_phys": 100 * (~w.physics).mean(), "rej_svm": 100 * (w.physics & ~w.forward).mean(),
            "fwd": 100 * w.forward.mean(),
            "rec": 100 * w.forward[viol].mean() if viol.any() else np.nan,
            "prec": 100 * viol[w.forward].mean() if w.forward.any() else np.nan,
            "cov": 100 * cv.any_forward.mean() if len(cv) else np.nan,
            "clips_per_vid": c.stage_b_invocations.mean(),
            "violent_no_group": int((cv.status == "no_group").sum()),
            "violent_gated_out": int((cv.status == "gated_out").sum()),
            "stage_a_only_default": clip_metrics(c.label, c.any_default),
            "stage_a_only_operating": clip_metrics(c.label, c.any_forward),
        })

    tag = f"{args.column}_{args.split}"
    pc.to_csv(out_dir / f"per_clip_{tag}.csv", index=False)
    with open(out_dir / f"stage_a_eval_{tag}.json", "w") as fh:
        json.dump({"model": display_path(model), "threshold": thr, "rows": rows}, fh, indent=2, default=float)

    md = [f"# Stage A on {args.column} = {args.split} (threshold {thr:.4f})", "",
          "| Source | Clips | Windows | Rej. phys (%) | Rej. SVM (%) | Fwd. (%) | Rec. (%) | Prec. (%) | Cov. (%) | Clips/vid | no group | gated out |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['source']} | {r['clips']} | {r['windows']} | {r['rej_phys']:.1f} | {r['rej_svm']:.1f} | {r['fwd']:.1f} | "
                  f"{r['rec']:.1f} | {r['prec']:.1f} | {r['cov']:.1f} | {r['clips_per_vid']:.2f} | "
                  f"{r['violent_no_group']} | {r['violent_gated_out']} |")
    md += ["", "## Stage A as a standalone clip classifier", "",
           "| Source | Thr. | Acc. (%) | Prec. (%) | Rec. (%) | F1 (%) | FP rate (%) |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        for name, key in (("0 (default)", "stage_a_only_default"), (f"{thr:.3f} (op.)", "stage_a_only_operating")):
            m = r[key]
            md.append(f"| {r['source']} | {name} | {100 * m['accuracy']:.1f} | {100 * m['precision']:.1f} | "
                      f"{100 * m['recall']:.1f} | {100 * m['f1']:.1f} | {100 * m['fp_rate']:.1f} |")
    (out_dir / f"stage_a_eval_{tag}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
