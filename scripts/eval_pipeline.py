#!/usr/bin/env python
"""End-to-end evaluation of the two stages on a benchmark partition (Sec. IV-E, IV-F).

For every video, Stage A is replayed from its track cache (the runtime gives the same windows; see
tests/test_offline_runtime_consistency.py), forwarded windows are merged into clips as in
Algorithm 1, each clip is cut from the source video with the runtime ROI rule and scored by
Stage B, and the video is violent if any of its clips is (no forwarded clip: non-violent).
A missed violent video is attributed to the stage that lost it: no interaction group was ever
formed, every window was gated out, or Stage B rejected every clip.

Writes metrics.json, per_source.md (F1 mean +- s.d. over the checkpoints given, one per seed) and
per_video.csv (video probability per checkpoint) to --out.

Example:
  python scripts/eval_pipeline.py --stage-b "runs/main/seed*/model.pt" --out runs/main/pipeline_test
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from tqdm import tqdm  # noqa: E402

from stage_a.config import config_from_dict, display_path, load_config, resolve_path  # noqa: E402
from stage_a.paths import cache_path, load_benchmark, models_dir  # noqa: E402
from stage_b.metrics import binary_metrics  # noqa: E402

log = logging.getLogger("eval_pipeline")

_CFG = _GATE = None


def _init(model, config, overrides):
    global _CFG, _GATE
    from stage_a.gate import StageAGate

    _GATE = StageAGate.load(model)
    _CFG = load_config(config, overrides) if config else config_from_dict(_GATE.config, overrides)


def stage_a_clips(clip_id: str, path: str):
    """Replay Stage A on one video. Returns (n_windows, n_forwarded, [clip (16, S, S, 3) RGB])."""
    from stage_a.clips import ClipMerger
    from stage_a.descriptor import WindowEngine
    from stage_a.offline import TrackCache
    from stage_a.runtime import crop_roi, roi_boxes
    from stage_a.video import read_frames

    cfg, rc = _CFG, _CFG.roi_clip
    cache = TrackCache.load(cache_path(cfg, clip_id))
    engine = WindowEngine.from_cfg(cfg, keep=100_000)
    merger = ClipMerger(rc.frames_before, rc.frames_after)
    unit, hw = cache.meta["energy_unit"], cache.analysis_hw
    n_windows = n_forwarded = 0
    ready = []
    for aidx, obs, units in cache.iter_frames():
        engine.observe(aidx, obs)
        windows = engine.windows(aidx, units, unit, hw)
        if windows:
            dec = _GATE.decide(np.stack([w.features for w in windows]))
            n_windows += len(windows)
            for i in np.flatnonzero(dec.forward):
                merger.add(aidx, windows[i].members, float(dec.score[i]), payload=windows[i])
                n_forwarded += 1
        ready += merger.pop_ready(aidx)
    ready += merger.pop_ready(None)
    if not ready:
        return n_windows, n_forwarded, []

    src_hw = (int(cache.meta["height"]), int(cache.meta["width"])) if rc.crop_from == "original" else hw
    n_analysis = len(cache.a_idx)
    plans = []
    for clip in ready:
        union = engine.member_box_union(clip.members, clip.lo, clip.hi)
        if union is None:
            union = np.array(clip.payload[0].roi, np.float32)
        _, roi = roi_boxes(union, hw, src_hw, cfg)
        ks = np.clip(np.arange(clip.lo, clip.hi + 1), 0, n_analysis - 1)  # nearest frame outside the video
        plans.append((roi, cache.a_frame[ks].tolist()))
    wanted = sorted({i for _, idx in plans for i in idx})
    frames = dict(zip(wanted, read_frames(resolve_path(path), wanted, rc.crop_from, cfg.frontend.resize_width)))
    clips = [np.stack([crop_roi(frames[i], roi, cfg)[..., ::-1] for i in idx]) for roi, idx in plans]
    return n_windows, n_forwarded, clips


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage-b", nargs="+", required=True, help="Stage B checkpoint(s), one per seed; globs allowed")
    ap.add_argument("--model", default=None, help="gate bundle (default models/stage_a_gate.joblib)")
    ap.add_argument("--config", default=None, help="Stage A YAML instead of the config stored in the bundle")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--split", default="test", help="benchmark split: test | train")
    ap.add_argument("--source", default=None, help="evaluate one source only (e.g. a leave-one-dataset-out fold)")
    ap.add_argument("--threshold", type=float, default=0.5, help="Stage B clip threshold")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import torch
    from stage_b.inference import Recognizer

    ckpts = sorted(p for pattern in args.stage_b for p in glob.glob(pattern))
    if not ckpts:
        sys.exit(f"no checkpoint matches {args.stage_b}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    recognizers = [Recognizer(c, device) for c in ckpts]

    base = load_config()
    model = Path(args.model) if args.model else models_dir(base) / "stage_a_gate.joblib"
    bench = load_benchmark(base)
    bench = bench[bench.split == args.split]
    if args.source:
        bench = bench[bench.source == args.source]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(model, args.config, args.set)) as ex:
        results = ex.map(stage_a_clips, bench.clip_id, bench.path, chunksize=4)
        for r, (n_win, n_fwd, clips) in tqdm(zip(bench.itertuples(), results), total=len(bench), desc="videos"):
            row = {"clip_id": r.clip_id, "source": r.source, "label": int(r.label),
                   "n_windows": n_win, "n_forwarded": n_fwd, "n_clips": len(clips)}
            for k, rec in enumerate(recognizers):
                # Eq. (roi_aggregation): the video is as violent as its most violent clip
                row[f"prob_{k}"] = max((rec(c) for c in clips), default=0.0)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_video.csv", index=False)

    report = {"checkpoints": [display_path(c) for c in ckpts], "gate": display_path(model), "split": args.split, "threshold": args.threshold,
              "clips_per_video": float(df.n_clips.mean()), "runs": []}
    for k in range(len(ckpts)):
        prob = df[f"prob_{k}"].to_numpy()
        run = {}
        for name, part in [("pooled", df)] + list(df.groupby("source")):
            m = binary_metrics(part.label, part[f"prob_{k}"], args.threshold)
            if part.label.nunique() == 2:
                m["auc"] = float(roc_auc_score(part.label, part[f"prob_{k}"]))
            run[name] = m
        missed = df[(df.label == 1) & (prob <= args.threshold)]
        run["misses"] = {"total": int(len(missed)),
                         "no_group": int((missed.n_windows == 0).sum()),
                         "gated_out": int(((missed.n_windows > 0) & (missed.n_forwarded == 0)).sum()),
                         "stage_b_rejects": int((missed.n_forwarded > 0).sum())}
        report["runs"].append(run)
    with open(out / "metrics.json", "w") as fh:
        json.dump(report, fh, indent=2)

    first = report["runs"][0]
    keys = ["pooled"] + sorted(k for k in first if k not in ("pooled", "misses"))
    lines = [f"Full pipeline on split = {args.split}; {len(ckpts)} Stage B checkpoint(s); "
             f"Acc/Prec/Recall/FP rate/AUC from {display_path(ckpts[0])}; {report['clips_per_video']:.2f} Stage B clips per video.", "",
             "| Source | Videos | Acc | Prec | Recall | F1 (mean ± s.d.) | FP rate | AUC |", "|---|---|---|---|---|---|---|---|"]
    for key in keys:
        f1 = np.array([r[key]["f1"] for r in report["runs"]]) * 100
        sd = f"{f1.std(ddof=1):.1f}" if len(f1) > 1 else "-"
        m = first[key]
        lines.append(f"| {key} | {m['n']} | {100 * m['acc']:.1f} | {100 * m['precision']:.1f} | {100 * m['recall']:.1f} | "
                     f"{f1.mean():.1f} ± {sd} | {100 * m['fpr']:.1f} | {100 * m.get('auc', float('nan')):.1f} |")
    mis, pooled = first["misses"], first["pooled"]
    lines += ["", f"Missed violent videos: {mis['total']} (no group {mis['no_group']}, gated out {mis['gated_out']}, "
                  f"Stage B rejects {mis['stage_b_rejects']}). MCC {pooled['mcc']:.3f}, kappa {pooled['kappa']:.3f}, "
                  f"Brier {pooled['brier']:.3f}."]
    (out / "per_source.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
