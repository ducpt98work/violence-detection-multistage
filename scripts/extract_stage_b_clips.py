#!/usr/bin/env python
"""Cut the Stage B training clips from the Stage A artefacts (Sec. III-B-3, offline protocol).

Each benchmark clip yields one 16-frame sub-sequence centred on the window Stage A flags most
strongly: the forwarded window with the largest SVM margin, or, if no window of the clip is
forwarded, the window with the largest margin. Frames t-8 .. t+7 are cut with the runtime's own ROI
rule (union of the group's boxes over the clip, padded 15 %, squared, cropped from the source frame,
resized to 224), so training and deployment see the same crops. One extra frame on each side
(t-9, t+8) is stored for temporal jitter. A clip in which no group is ever admitted falls back to
the union of all tracked boxes around its middle, and to the full frame if nobody is tracked.

Needs the track caches and benchmark split of prepare_data.py and a trained gate. Writes
  <out-dir>/benchmark.csv        clip_id, source, label, split (train / val / test), trigger
  <out-dir>/clips/<clip_id>.npy  (18, 224, 224, 3) RGB uint8

Example:
  python scripts/extract_stage_b_clips.py --out-dir work/stage_b
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from stage_a.config import config_from_dict, load_config, resolve_path  # noqa: E402
from stage_a.paths import cache_path, load_benchmark, models_dir  # noqa: E402
from stage_b.data import STORED_FRAMES, clip_file  # noqa: E402

log = logging.getLogger("extract_stage_b_clips")

_CFG = _GATE = None


def _init(model, config, overrides):
    global _CFG, _GATE
    from stage_a.gate import StageAGate

    _GATE = StageAGate.load(model)
    _CFG = load_config(config, overrides) if config else config_from_dict(_GATE.config, overrides)


def flagged_trigger(windows, gate):
    """(window, margin, how) for the window Stage A flags most strongly."""
    X = np.stack([w.features for w in windows])
    margin = gate.score(X)
    forward = gate.decide(X).forward
    if forward.any():
        i = int(np.flatnonzero(forward)[np.argmax(margin[forward])])
        return windows[i], float(margin[i]), "forwarded"
    i = int(np.argmax(margin))
    return windows[i], float(margin[i]), "scored"


def extract(clip_id: str, path: str, out_dir: Path, overwrite: bool):
    from stage_a.descriptor import WindowEngine
    from stage_a.offline import TrackCache
    from stage_a.runtime import crop_roi, roi_boxes
    from stage_a.video import read_frames

    cfg, rc = _CFG, _CFG.roi_clip
    cache = TrackCache.load(cache_path(cfg, clip_id))
    engine = WindowEngine.from_cfg(cfg, keep=100_000)  # keep every record: the clip ROI spans t-8 .. t+7
    unit, hw = cache.meta["energy_unit"], cache.analysis_hw
    windows = []
    for aidx, obs, units in cache.iter_frames():
        engine.observe(aidx, obs)
        windows.extend(engine.windows(aidx, units, unit, hw))

    n_analysis = len(cache.a_idx)
    if windows:
        w, margin, how = flagged_trigger(windows, _GATE)
        t, members = w.aidx, w.members
    else:
        t, margin, how = n_analysis // 2, float("nan"), "no_group"
        members = list(engine.hist)
    lo, hi = t - rc.frames_before, t + rc.frames_after
    union = engine.member_box_union(members, lo, hi) if members else None
    if union is None and windows:
        union = np.array(w.roi, np.float32)
    if union is None:
        how = "no_track"
        union = np.array([0, 0, hw[1], hw[0]], np.float32)

    src_hw = (int(cache.meta["height"]), int(cache.meta["width"]))
    if rc.crop_from != "original":
        src_hw = hw
    roi_a, roi_s = roi_boxes(union, hw, src_hw, cfg)

    dest = clip_file(out_dir / "clips", clip_id)
    if overwrite or not dest.exists():
        # one stored frame on each side of the 16 for temporal jitter; outside the clip, the nearest frame
        ks = np.clip(np.arange(lo - 1, hi + 2), 0, n_analysis - 1)
        frames = read_frames(resolve_path(path), cache.a_frame[ks].tolist(), rc.crop_from,
                             cfg.frontend.resize_width)
        clip = np.stack([crop_roi(f, roi_s, cfg)[..., ::-1] for f in frames])
        assert len(clip) == STORED_FRAMES
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.stem + ".tmp.npy")
        np.save(tmp, clip)
        os.replace(tmp, dest)
    return {"clip_id": clip_id, "trigger_aidx": int(t), "trigger": how, "margin": margin,
            "n_windows": len(windows), "roi_x1": roi_s[0], "roi_y1": roi_s[1], "roi_x2": roi_s[2],
            "roi_y2": roi_s[3]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="gate bundle (default models/stage_a_gate.joblib)")
    ap.add_argument("--config", default=None, help="Stage A YAML instead of the config stored in the bundle")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--out-dir", default="work/stage_b")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--limit", type=int, default=0, help="at most this many clips per (source, label, split), for a quick check")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    base = load_config()
    model = Path(args.model) if args.model else models_dir(base) / "stage_a_gate.joblib"
    bench = load_benchmark(base)
    bench["split"] = np.where(bench.split == "test", "test", bench.stageb_split)
    if args.limit:
        bench = bench.groupby(["source", "label", "split"], group_keys=False).head(args.limit)
    out_dir = resolve_path(args.out_dir)

    rows = []
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(model, args.config, args.set)) as ex:
        jobs = [ex.submit(extract, cid, path, out_dir, args.overwrite)
                for cid, path in zip(bench.clip_id, bench.path)]
        for job in tqdm(jobs, desc="stage B clips"):
            rows.append(job.result())

    out = bench[["clip_id", "source", "label", "split"]].merge(pd.DataFrame(rows), on="clip_id")
    out.to_csv(out_dir / "benchmark.csv", index=False)
    log.info("%d clips -> %s", len(out), out_dir)
    log.info("trigger: %s", out.trigger.value_counts().to_dict())
    log.info("split: %s", out.groupby(["split", "label"]).size().to_dict())


if __name__ == "__main__":
    main()
