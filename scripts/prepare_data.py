#!/usr/bin/env python
"""Stage A data preparation (Sec. III-A-5 and IV-A of the paper).

Sub-commands, in order:
  scan     index every clip of the enabled sources: label, split-grouping key, md5, video metadata
  track    front-end pass (resize, YOLOv8 + DeepSORT on every frame, motion energy at 10 FPS);
           one compressed cache per clip; resumable and shardable over GPUs
  windows  replay every cache through the window engine: one row per window + per-clip statistics
  split    screening, 1:1 balancing, grouped stratified 80/20 split, Stage B 80/20,
           D_SVM clip subset (quotas, continuous tracks) and its 400/100 grouped split

Example (two GPUs):
  python scripts/prepare_data.py scan
  python scripts/prepare_data.py track --gpu 0 --shard 0 --num-shards 2 &
  python scripts/prepare_data.py track --gpu 1 --shard 1 --num-shards 2
  python scripts/prepare_data.py windows
  python scripts/prepare_data.py split
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from stage_a.config import load_config, resolve_path  # noqa: E402
from stage_a.paths import cache_path, manifest_path, windows_path  # noqa: E402

log = logging.getLogger("prepare_data")


# ---------------------------------------------------------------------------- scan
def _probe(path: str, md5: bool = True) -> dict:
    import cv2
    from stage_a.datasets import file_md5
    from stage_a.video import fourcc_of

    out = {"md5": file_md5(path)} if md5 else {}
    out["opened"] = False
    cap = cv2.VideoCapture(str(resolve_path(path)))
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        ok, _ = cap.read()
        out.update(opened=bool(ok), fourcc=fourcc_of(cap), fps=round(fps, 3), frames_meta=int(n),
                   width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                   duration_meta=round(n / fps, 3) if fps > 0 else np.nan)
    cap.release()
    return out


def transcode_unreliable(cfg, df: pd.DataFrame, workers: int) -> pd.DataFrame:
    """Re-encode clips whose codec OpenCV decodes unreliably (DV) to H.264; the index then points
    to the copy and keeps the original path. md5 / dedupe stay on the original files."""
    from stage_a.datasets import _rel
    from stage_a.transcode import find_ffmpeg, transcode_h264
    from stage_a.video import UNRELIABLE_FOURCC

    df["original_path"] = df.path
    todo = df[df.fourcc.isin(UNRELIABLE_FOURCC)]
    if todo.empty:
        return df
    ffmpeg = find_ffmpeg(cfg.tools.get("ffmpeg"))
    if ffmpeg is None:
        raise SystemExit(f"{len(todo)} clips use a codec OpenCV decodes unreliably ({sorted(set(todo.fourcc))}); "
                         "install ffmpeg (or `pip install imageio-ffmpeg`) or set tools.ffmpeg in the config")
    out_root = resolve_path(cfg.paths.transcode_dir)
    dsts = {cid: out_root / f"{cid}.mp4" for cid in todo.clip_id}

    def run(row):
        dst = dsts[row.clip_id]
        if not dst.exists():
            transcode_h264(resolve_path(row.path), dst, ffmpeg)
        return row.clip_id, _probe(_rel(dst), md5=False)

    with ThreadPoolExecutor(min(workers, 6)) as ex:
        done = dict(tqdm(ex.map(run, todo.itertuples()), total=len(todo), desc=f"transcode ({ffmpeg})"))
    for cid, probe in done.items():
        i = df.index[df.clip_id == cid][0]
        if not probe["opened"] or abs(probe["frames_meta"] - df.at[i, "frames_meta"]) > 1:
            raise SystemExit(f"transcoded {cid} does not match its source: {probe}")
        df.at[i, "path"] = _rel(dsts[cid])
        df.at[i, "fourcc"] = probe["fourcc"]
    log.info("transcoded %d clips (%s) to H.264 under %s", len(todo), ", ".join(sorted(set(todo.fourcc))), out_root)
    return df


def cmd_scan(cfg, args):
    from stage_a.datasets import scan_sources

    found = scan_sources(cfg)
    records = [r.to_dict() for recs in found.values() for r in recs]
    for name, recs in found.items():
        log.info("%-8s %5d clips (%d violent / %d non-violent)", name, len(recs),
                 sum(r.label for r in recs), sum(1 - r.label for r in recs))
    missing = [n for n, r in found.items() if not r]
    if missing:
        log.warning("no clips found for: %s (check sources.<name>.root)", ", ".join(missing))
    df = pd.DataFrame(records)
    if df.empty:
        raise SystemExit("nothing to index")
    if df.clip_id.duplicated().any():
        raise SystemExit(f"clip_id collision: {df.clip_id[df.clip_id.duplicated()].tolist()[:5]}")
    with ThreadPoolExecutor(args.workers) as ex:
        probes = list(tqdm(ex.map(_probe, df.path), total=len(df), desc="probe+md5"))
    df = pd.concat([df, pd.DataFrame(probes)], axis=1)
    df = transcode_unreliable(cfg, df, args.workers)
    first = df.groupby("md5").clip_id.transform("first")
    df["duplicate_of"] = np.where(first != df.clip_id, first, "")
    out = manifest_path(cfg, "index.csv")
    df.to_csv(out, index=False)
    dup = df[df.duplicate_of != ""]
    log.info("indexed %d clips -> %s ; %d byte-identical duplicates%s", len(df), out, len(dup),
             "" if dup.empty else ": " + ", ".join(f"{a}={b}" for a, b in zip(dup.clip_id, dup.duplicate_of)))
    log.info("unreadable: %d", int((~df.opened.astype(bool)).sum()))


# ---------------------------------------------------------------------------- track
def cmd_track(cfg, args):
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    from stage_a.frontend import FrontEnd, limit_cpu_threads
    from stage_a.offline import save_cache, track_clip

    limit_cpu_threads(args.threads)

    df = pd.read_csv(manifest_path(cfg, "index.csv"), keep_default_na=False)
    if cfg.benchmark.dedupe:
        df = df[df.duplicate_of == ""]
    if args.sources:
        df = df[df.source.isin(args.sources)]
    df = df.sort_values("clip_id").iloc[args.shard::args.num_shards]
    if args.limit:
        df = df.head(args.limit)
    todo = [r for r in df.itertuples() if args.overwrite or not cache_path(cfg, r.clip_id).exists()]
    log.info("shard %d/%d: %d clips, %d to process", args.shard, args.num_shards, len(df), len(todo))
    if not todo:
        return
    device = "cuda:0" if torch.cuda.is_available() and args.gpu != -1 else "cpu"
    frontend = FrontEnd(cfg.frontend, device)
    err_file = manifest_path(cfg, f"track_errors_shard{args.shard}.csv")
    n_frames, t0 = 0, time.time()
    bar = tqdm(todo, desc=f"track[{args.shard}]")
    for r in bar:
        try:
            data = track_clip(resolve_path(r.path), frontend, cfg)
            save_cache(cache_path(cfg, r.clip_id), data)
            n_frames += data["meta"]["decoded_frames"]
            bar.set_postfix(fps=f"{n_frames / (time.time() - t0):.1f}")
        except Exception as exc:  # keep going; the clip is reported as decode-failed later
            log.error("%s: %s", r.clip_id, exc)
            with open(err_file, "a") as fh:
                fh.write(f"{r.clip_id},{type(exc).__name__}: {str(exc).replace(',', ';')}\n")
    log.info("done: %d frames in %.1f min (%.1f frames/s)", n_frames, (time.time() - t0) / 60,
             n_frames / max(time.time() - t0, 1e-9))


# ---------------------------------------------------------------------------- windows
_WCFG = None


def _init_worker(config_path, overrides):
    global _WCFG
    _WCFG = load_config(config_path, overrides)


def _replay_one(item):
    clip_id, path = item
    from stage_a.offline import TrackCache, replay_windows

    try:
        cache = TrackCache.load(path)
    except Exception as exc:
        return clip_id, [], {"cache_error": f"{type(exc).__name__}: {exc}"}
    windows, stats = replay_windows(cache, _WCFG)
    m = cache.meta
    stats.update(decoded_frames=m["decoded_frames"], decoded_seconds=round(m["decoded_seconds"], 3),
                 truncated=m["truncated"], src_fps=m["fps"], src_width=m["width"], src_height=m["height"])
    rows = [dict(clip_id=clip_id, **w.as_row()) for w in windows]
    return clip_id, rows, stats


def cmd_windows(cfg, args):
    df = pd.read_csv(manifest_path(cfg, "index.csv"), keep_default_na=False)
    items = [(c, str(cache_path(cfg, c))) for c in df.clip_id if cache_path(cfg, c).exists()]
    log.info("replaying %d cached clips (of %d indexed)", len(items), len(df))
    all_rows, stats = [], []
    with ProcessPoolExecutor(args.workers, initializer=_init_worker,
                             initargs=(args.config, args.set)) as ex:
        for clip_id, rows, st in tqdm(ex.map(_replay_one, items, chunksize=8), total=len(items), desc="windows"):
            all_rows.extend(rows)
            stats.append(dict(clip_id=clip_id, **st))
    wdf = pd.DataFrame(all_rows)
    wdf.to_csv(windows_path(cfg), index=False, float_format="%.6g")
    sdf = pd.DataFrame(stats)
    sdf.to_csv(manifest_path(cfg, "clip_stats.csv"), index=False)
    log.info("%d windows from %d clips -> %s", len(wdf), len(sdf), windows_path(cfg))
    log.info("clips without any window: %d", int((sdf.get("n_windows", pd.Series(dtype=int)) == 0).sum()))


# ---------------------------------------------------------------------------- split
def grouped_holdout(df: pd.DataFrame, frac: float, seed: int, strata=("source", "label"),
                    group_col: str = "group") -> pd.Series:
    """Stratified hold-out of ~frac of each stratum, drawn over whole groups. A group can span
    strata (an RWF-2000 video cut into Fight and NonFight clips); its clips already held out
    through another stratum count toward this stratum's target."""
    rng = np.random.RandomState(seed)
    hold_groups = set()
    for _, sub in df.groupby(list(strata), sort=True):
        sizes = sub.groupby(group_col).size()
        target = int(round(frac * len(sub)))
        n = int(sizes[sizes.index.isin(hold_groups)].sum())
        for g in rng.permutation(sizes.index.difference(list(hold_groups)).to_numpy()):
            if n >= target:
                break
            if n > 0 and n + sizes[g] - target > sizes[g] / 2:
                continue  # taking this group would overshoot by more than it helps
            hold_groups.add(g)
            n += sizes[g]
    return df[group_col].isin(hold_groups)


def subsample_groups(sub: pd.DataFrame, target: int, rng) -> pd.Index:
    """Pick whole groups at random until ``target`` clips are reached; returns the kept index."""
    sizes = sub.groupby("group").size()
    keep, n = [], 0
    for g in rng.permutation(sizes.index.to_numpy()):
        if n + sizes[g] > target:
            continue
        keep.append(g)
        n += sizes[g]
        if n == target:
            break
    return sub.index[sub.group.isin(keep)]


def cmd_split(cfg, args):
    b = cfg.benchmark
    rng = np.random.RandomState(b.seed)
    idx = pd.read_csv(manifest_path(cfg, "index.csv"), keep_default_na=False)
    stats = pd.read_csv(manifest_path(cfg, "clip_stats.csv"))
    df = idx.merge(stats, on="clip_id", how="left")
    df["excluded"] = ""

    def exclude(mask, reason):
        mask = mask & (df.excluded == "")
        df.loc[mask, "excluded"] = reason
        return int(mask.sum())

    screened = df.source.isin(b.screened_sources)
    duration = pd.to_numeric(df.duration_meta, errors="coerce").fillna(df.decoded_seconds)
    report = {"screening": {}}
    if b.dedupe:
        # a file shipped under both labels cannot be trusted either way: drop every copy
        conflict = df.groupby("md5").label.transform("nunique") > 1
        report["screening"]["label_conflict"] = exclude(conflict, "label_conflict")
        report["screening"]["duplicate"] = exclude(df.duplicate_of != "", "duplicate")
    report["screening"]["decode_failed"] = exclude(df.decoded_frames.fillna(0) <= 0, "decode_failed")
    report["screening"]["too_short"] = exclude(screened & (duration < b.min_duration_s), "too_short")
    long_mask = screened & (duration > b.max_duration_s)
    if b.long_clip_policy == "drop":
        report["screening"]["too_long"] = exclude(long_mask, "too_long")
    df["trimmed"] = long_mask & (df.excluded == "")
    report["screening"]["trimmed_to_max_duration"] = int(df.trimmed.sum())
    report["screening"]["no_usable_track"] = exclude(
        screened & (df.max_track_run.fillna(0) < cfg.descriptor.window), "no_usable_track")

    # 1:1 balance of the screened sources, by subsampling the larger class (whole groups)
    for src in b.balance_sources:
        kept = df[(df.source == src) & (df.excluded == "")]
        n_v, n_nv = int((kept.label == 1).sum()), int((kept.label == 0).sum())
        if n_v == n_nv or min(n_v, n_nv) == 0:
            continue
        big = 1 if n_v > n_nv else 0
        sub = kept[kept.label == big]
        keep = subsample_groups(sub, min(n_v, n_nv), rng)
        report["screening"][f"balance_{src}"] = exclude(df.index.isin(sub.index.difference(keep)),
                                                        "balance_subsampled")

    kept = df[df.excluded == ""].copy()
    spans = kept.groupby("group")[["source", "label"]].nunique().max(axis=1)
    if (spans > 1).any():
        log.warning("%d split groups span several (source, label) strata", int((spans > 1).sum()))
    test = grouped_holdout(kept, b.test_fraction, b.seed)
    kept["split"] = np.where(test, "test", "train")
    train = kept[kept.split == "train"]
    val = grouped_holdout(train, b.stageb_val_fraction, b.seed + 1)
    kept["stageb_split"] = ""
    kept.loc[train.index, "stageb_split"] = np.where(val, "val", "train")

    # D_SVM clip subset: quotas per source and label, clips with continuous tracks
    s = cfg.svm
    kept["svm_selected"] = False
    shortfalls = {}
    for src, q in s.quotas.items():
        for label, key in ((1, "violent"), (0, "non_violent")):
            want = int(q[key])
            pool = kept[(kept.split == "train") & (kept.source == src) & (kept.label == label)]
            if want == 0:
                continue
            if pool.empty:
                shortfalls[f"{src}/{key}"] = f"0 of {want} (source unavailable)"
                continue
            good = pool[(pool.window_coverage >= s.min_window_coverage) & (pool.n_windows > 0)]
            pick = list(good.index[rng.permutation(len(good))][:want])
            if len(pick) < want:  # fill with the best remaining clips that still yield windows
                rest = pool.drop(index=pick)
                rest = rest[rest.n_windows > 0].sort_values("window_coverage", ascending=False)
                pick += list(rest.index[: want - len(pick)])
                shortfalls[f"{src}/{key}"] = (f"{len(good)} clips meet coverage >= {s.min_window_coverage}; "
                                              f"selected {len(pick)} of {want}")
            kept.loc[pick, "svm_selected"] = True
    svm = kept[kept.svm_selected]
    svm_val = grouped_holdout(svm, s.val_fraction, b.seed + 2)
    kept["svm_split"] = ""
    kept.loc[svm.index, "svm_split"] = np.where(svm_val, "val", "train")

    out = df.merge(kept[["clip_id", "split", "stageb_split", "svm_selected", "svm_split"]], on="clip_id", how="left")
    out[["split", "stageb_split", "svm_split"]] = out[["split", "stageb_split", "svm_split"]].fillna("")
    out["svm_selected"] = out.svm_selected.astype("boolean").fillna(False).astype(bool)
    out.to_csv(manifest_path(cfg, "benchmark.csv"), index=False)

    def table(frame, col):
        t = frame.groupby(["source", col, "label"]).size().unstack("label", fill_value=0)
        return {f"{s_}/{c_}": {"violent": int(r.get(1, 0)), "non_violent": int(r.get(0, 0))}
                for (s_, c_), r in t.iterrows()}

    kept_out = out[out.excluded == ""]
    report.update({
        "benchmark_total": int(len(kept_out)),
        "per_source": {src: {"violent": int((g.label == 1).sum()), "non_violent": int((g.label == 0).sum())}
                       for src, g in kept_out.groupby("source")},
        "split": table(kept_out, "split"),
        "stageb_split": table(kept_out[kept_out.stageb_split != ""], "stageb_split"),
        "svm_split": table(kept_out[kept_out.svm_split != ""], "svm_split"),
        "svm_shortfalls": shortfalls,
    })
    with open(manifest_path(cfg, "split_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["scan", "track", "windows", "split"])
    ap.add_argument("--config", default=None, help="YAML config (default configs/stage_a.yaml)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides, e.g. descriptor.grouping_k=2.0")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--gpu", type=int, default=None, help="track: GPU index (-1 = CPU)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--sources", nargs="*", default=None, help="track: restrict to these sources")
    ap.add_argument("--limit", type=int, default=0, help="track: at most this many clips, for a quick check")
    ap.add_argument("--overwrite", action="store_true", help="track: recompute existing caches")
    ap.add_argument("--threads", type=int, default=2, help="track: CPU threads per process")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config, args.set)
    {"scan": cmd_scan, "track": cmd_track, "windows": cmd_windows, "split": cmd_split}[args.command](cfg, args)


if __name__ == "__main__":
    main()
