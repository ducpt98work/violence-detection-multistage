#!/usr/bin/env python
"""Stage A signals on labelled time segments of one video (the diagnostic of Sec. IV-D / Fig. 5).

The video is tracked once (cached in --out-dir), then replayed under one or more descriptor
variants. For each variant and segment it reports, over all windows of all groups: the share of
windows clearing each physics threshold and their disjunction, the median of each descriptor
component, and the single-component AUC between the first two segments (Table "feature_importance").

Example (the paper's diagnostic evaluates every analysis frame, stride 1):
  python scripts/diagnose_video.py --source video.mp4 --segments walking:20-50 fight:110-140 \
      --stride 1 --variants kalman:descriptor.box_source=kalman detection:descriptor.box_source=detection
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from stage_a.config import load_config  # noqa: E402
from stage_a.descriptor import FEATURE_NAMES  # noqa: E402
from stage_a.gate import physics_pass  # noqa: E402
from stage_a.offline import TrackCache, replay_windows, save_cache  # noqa: E402

log = logging.getLogger("diagnose_video")


def parse_segments(items):
    segs = []
    for it in items:
        name, span = it.split(":")
        a, b = span.split("-")
        segs.append((name, float(a), float(b)))
    return segs


def summarize(df: pd.DataFrame, segs, cfg) -> dict:
    from sklearn.metrics import roc_auc_score

    pa, ph = cfg.gate.physics.a_max, cfg.gate.physics.mu_heat
    X = df[list(FEATURE_NAMES)].to_numpy()
    df = df.assign(clear_a_max=X[:, FEATURE_NAMES.index("A_max")] > pa,
                   clear_mu_heat=X[:, FEATURE_NAMES.index("mu_heat")] > ph,
                   physics=physics_pass(X, pa, ph))
    out = {"segments": {}}
    for name, _, _ in segs:
        g = df[df.segment == name]
        if g.empty:
            continue
        out["segments"][name] = {
            "windows": len(g),
            "clear_mu_heat_pct": 100 * g.clear_mu_heat.mean(), "clear_a_max_pct": 100 * g.clear_a_max.mean(),
            "physics_or_pct": 100 * g.physics.mean(),
            "a_max_subset_of_mu_heat": bool((~g.clear_a_max | g.clear_mu_heat).all()),
            "medians": {f: float(g[f].median()) for f in FEATURE_NAMES},
        }
    names = [n for n, _, _ in segs]
    if len(names) >= 2 and all(n in out["segments"] for n in names[:2]):
        sub = df[df.segment.isin(names[:2])]
        y = (sub.segment == names[1]).to_numpy()
        out["auc_alone"] = {f: float(roc_auc_score(y, sub[f].abs() if f == "mu_a" else sub[f])) for f in FEATURE_NAMES}
    return out


def to_markdown(name, summ) -> list:
    segs = summ["segments"]
    lines = [f"## Variant `{name}`", "",
             "| Segment | Windows | > mu_heat (%) | > A_max (%) | OR (%) | A_max set inside mu_heat set |",
             "|---|---|---|---|---|---|"]
    for n, s in segs.items():
        lines.append(f"| {n} | {s['windows']} | {s['clear_mu_heat_pct']:.1f} | {s['clear_a_max_pct']:.1f} | "
                     f"{s['physics_or_pct']:.1f} | {s['a_max_subset_of_mu_heat']} |")
    has_auc = "auc_alone" in summ
    lines += ["", "| Component | " + " | ".join(f"median {n}" for n in segs) + (" | AUC alone |" if has_auc else " |"),
              "|---" * (len(segs) + 1 + has_auc) + "|"]
    for f in FEATURE_NAMES:
        cells = " | ".join(f"{s['medians'][f]:.2f}" for s in segs.values())
        lines.append(f"| {f} | {cells}" + (f" | {summ['auc_alone'][f]:.2f} |" if has_auc else " |"))
    return lines + [""]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--segments", nargs="+", required=True, metavar="NAME:START-END", help="seconds")
    ap.add_argument("--stride", type=int, default=None, help="evaluate windows every k analysis frames (paper's diagnostic: 1)")
    ap.add_argument("--variants", nargs="*", default=["default:"], metavar="NAME:KEY=VALUE[,KEY=VALUE]",
                    help="descriptor variants replayed on the same tracks")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--retrack", action="store_true", help="ignore an existing track cache")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    segs = parse_segments(args.segments)
    end = max(b for _, _, b in segs)
    out_dir = Path(args.out_dir) if args.out_dir else Path("outputs") / f"diagnose_{Path(args.source).stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    # every analysis frame keeps its energy map so that any stride can be replayed
    base = list(args.set) + [f"frontend.max_seconds={end + 0.05}", "energy.store_every=1"]
    cache_file = out_dir / "track_cache.npz"
    if args.retrack or not cache_file.exists():
        if args.gpu >= 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        import torch

        from stage_a.frontend import FrontEnd, limit_cpu_threads
        from stage_a.offline import track_clip

        limit_cpu_threads(args.threads)
        cfg = load_config(args.config, base)
        fe = FrontEnd(cfg.frontend, "cuda:0" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
        log.info("tracking %s up to %.1f s", args.source, end)
        save_cache(cache_file, track_clip(args.source, fe, cfg))
    cache = TrackCache.load(cache_file)

    report, md = {}, [f"# Stage A signals on {Path(args.source).name}", ""]
    for item in args.variants:
        name, _, ov = item.partition(":")
        overrides = base + ([f"descriptor.stride={args.stride}"] if args.stride else []) + [o for o in ov.split(",") if o]
        cfg = load_config(args.config, overrides)
        windows, _ = replay_windows(cache, cfg, max_time=end)
        rows = []
        for w in windows:
            seg = next((n for n, a, b in segs if a <= w.time < b), None)
            if seg is not None:
                rows.append(dict(segment=seg, **w.as_row()))
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / f"windows_{name}.csv", index=False)
        if df.empty:
            log.warning("variant %s: no window inside the segments", name)
            continue
        report[name] = summarize(df, segs, cfg)
        report[name]["stride"] = cfg.descriptor.stride
        md += to_markdown(name, report[name])
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(report, fh, indent=2)
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
