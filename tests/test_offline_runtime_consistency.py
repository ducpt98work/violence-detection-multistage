"""Integration check: the offline path used to build the training data (track_clip -> cache ->
replay_windows) and the online runtime (StageARuntime, frame by frame) must give the same windows
for the same video, since the gate is trained on the first and deployed on the second.

Needs a GPU, the detector weights and the AVD data. Run:
  python tests/test_offline_runtime_consistency.py [video ...]
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stage_a.config import PROJECT_ROOT, load_config  # noqa: E402
from stage_a.gate import GateDecision, physics_pass  # noqa: E402

DEFAULT_VIDEOS = [
    "data/A-Dataset-for-Automatic-Violence-Detection-in-Videos-master/violence-detection-dataset/violent/cam1/12.mp4",
    "data/hockey_fight/data/fi10_xvid.avi",
]


class ForwardAllGate:
    """Stand-in for a trained gate: forwards every window that passes the physics threshold."""
    threshold = 0.0

    def decide(self, X):
        phys = physics_pass(X)
        return GateDecision(phys, np.where(phys, 1.0, np.nan), phys)


def offline_windows(path, cfg, fe):
    from stage_a.offline import TrackCache, replay_windows, save_cache, track_clip

    with tempfile.TemporaryDirectory() as tmp:
        cache_file = Path(tmp) / "c.npz"
        save_cache(cache_file, track_clip(path, fe, cfg))     # exercises the on-disk round trip too
        windows, _ = replay_windows(TrackCache.load(cache_file), cfg)
    return windows


def runtime_windows(path, cfg, rt):
    from stage_a.video import VideoReader

    rt.reset()
    out = []
    with VideoReader(path, cfg.frontend.resize_width, cfg.frontend.analysis_fps, cfg.frontend.max_seconds) as reader:
        for pkt in reader:
            out.extend(rt.process(pkt).windows)
    return out


def gate_check(videos, cfg1, frontend, model_path) -> bool:
    """With the trained gate: the windows the runtime forwards are the windows the offline path
    (the data the gate was evaluated on) would forward."""
    from stage_a.gate import StageAGate
    from stage_a.runtime import StageARuntime
    from stage_a.video import VideoReader

    gate = StageAGate.load(model_path)
    rt = StageARuntime(cfg1, gate, "cuda:0")
    rt.frontend = frontend
    ok = True
    for v in videos:
        path = PROJECT_ROOT / v
        off = offline_windows(path, cfg1, frontend)
        dec = gate.decide(np.stack([w.features for w in off])) if off else None
        off_fwd = {(w.aidx, w.members) for i, w in enumerate(off) if dec.forward[i]} if off else set()
        rt.reset()
        on_fwd, n_clips = set(), 0
        with VideoReader(path, cfg1.frontend.resize_width, cfg1.frontend.analysis_fps, cfg1.frontend.max_seconds) as reader:
            for pkt in reader:
                res = rt.process(pkt)
                on_fwd |= {(w.aidx, w.members) for w in res.forwarded}
                n_clips += len(res.clips)
        n_clips += len(rt.flush())
        good = off_fwd == on_fwd
        ok &= good
        print(f"{'OK' if good else 'MISMATCH'} trained gate | {v}: forwarded offline {len(off_fwd)} / runtime "
              f"{len(on_fwd)} windows, identical: {good}; runtime cut {n_clips} Stage B clips")
    return ok


def main(videos):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    from stage_a.frontend import limit_cpu_threads
    from stage_a.runtime import StageARuntime

    limit_cpu_threads(2)
    cfg = load_config()
    cfg1 = load_config(overrides=["frontend.batch_size=1"])
    rt = StageARuntime(cfg, ForwardAllGate(), "cuda:0")
    key = lambda w: (w.aidx, w.members)  # noqa: E731
    ok = True
    model = PROJECT_ROOT / "models" / "stage_a_gate.joblib"
    if model.exists():
        ok &= gate_check(videos, cfg1, rt.frontend, model)
    for v in videos:
        path = PROJECT_ROOT / v
        on = {key(w): w for w in runtime_windows(path, cfg, rt)}
        # batch 1 offline = the runtime's own call pattern: must agree exactly;
        # batch 16 offline (used to build the data) may differ only by GPU float noise
        for name, c, tol in (("offline batch 1", cfg1, 1e-6), ("offline batch 16", cfg, 1e-3)):
            off = {key(w): w for w in offline_windows(path, c, rt.frontend)}
            same_keys = off.keys() == on.keys()
            common = sorted(off.keys() & on.keys())
            rel = max((float(np.max(np.abs(off[k].features - on[k].features) / np.maximum(1.0, np.abs(on[k].features))))
                       for k in common), default=0.0)
            roi_same = all(off[k].roi == on[k].roi for k in common)
            good = same_keys and rel <= tol and roi_same
            ok &= good
            print(f"{'OK' if good else 'MISMATCH'} {name} vs runtime | {v}: {len(off)} / {len(on)} windows, "
                  f"same (aidx, group) set: {same_keys}, max relative feature diff {rel:.1e} (tol {tol:g}), "
                  f"identical ROIs: {roi_same}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1:] or DEFAULT_VIDEOS) else 1)
