"""Offline front-end pass and window replay.

``track_clip`` runs the expensive part once per video (decode, resize, YOLOv8 + DeepSORT on every
frame, motion-energy map at the analysis rate) and stores the result in a compressed cache:
tracks at every analysis frame and E_t at every window-evaluation frame.

``replay_windows`` feeds that cache through the same ``WindowEngine`` the runtime uses, so the
grouping / descriptor settings can be changed without re-running the detector.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from .config import display_path
from .descriptor import WindowEngine, WindowSample
from .energy import MotionEnergy
from .frontend import TrackObs
from .video import VideoReader

CACHE_VERSION = 2
# box = associated detection (Kalman box while coasting); k* = Kalman state box of the tracker
TRACK_COLS = ("aidx", "tid", "x1", "y1", "x2", "y2", "conf", "matched", "tsu", "kx1", "ky1", "kx2", "ky2")


def track_clip(path: str | Path, frontend, cfg) -> dict:
    """Front-end pass over one video. Returns the arrays that ``save_cache`` writes."""
    fe = cfg.frontend
    en = cfg.energy
    energy = MotionEnergy(en.intensity_threshold, en.alpha, en.decay_ratio)
    frontend.reset()
    rows: List[list] = []
    a_idx, a_frame, a_time = [], [], []
    e_idx, e_maps = [], []
    n_det = []
    batch = []
    last_index = -1

    def flush():
        if not batch:
            return
        frames = [p.resized for p in batch]
        dets = frontend.detector(frames)
        embeds = frontend.tracker.embed_batch(frames, dets)
        for pkt, d, e, frame in zip(batch, dets, embeds, frames):
            tracks = frontend.tracker.update(d, frame, e)
            if pkt.analysis_index is None:
                continue
            k = pkt.analysis_index
            a_idx.append(k)
            a_frame.append(pkt.index)
            a_time.append(pkt.time)
            n_det.append(len(d))
            for o in tracks:
                rows.append([k, o.tid, *o.box.tolist(), o.conf, float(o.matched), o.tsu, *o.kbox.tolist()])
            units = energy.update(frame)
            if k % en.store_every == 0:
                e_idx.append(k)
                e_maps.append(np.minimum(units, 65535).astype(np.uint16))
        batch.clear()

    with VideoReader(path, fe.resize_width, fe.analysis_fps, fe.max_seconds) as reader:
        for pkt in reader:
            batch.append(pkt)
            last_index = pkt.index
            if len(batch) >= fe.batch_size:
                flush()
        flush()
        decoded = last_index + 1
        meta = {
            "version": CACHE_VERSION, "path": display_path(path), "fps": reader.fps,
            "width": reader.width, "height": reader.height, "scale": reader.scale,
            "frame_count_meta": reader.frame_count,
            "duration_meta": reader.frame_count / reader.fps if reader.frame_count else None,
            "decoded_frames": decoded, "decoded_seconds": decoded / reader.fps,
            "truncated": bool(fe.max_seconds and decoded / reader.fps >= fe.max_seconds - 1.0 / reader.fps),
            "analysis_fps": fe.analysis_fps, "resize_width": fe.resize_width,
            "energy_unit": en.alpha * en.decay_ratio, "energy_store_every": en.store_every,
            "frontend": cfg.frontend.to_dict() if hasattr(cfg.frontend, "to_dict") else dict(cfg.frontend),
            "energy": cfg.energy.to_dict() if hasattr(cfg.energy, "to_dict") else dict(cfg.energy),
        }
    if e_maps:
        meta["analysis_hw"] = list(e_maps[0].shape)
    return {
        "meta": meta,
        "a_idx": np.asarray(a_idx, np.int32), "a_frame": np.asarray(a_frame, np.int32),
        "a_time": np.asarray(a_time, np.float32), "n_det": np.asarray(n_det, np.int16),
        "tracks": np.asarray(rows, np.float32).reshape(-1, len(TRACK_COLS)),
        "e_idx": np.asarray(e_idx, np.int32),
        "energy": np.stack(e_maps) if e_maps else np.zeros((0, 1, 1), np.uint16),
    }


def save_cache(path: str | Path, data: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    payload = dict(data)
    payload["meta"] = np.frombuffer(json.dumps(data["meta"]).encode(), np.uint8)
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)


@dataclass
class TrackCache:
    meta: dict
    a_idx: np.ndarray
    a_frame: np.ndarray
    a_time: np.ndarray
    n_det: np.ndarray
    tracks: np.ndarray
    e_idx: np.ndarray
    energy: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> "TrackCache":
        with np.load(path) as z:
            meta = json.loads(bytes(z["meta"]).decode())
            return cls(meta, z["a_idx"], z["a_frame"], z["a_time"], z["n_det"], z["tracks"],
                       z["e_idx"], z["energy"])

    @property
    def analysis_hw(self) -> Tuple[int, int]:
        if "analysis_hw" in self.meta:
            return tuple(self.meta["analysis_hw"])
        w = self.meta["resize_width"]
        return int(round(self.meta["height"] * w / max(self.meta["width"], 1))), w

    def iter_frames(self) -> Iterator[Tuple[int, List[TrackObs], Optional[np.ndarray]]]:
        """Yields (aidx, tracks, energy units or None) in analysis order."""
        energy_at = {int(k): i for i, k in enumerate(self.e_idx)}
        order = np.argsort(self.tracks[:, 0], kind="stable") if len(self.tracks) else np.zeros(0, int)
        tr = self.tracks[order]
        bounds = np.searchsorted(tr[:, 0], self.a_idx, side="left"), np.searchsorted(tr[:, 0], self.a_idx, side="right")
        has_kalman = tr.shape[1] >= 13  # caches without the Kalman columns fall back to the detection box
        for k, lo, hi in zip(self.a_idx, *bounds):
            obs = [TrackObs(int(r[1]), r[2:6].copy(), float(r[6]), bool(r[7] > 0.5), int(r[8]),
                            r[9:13].copy() if has_kalman else r[2:6].copy()) for r in tr[lo:hi]]
            i = energy_at.get(int(k))
            yield int(k), obs, (self.energy[i] if i is not None else None)

    def max_track_run(self) -> int:
        """Longest run of consecutive analysis frames held by one track id."""
        best = 0
        if not len(self.tracks):
            return 0
        for tid in np.unique(self.tracks[:, 1]):
            ks = np.unique(self.tracks[self.tracks[:, 1] == tid, 0]).astype(int)
            if len(ks) == 0:
                continue
            breaks = np.flatnonzero(np.diff(ks) != 1)
            runs = np.diff(np.concatenate([[-1], breaks, [len(ks) - 1]]))
            best = max(best, int(runs.max()))
        return best


def replay_windows(cache: TrackCache, cfg, max_time: Optional[float] = None) -> Tuple[List[WindowSample], Dict]:
    """Run the window engine over a cached clip. Returns the windows and per-clip statistics."""
    engine = WindowEngine.from_cfg(cfg)
    unit = cache.meta["energy_unit"]
    store_every = cache.meta["energy_store_every"]
    if engine.stride % store_every:
        raise ValueError(f"stride {engine.stride} is not a multiple of the cached energy step {store_every}")
    hw = cache.analysis_hw
    windows: List[WindowSample] = []
    eligible = covered = 0
    max_members = 0
    for aidx, obs, units in cache.iter_frames():
        if max_time is not None and aidx / engine.fps > max_time:
            break
        groups = engine.observe(aidx, obs)
        max_members = max([max_members] + [len(g) for g in groups])
        if not engine.is_eval_frame(aidx):
            continue
        ws = engine.windows(aidx, units, unit, hw)
        windows.extend(ws)
        if aidx >= engine.min_history:
            eligible += 1
            covered += int(bool(ws))
    stats = {
        "n_analysis": int(len(cache.a_idx)),
        "n_windows": len(windows),
        "eval_instants": eligible,
        "window_coverage": covered / eligible if eligible else 0.0,
        "max_track_run": cache.max_track_run(),
        "max_group_size": max_members,
        "n_track_ids": int(len(np.unique(cache.tracks[:, 1]))) if len(cache.tracks) else 0,
    }
    return windows, stats
