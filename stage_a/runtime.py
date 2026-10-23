"""Online Stage A (Algorithm 1): one decoded frame in, forwarded 16-frame ROI clips out.

Per frame: YOLOv8 + DeepSORT. Per analysis frame (10 FPS): motion-energy update, track
histories, adaptive grouping. Every ``stride`` analysis frames: window descriptors, physics gate,
SVM gate, then merging of forwarded windows into Stage B clips that are cut once their last
frame (trigger + 7) has arrived.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .clips import ClipMerger, PendingClip
from .descriptor import FEATURE_NAMES, WindowEngine, WindowSample, pad_box
from .energy import MotionEnergy
from .frontend import FrontEnd, TrackObs
from .gate import StageAGate
from .video import FramePacket

# group status, in the order a window meets the gates
SINGLE, WARMING, REJ_PHYSICS, REJ_SVM, FORWARDED = "single", "warming", "rejected_physics", "rejected_svm", "forwarded"


def letterbox_square(img: np.ndarray) -> np.ndarray:
    """Pad to a square with black borders (a ROI clipped by the frame edge cannot be squared by growing)."""
    h, w = img.shape[:2]
    if h == w:
        return img
    s = max(h, w)
    out = np.zeros((s, s) + img.shape[2:], img.dtype)
    y0, x0 = (s - h) // 2, (s - w) // 2
    out[y0:y0 + h, x0:x0 + w] = img
    return out


def roi_boxes(union: np.ndarray, analysis_hw: Tuple[int, int], source_hw: Tuple[int, int], cfg):
    """ROI of a Stage B clip from the union of the group's boxes (analysis-frame pixels): padded,
    squared, clipped. Returns (roi in analysis pixels, roi in the pixels of the frames to crop)."""
    rc = cfg.roi_clip
    ah, aw = analysis_hw
    # a coasting Kalman box can drift off the frame or invert; fall back to the whole frame
    x1, y1, x2, y2 = np.clip(np.asarray(union, np.float64), 0, [aw, ah, aw, ah])
    union = (x1, y1, x2, y2) if x2 - x1 >= 1 and y2 - y1 >= 1 else (0, 0, aw, ah)
    roi_a = pad_box(union, cfg.descriptor.roi_pad, aw, ah, square=rc.square)
    if rc.crop_from != "original":
        return roi_a, roi_a
    h, w = source_hw
    s = w / float(aw)
    return roi_a, pad_box(np.array(roi_a, np.float64) * s, 0.0, w, h, square=False)


def crop_roi(frame: np.ndarray, roi: Tuple[int, int, int, int], cfg) -> np.ndarray:
    """One frame of a Stage B clip: crop, letterbox to a square, resize to roi_clip.size."""
    rc = cfg.roi_clip
    x1, y1, x2, y2 = roi
    crop = frame[y1:y2, x1:x2]
    if rc.square:
        crop = letterbox_square(crop)
    interp = cv2.INTER_AREA if crop.shape[1] > rc.size else cv2.INTER_LINEAR
    return cv2.resize(crop, (rc.size, rc.size), interpolation=interp)


@dataclass
class GroupState:
    members: Tuple[int, ...]
    box: np.ndarray                     # current union box, analysis-frame pixels
    status: str
    window: Optional[WindowSample] = None
    score: float = float("nan")


@dataclass
class ForwardedClip:
    clip_no: int
    anchor_aidx: int
    first_aidx: int
    last_aidx: int
    t_start: float
    t_end: float
    members: Tuple[int, ...]
    roi_analysis: Tuple[int, int, int, int]
    roi_source: Tuple[int, int, int, int]
    frames: np.ndarray                  # (T, size, size, 3) uint8, BGR
    triggers: List[dict] = field(default_factory=list)
    padded_frames: int = 0              # frames repeated because the stream ended or started early
    stage_b_prob: Optional[float] = None

    def to_meta(self) -> dict:
        return {"clip_no": self.clip_no, "anchor_aidx": self.anchor_aidx, "first_aidx": self.first_aidx,
                "last_aidx": self.last_aidx, "t_start": round(self.t_start, 3), "t_end": round(self.t_end, 3),
                "members": list(self.members), "roi_analysis": list(self.roi_analysis),
                "roi_source": list(self.roi_source), "n_frames": int(len(self.frames)),
                "padded_frames": self.padded_frames, "n_merged_windows": len(self.triggers),
                "triggers": self.triggers, "stage_b_prob": self.stage_b_prob}


@dataclass
class StepResult:
    packet: FramePacket
    tracks: List[TrackObs]
    analysis: bool = False
    groups: List[GroupState] = field(default_factory=list)
    windows: List[WindowSample] = field(default_factory=list)
    forwarded: List[WindowSample] = field(default_factory=list)
    clips: List[ForwardedClip] = field(default_factory=list)


class Timer:
    def __init__(self):
        self.total: Dict[str, float] = {}
        self.count: Dict[str, int] = {}

    def add(self, key: str, seconds: float):
        self.total[key] = self.total.get(key, 0.0) + seconds
        self.count[key] = self.count.get(key, 0) + 1

    def summary(self) -> Dict[str, dict]:
        return {k: {"mean_ms": 1000 * self.total[k] / max(self.count[k], 1), "calls": self.count[k]}
                for k in self.total}


class StageARuntime:
    def __init__(self, cfg, gate: StageAGate, device: str = "cuda:0",
                 stage_b: Optional[Callable[[np.ndarray], float]] = None):
        self.cfg = cfg
        self.gate = gate
        self.stage_b = stage_b
        self.frontend = FrontEnd(cfg.frontend, device)
        en = cfg.energy
        self.energy = MotionEnergy(en.intensity_threshold, en.alpha, en.decay_ratio)
        rc = cfg.roi_clip
        self.before, self.after = rc.frames_before, rc.frames_after
        keep = max(32, cfg.descriptor.window + self.before + self.after + 4)
        self.engine = WindowEngine.from_cfg(cfg, keep=keep)
        self.merger = ClipMerger(self.before, self.after)
        self.buffer: deque = deque(maxlen=self.before + self.after + 8)
        self.timer = Timer()
        self.status: Dict[frozenset, Tuple[str, Optional[WindowSample], float]] = {}
        self.last_time = 0.0
        self.scale = 1.0

    def reset(self):
        self.frontend.reset()
        self.energy.reset()
        self.engine.reset()
        self.merger = ClipMerger(self.before, self.after)
        self.buffer.clear()
        self.status.clear()

    # ------------------------------------------------------------------ main step
    def process(self, pkt: FramePacket) -> StepResult:
        t0 = time.perf_counter()
        dets = self.frontend.detector([pkt.resized])[0]
        t1 = time.perf_counter()
        tracks = self.frontend.tracker.update(dets, pkt.resized)
        t2 = time.perf_counter()
        self.timer.add("detect", t1 - t0)
        self.timer.add("track", t2 - t1)
        res = StepResult(pkt, tracks)
        self.last_time = pkt.time
        self.scale = pkt.resized.shape[1] / pkt.frame.shape[1]
        if pkt.analysis_index is None:
            return res
        k = pkt.analysis_index
        res.analysis = True
        units = self.energy.update(pkt.resized)
        t3 = time.perf_counter()
        self.timer.add("energy", t3 - t2)
        src = pkt.frame if self.cfg.roi_clip.crop_from == "original" else pkt.resized
        self.buffer.append((k, pkt.time, src))
        groups = self.engine.observe(k, tracks)
        hw = pkt.resized.shape[:2]
        windows = self.engine.windows(k, units, self.energy.unit, hw)
        res.windows = windows
        if self.engine.is_eval_frame(k):
            self.status = {}
            if windows:
                X = np.stack([w.features for w in windows])
                dec = self.gate.decide(X)
                for i, w in enumerate(windows):
                    st = FORWARDED if dec.forward[i] else (REJ_SVM if dec.physics[i] else REJ_PHYSICS)
                    self.status[frozenset(w.members)] = (st, w, float(dec.score[i]))
                    if dec.forward[i]:
                        res.forwarded.append(w)
                        self.merger.add(k, w.members, float(dec.score[i]), payload=w)
        t4 = time.perf_counter()
        self.timer.add("features_gate", t4 - t3)
        res.groups = self._group_states(groups, k)
        res.clips = [self._cut(c) for c in self.merger.pop_ready(k)]
        self.timer.add("clip_cut", time.perf_counter() - t4)
        return res

    def flush(self) -> List[ForwardedClip]:
        """End of stream: cut the clips that are still waiting for future frames."""
        return [self._cut(c) for c in self.merger.pop_ready(None)]

    # ------------------------------------------------------------------ helpers
    def _group_states(self, groups: List[List[int]], k: int) -> List[GroupState]:
        out = []
        for members in groups:
            box = self.engine.member_box_union(members, k, k)
            if box is None:
                continue
            key = frozenset(members)
            if len(members) < self.engine.min_members:
                st, w, s = SINGLE, None, float("nan")
            elif key in self.status:
                st, w, s = self.status[key]
            elif not self.engine.admitted(members):
                st, w, s = WARMING, None, float("nan")
            else:  # admitted, between two evaluation frames: show the last decision of an overlapping group
                prev = [v for kk, v in self.status.items() if kk & key]
                st, w, s = prev[0] if prev else (WARMING, None, float("nan"))
            out.append(GroupState(tuple(sorted(members)), box, st, w, s))
        return out

    def _cut(self, clip: PendingClip) -> ForwardedClip:
        rc = self.cfg.roi_clip
        frames_by_k = {k: (t, f) for k, t, f in self.buffer}
        union = self.engine.member_box_union(clip.members, clip.lo, clip.hi)
        if union is None:
            union = np.array(clip.payload[0].roi, np.float32)
        ref = next(iter(frames_by_k.values()))[1]
        if rc.crop_from == "original":
            analysis_hw = (int(round(ref.shape[0] * self.scale)), int(round(ref.shape[1] * self.scale)))
        else:
            analysis_hw = ref.shape[:2]
        roi_a, roi_s = roi_boxes(union, analysis_hw, ref.shape[:2], self.cfg)
        available = sorted(frames_by_k)
        crops, padded, times = [], 0, []
        for k in range(clip.lo, clip.hi + 1):
            if k in frames_by_k:
                t, f = frames_by_k[k]
            else:  # before the stream start or after its end: repeat the nearest frame
                near = min(available, key=lambda a: abs(a - k))
                t, f = frames_by_k[near]
                padded += 1
            times.append(t)
            crops.append(crop_roi(f, roi_s, self.cfg))
        triggers = []
        for (aidx, members, score), w in zip(clip.triggers, clip.payload):
            triggers.append({"aidx": aidx, "time": round(aidx / self.engine.fps, 3), "members": list(members),
                             "score": round(score, 4), "roi": list(w.roi),
                             "features": {n: round(float(v), 4) for n, v in zip(FEATURE_NAMES, w.features)}})
        out = ForwardedClip(clip.clip_no, clip.anchor, clip.lo, clip.hi, min(times), max(times),
                            tuple(sorted(clip.members)), tuple(roi_a), tuple(roi_s), np.stack(crops),
                            triggers, padded)
        if self.stage_b is not None:
            out.stage_b_prob = float(self.stage_b(out.frames))
        return out
