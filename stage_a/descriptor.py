"""Per-person kinematics, interaction groups and the 7-D window descriptor V_raw.

Sec. III-A-2 (per frame, per tracked person, box centre (x_t, y_t), box height H_t):
    v_t     = sqrt((x_t - x_{t-1})^2 + (y_t - y_{t-1})^2) / H_t * FPS   [body-lengths / s]
    a_t     = (v_t - v_{t-1}) * FPS                                   [body-lengths / s^2]
    theta_t = atan2(y_t - y_{t-1}, x_t - x_{t-1})

Sec. III-A-4 (causal window W = {t-N+1 .. t}, group G, S = {(p, tau)}):
    V_max = max_S v              sigma_v = std_W v of p* = argmax member
    mu_a  = mean_S a             A_max   = max_S a
    S_interaction = circular std (Mardia, sqrt(-2 ln R)) of theta over S, in degrees
    mu_heat, sigma_heat = mean / std of E_t over R, the union of member boxes over W, padded 15%

The same ``WindowEngine`` is fed from the offline track cache (training data) and from the live
front end (runtime), so the descriptor is computed identically in both places.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .energy import region_stats
from .grouping import adaptive_groups

FEATURE_NAMES = ("V_max", "sigma_v", "mu_a", "A_max", "S_interaction", "mu_heat", "sigma_heat")
NAN = float("nan")


@dataclass
class _Rec:
    aidx: int
    cx: float
    cy: float
    h: float
    box: np.ndarray
    matched: bool
    v: float = NAN
    a: float = NAN
    theta: float = NAN   # undefined when v is undefined or the centre did not move at all
    run: int = 1         # consecutive analysis frames this track has been present, ending here


@dataclass
class WindowSample:
    aidx: int                          # analysis frame t that closes the window
    time: float                        # t / analysis_fps, seconds
    members: Tuple[int, ...]           # track ids of the group at t
    roi: Tuple[int, int, int, int]     # R in analysis-frame pixels (x1, y1, x2, y2), padded, clipped
    features: np.ndarray               # (7,) in FEATURE_NAMES order
    n_meas: int                        # |S| counted on velocity samples
    n_full: int                        # members whose track spans >= min_history frames

    def as_row(self) -> dict:
        row = {"aidx": self.aidx, "time": round(self.time, 3),
               "members": ";".join(map(str, self.members)), "n_members": len(self.members),
               "n_full": self.n_full, "n_meas": self.n_meas,
               "roi_x1": self.roi[0], "roi_y1": self.roi[1], "roi_x2": self.roi[2], "roi_y2": self.roi[3]}
        row.update({name: float(v) for name, v in zip(FEATURE_NAMES, self.features)})
        return row


def circular_std_deg(angles: Sequence[float]) -> float:
    """Mardia circular standard deviation sqrt(-2 ln R), in degrees (Eq. std_circ)."""
    if len(angles) == 0:
        return 0.0
    ang = np.asarray(angles, np.float64)
    r = abs(np.exp(1j * ang).mean())
    r = min(max(r, 1e-12), 1.0)
    return math.degrees(math.sqrt(-2.0 * math.log(r)))


def pad_box(box: Sequence[float], pad: float, width: int, height: int,
            square: bool = False) -> Tuple[int, int, int, int]:
    """Enlarge a box about its centre by (1 + pad), optionally square it, clip to the frame."""
    x1, y1, x2, y2 = map(float, box)
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    w, h = (x2 - x1) * (1.0 + pad), (y2 - y1) * (1.0 + pad)
    if square:
        w = h = max(w, h)
        w = min(w, width)
        h = min(h, height)
        # shift instead of clipping so the square keeps its size near the border
        cx = min(max(cx, w * 0.5), width - w * 0.5)
        cy = min(max(cy, h * 0.5), height - h * 0.5)
    nx1 = max(0, int(math.floor(cx - w * 0.5)))
    ny1 = max(0, int(math.floor(cy - h * 0.5)))
    nx2 = min(width, int(math.ceil(cx + w * 0.5)))
    ny2 = min(height, int(math.ceil(cy + h * 0.5)))
    return nx1, ny1, max(nx2, nx1 + 1), max(ny2, ny1 + 1)


class WindowEngine:
    """Keeps per-track histories at the analysis rate, forms groups and emits window descriptors."""

    def __init__(self, window: int = 10, stride: int = 5, grouping_k: float = 1.5,
                 min_members: int = 2, min_history: int = 10, roi_pad: float = 0.15,
                 analysis_fps: float = 10.0, use_predicted_boxes: bool = True, keep: int = 32,
                 box_source: str = "kalman"):
        if box_source not in ("kalman", "detection"):
            raise ValueError(f"box_source must be 'kalman' or 'detection', got {box_source!r}")
        self.box_source = box_source
        self.window = window
        self.stride = stride
        self.k = grouping_k
        self.min_members = min_members
        self.min_history = min_history
        self.roi_pad = roi_pad
        self.fps = analysis_fps
        self.use_predicted = use_predicted_boxes
        self.keep = max(keep, window + 2)
        self.reset()

    @classmethod
    def from_cfg(cls, cfg, keep: int = 32) -> "WindowEngine":
        d = cfg.descriptor
        return cls(d.window, d.stride, d.grouping_k, d.min_members, d.min_history, d.roi_pad,
                   cfg.frontend.analysis_fps, d.use_predicted_boxes, keep, d.get("box_source", "kalman"))

    def reset(self):
        self.hist: Dict[int, deque] = {}
        self.groups: List[List[int]] = []
        self.aidx: Optional[int] = None

    # ------------------------------------------------------------------ per analysis frame
    def observe(self, aidx: int, tracks: Iterable) -> List[List[int]]:
        """Record the tracks of analysis frame ``aidx`` and regroup. Returns all groups (with singletons)."""
        fps = self.fps
        for obs in tracks:
            if not obs.matched and not self.use_predicted:
                continue
            kbox = getattr(obs, "kbox", None)
            src = kbox if (self.box_source == "kalman" and kbox is not None) else obs.box
            box = np.asarray(src, np.float32)
            h = max(float(box[3] - box[1]), 1.0)
            cx, cy = float(box[0] + box[2]) * 0.5, float(box[1] + box[3]) * 0.5
            dq = self.hist.get(obs.tid)
            if dq is None:
                dq = self.hist[obs.tid] = deque(maxlen=self.keep)
            elif dq[-1].aidx == aidx:
                continue
            rec = _Rec(aidx, cx, cy, h, box, bool(obs.matched))
            if dq and dq[-1].aidx == aidx - 1:
                prev = dq[-1]
                dx, dy = cx - prev.cx, cy - prev.cy
                rec.v = math.hypot(dx, dy) / h * fps
                rec.theta = math.atan2(dy, dx) if (dx != 0.0 or dy != 0.0) else NAN
                if not math.isnan(prev.v):
                    rec.a = (rec.v - prev.v) * fps
                rec.run = prev.run + 1
            dq.append(rec)
        for tid in [t for t, dq in self.hist.items() if dq[-1].aidx < aidx - self.keep]:
            del self.hist[tid]
        present = [t for t, dq in self.hist.items() if dq[-1].aidx == aidx]
        comps = adaptive_groups(np.array([self.hist[t][-1].box for t in present]).reshape(-1, 4), self.k)
        self.groups = [[present[i] for i in comp] for comp in comps]
        self.aidx = aidx
        return self.groups

    def is_eval_frame(self, aidx: int) -> bool:
        return aidx % self.stride == 0

    def admitted(self, members: Sequence[int]) -> bool:
        if len(members) < self.min_members:
            return False
        full = sum(self.hist[p][-1].run >= self.min_history for p in members)
        return full >= self.min_members

    def windows(self, aidx: int, energy_units: Optional[np.ndarray], energy_unit: float,
                frame_hw: Tuple[int, int]) -> List[WindowSample]:
        """Descriptors of every admitted group at ``aidx`` (call after ``observe``)."""
        if not self.is_eval_frame(aidx):
            return []
        out = []
        for members in self.groups:
            if not self.admitted(members):
                continue
            sample = self._describe(aidx, members, energy_units, energy_unit, frame_hw)
            if sample is not None:
                out.append(sample)
        return out

    # ------------------------------------------------------------------ helpers
    def member_box_union(self, members: Iterable[int], lo: int, hi: int) -> Optional[np.ndarray]:
        """Union of the members' boxes over analysis frames lo..hi (inclusive)."""
        boxes = [r.box for p in members if p in self.hist for r in self.hist[p] if lo <= r.aidx <= hi]
        if not boxes:
            return None
        b = np.stack(boxes)
        return np.array([b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()], np.float32)

    def _describe(self, aidx, members, units, unit, frame_hw) -> Optional[WindowSample]:
        lo = aidx - self.window + 1
        v_all, a_all, th_all, boxes = [], [], [], []
        best_v, best_p = -1.0, None
        per_v = {}
        for p in members:
            recs = [r for r in self.hist[p] if r.aidx >= lo]
            boxes.extend(r.box for r in recs)
            vs = [r.v for r in recs if not math.isnan(r.v)]
            per_v[p] = vs
            v_all.extend(vs)
            a_all.extend(r.a for r in recs if not math.isnan(r.a))
            th_all.extend(r.theta for r in recs if not math.isnan(r.theta))
            if vs and max(vs) > best_v:
                best_v, best_p = max(vs), p
        if not v_all:
            return None
        sigma_v = float(np.std(per_v[best_p]))
        mu_a = float(np.mean(a_all)) if a_all else 0.0
        a_max = float(np.max(a_all)) if a_all else 0.0
        s_int = circular_std_deg(th_all)
        b = np.stack(boxes)
        union = (b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max())
        height, width = frame_hw
        roi = pad_box(union, self.roi_pad, width, height)
        if units is not None:
            mu_heat, sigma_heat = region_stats(units, unit, roi)
        else:
            mu_heat, sigma_heat = NAN, NAN
        feats = np.array([best_v, sigma_v, mu_a, a_max, s_int, mu_heat, sigma_heat], np.float64)
        n_full = sum(self.hist[p][-1].run >= self.min_history for p in members)
        return WindowSample(aidx, aidx / self.fps, tuple(sorted(members)), roi, feats, len(v_all), n_full)
