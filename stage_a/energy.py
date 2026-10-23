"""Motion-energy map E_t (Sec. III-A-3, Eq. energy_update).

    FD_t = |f_t - f_{t-1}|              (grey frames, consecutive analysis frames)
    E_t  = E_{t-1} + alpha              if FD_t > I_threshold
    E_t  = max(0, E_{t-1} - 0.1 alpha)  otherwise

E is kept as an integer count of decay steps (unit = decay_ratio * alpha), so the map is exact
and the offline cache can store it losslessly as uint16.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


class MotionEnergy:
    def __init__(self, intensity_threshold: int = 15, alpha: float = 1.0, decay_ratio: float = 0.1):
        steps = 1.0 / decay_ratio
        if abs(steps - round(steps)) > 1e-6:
            raise ValueError("decay_ratio must be 1/n for an integer n")
        self.threshold = int(intensity_threshold)
        self.inc = int(round(steps))      # +alpha, in units
        self.unit = alpha * decay_ratio   # value of one unit
        self.reset()

    def reset(self):
        self.prev: Optional[np.ndarray] = None
        self.units: Optional[np.ndarray] = None   # int32 map, E = units * self.unit

    def update(self, frame: np.ndarray) -> np.ndarray:
        """Feed the next analysis frame (BGR or grey); returns the int32 unit map."""
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if self.prev is None or self.prev.shape != grey.shape:
            self.prev = grey
            self.units = np.zeros(grey.shape, np.int32)
            return self.units
        moving = cv2.absdiff(grey, self.prev) > self.threshold
        decayed = self.units - 1
        np.maximum(decayed, 0, out=decayed)                       # static: max(0, E - 0.1 alpha)
        self.units = np.where(moving, self.units + self.inc, decayed)  # moving: E + alpha
        self.prev = grey
        return self.units

    def energy(self) -> np.ndarray:
        return self.units.astype(np.float32) * self.unit


def region_stats(units: np.ndarray, unit: float, roi: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """mu_heat and sigma_heat of E over the pixel set R = [x1, x2) x [y1, y2)."""
    x1, y1, x2, y2 = roi
    patch = units[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0, 0.0
    patch = patch.astype(np.float64) * unit
    return float(patch.mean()), float(patch.std())
