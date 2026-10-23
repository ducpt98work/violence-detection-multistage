"""The two cascaded gates of Stage A (Sec. III-A-5): physics threshold, then RBF-SVM.

The trained gate is stored as one joblib bundle that also carries the configuration it was
trained with, so the runtime rebuilds exactly the same front end and descriptor.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from .descriptor import FEATURE_NAMES

I_A_MAX = FEATURE_NAMES.index("A_max")
I_MU_HEAT = FEATURE_NAMES.index("mu_heat")


def physics_pass(X: np.ndarray, a_max: float = 15.0, mu_heat: float = 2.0) -> np.ndarray:
    """Retain a window if A_max > 15.0 OR mu_heat > 2.0."""
    X = np.atleast_2d(X)
    return (X[:, I_A_MAX] > a_max) | (X[:, I_MU_HEAT] > mu_heat)


@dataclass
class GateDecision:
    physics: np.ndarray    # bool, passed the physics threshold
    score: np.ndarray      # signed SVM margin (nan where physics rejected and not scored)
    forward: np.ndarray    # bool, passed both gates


class StageAGate:
    def __init__(self, pipeline, threshold: float, a_max: float, mu_heat: float,
                 config: Optional[dict] = None, meta: Optional[dict] = None):
        self.pipeline = pipeline          # StandardScaler + SVC(kernel="rbf")
        self.threshold = float(threshold)  # operating point on the signed margin
        self.a_max = float(a_max)
        self.mu_heat = float(mu_heat)
        self.config = config or {}
        self.meta = meta or {}

    def physics(self, X: np.ndarray) -> np.ndarray:
        return physics_pass(X, self.a_max, self.mu_heat)

    def score(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, np.float64))
        if len(X) == 0:
            return np.zeros(0)
        return self.pipeline.decision_function(X)

    def decide(self, X: np.ndarray, threshold: Optional[float] = None) -> GateDecision:
        X = np.atleast_2d(np.asarray(X, np.float64))
        thr = self.threshold if threshold is None else threshold
        phys = self.physics(X)
        score = np.full(len(X), np.nan)
        if phys.any():
            score[phys] = self.score(X[phys])
        forward = phys & (np.nan_to_num(score, nan=-np.inf) >= thr)
        return GateDecision(phys, score, forward)

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "threshold": self.threshold,
                     "physics": {"a_max": self.a_max, "mu_heat": self.mu_heat},
                     "feature_names": list(FEATURE_NAMES), "config": self.config,
                     "meta": self.meta}, path)

    @classmethod
    def load(cls, path: str | Path) -> "StageAGate":
        b = joblib.load(path)
        if list(b["feature_names"]) != list(FEATURE_NAMES):
            raise ValueError(f"feature order mismatch in {path}: {b['feature_names']}")
        return cls(b["pipeline"], b["threshold"], b["physics"]["a_max"], b["physics"]["mu_heat"],
                   b.get("config"), b.get("meta"))
