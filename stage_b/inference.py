"""Online use of Stage B behind Stage A (Algorithm 2 of the paper).

Stage A hands over one 16-frame ROI clip per forwarded window. The recognizer
scores it, the decision is appended to the history of its interaction group,
and an alert fires after n_p = 3 consecutive positive decisions; an open alert
absorbs further positives for a 30 s cooldown.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

from .data import IMAGENET_MEAN, IMAGENET_STD
from .engine import build_model


class Recognizer:
    def __init__(self, checkpoint, device="cuda"):
        ckpt = torch.load(checkpoint, map_location="cpu")
        self.cfg = ckpt["config"]
        self.model = build_model(self.cfg, pretrained=False)
        self.model.load_state_dict(ckpt["model"])
        self.model.freeze_backbone()
        self.model.to(device).eval()
        self.device = device
        self.size = self.cfg["data"]["image_size"]

    def preprocess(self, crops):
        """crops: 16 RGB uint8 ROI crops of any size (a list or a (16, H, W, 3) array)."""
        frames = np.stack([
            c if c.shape[:2] == (self.size, self.size)
            else cv2.resize(np.ascontiguousarray(c), (self.size, self.size), interpolation=cv2.INTER_AREA)
            for c in crops
        ])
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255
        return ((x - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(0)

    @torch.no_grad()
    def __call__(self, crops):
        x = self.preprocess(crops).to(self.device)
        with torch.autocast("cuda", torch.float16, enabled=self.device != "cpu"):
            return torch.sigmoid(self.model(x).float()).item()


class GroupRegistry:
    """Stable ids for interaction groups whose membership changes from clip to clip: a clip joins
    the known group it shares the most tracks with, or starts a new one."""

    def __init__(self):
        self.members = {}
        self._next = 0

    def key(self, members):
        m = set(members)
        overlap = {k: len(v & m) for k, v in self.members.items()}
        k = max(overlap, key=overlap.get, default=None)
        if k is None or overlap[k] == 0:
            k, self._next = self._next, self._next + 1
        self.members[k] = m
        return k


@dataclass
class AlertManager:
    persistence: int = 3       # n_p consecutive positive windows
    cooldown: float = 30.0     # seconds
    threshold: float = 0.5
    soft_discount: float = 0.75  # lambda, for clips forwarded from the soft band
    history: dict = field(default_factory=dict)
    open_alerts: dict = field(default_factory=dict)  # group -> deadline

    def update(self, group, prob, now, windows=1, soft=False):
        """Feed one Stage B decision; returns True when a new alert fires.
        `windows` is the number of Stage A windows merged into this clip."""
        thr = self.threshold / self.soft_discount if soft else self.threshold
        positive = prob > thr
        hist = self.history.setdefault(group, [])
        hist.extend([positive] * windows)

        fired = False
        if group in self.open_alerts:
            if positive:
                self.open_alerts[group] = now + self.cooldown
        elif len(hist) >= self.persistence and all(hist[-self.persistence:]):
            self.open_alerts[group] = now + self.cooldown
            fired = True
        self.open_alerts = {g: d for g, d in self.open_alerts.items() if d >= now}
        return fired
