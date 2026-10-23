"""Person detection (YOLOv8) and tracking (DeepSORT), run on every decoded frame.

Boxes are in the coordinates of the frame passed in, i.e. the W_target-wide analysis frame.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
import os
from typing import List, Optional, Sequence

import numpy as np

from .config import resolve_path

log = logging.getLogger(__name__)

PERSON_CLASS = 0


@dataclass
class TrackObs:
    """One confirmed track in one frame."""
    tid: int
    box: np.ndarray      # associated detection x1, y1, x2, y2 if matched, else the Kalman box (float32)
    conf: float          # detector confidence if matched this frame, else nan
    matched: bool        # True if a detection was associated in this frame
    tsu: int             # DeepSORT time_since_update, in source frames
    kbox: Optional[np.ndarray] = None   # the tracker's Kalman state box (smoothed), x1, y1, x2, y2


class PersonDetector:
    def __init__(self, weights: str, conf: float = 0.25, iou: float = 0.45, imgsz: int = 640,
                 half: bool = False, device: str = "cuda:0"):
        from ultralytics import YOLO

        path = resolve_path(weights)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Ultralytics downloads a known asset name (yolov8s.pt) to the given path if missing.
        self.model = YOLO(str(path))
        self.kw = dict(imgsz=imgsz, conf=conf, iou=iou, classes=[PERSON_CLASS], half=half,
                       device=device, verbose=False)

    def __call__(self, frames: Sequence[np.ndarray]) -> List[np.ndarray]:
        """Returns one (n, 5) array [x1, y1, x2, y2, conf] per frame."""
        if not len(frames):
            return []
        results = self.model.predict(list(frames), **self.kw)
        out = []
        for res in results:
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                out.append(np.zeros((0, 5), np.float32))
                continue
            xyxy = boxes.xyxy.float().cpu().numpy()
            conf = boxes.conf.float().cpu().numpy()[:, None]
            out.append(np.concatenate([xyxy, conf], axis=1).astype(np.float32))
        return out


class PersonTracker:
    def __init__(self, max_age: int = 30, n_init: int = 3, max_cosine_distance: float = 0.2,
                 nn_budget: int = 100, max_iou_distance: float = 0.7, embedder: str = "mobilenet",
                 half: bool = False, gpu: bool = True):
        self.kw = dict(max_age=max_age, n_init=n_init, max_cosine_distance=max_cosine_distance,
                       nn_budget=nn_budget, max_iou_distance=max_iou_distance, embedder=embedder,
                       half=half, bgr=True, embedder_gpu=gpu)
        self.reset()

    def reset(self):
        from deep_sort_realtime.deepsort_tracker import DeepSort
        if getattr(self, "_ds", None) is None:
            self._ds = DeepSort(**self.kw)
            self._ds.embedder.max_batch_size = 64
        else:  # keep the loaded embedder, drop all tracks and appearance samples
            self._ds.delete_all_tracks()
            self._ds.tracker.metric.samples = {}

    @staticmethod
    def _raw(dets: np.ndarray) -> list:
        # boxes under one pixel would give an empty crop; DeepSORT would also drop them
        return [([float(x1), float(y1), float(x2 - x1), float(y2 - y1)], float(c), "person")
                for x1, y1, x2, y2, c in dets if x2 - x1 >= 1 and y2 - y1 >= 1]

    def embed_batch(self, frames: Sequence[np.ndarray], dets: Sequence[np.ndarray]) -> List[list]:
        """Appearance embeddings for the detections of several frames in one forward pass
        (same crops and network as DeepSORT's own per-frame call, far fewer kernel launches)."""
        import torch

        crops, counts = [], []
        for frame, d in zip(frames, dets):
            c, _ = self._ds.crop_bb(frame, self._raw(d))
            crops.extend(c)
            counts.append(len(c))
        with torch.inference_mode():  # the library's predict() keeps the autograd graph otherwise
            feats = self._ds.embedder.predict(crops) if crops else []
        out, i = [], 0
        for n in counts:
            out.append(list(feats[i:i + n]))
            i += n
        return out

    def update(self, dets: np.ndarray, frame: np.ndarray, embeds: Optional[list] = None) -> List[TrackObs]:
        import torch

        raw = self._raw(dets)
        with torch.inference_mode():
            if embeds is None:
                tracks = self._ds.update_tracks(raw, frame=frame)
            else:
                tracks = self._ds.update_tracks(raw, embeds=embeds, frame=frame)
        out = []
        for trk in tracks:
            if not trk.is_confirmed():
                continue
            matched = trk.time_since_update == 0
            box = trk.to_ltrb(orig=True)  # associated detection if matched, else Kalman mean
            kbox = trk.to_ltrb()          # Kalman state mean
            conf = trk.get_det_conf() if matched else None
            out.append(TrackObs(int(trk.track_id), np.asarray(box, np.float32),
                                float("nan") if conf is None else float(conf),
                                bool(matched), int(trk.time_since_update), np.asarray(kbox, np.float32)))
        return out


class FrontEnd:
    """YOLOv8 + DeepSORT. ``process`` takes consecutive frames of one stream."""

    def __init__(self, cfg, device: str = "cuda:0"):
        det = cfg.detector
        trk = cfg.tracker
        self.detector = PersonDetector(det.weights, det.conf, det.iou, det.imgsz, det.half, device)
        self.tracker = PersonTracker(trk.max_age, trk.n_init, trk.max_cosine_distance, trk.nn_budget,
                                     trk.max_iou_distance, trk.embedder, trk.half,
                                     gpu=device.startswith("cuda"))

    def reset(self):
        self.tracker.reset()

    def process(self, frames: Sequence[np.ndarray]) -> List[List[TrackObs]]:
        """Detect on a batch of consecutive frames, then track them in order."""
        dets = self.detector(frames)
        embeds = self.tracker.embed_batch(frames, dets)
        return [self.tracker.update(d, f, e) for d, f, e in zip(dets, frames, embeds)]


def limit_cpu_threads(n: int):
    """Small per-crop tensors make torch/OpenCV thread pools a net loss; several worker processes
    sharing the CPU make it worse. Call before heavy work starts."""
    import cv2
    import torch

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(n)
    torch.set_num_threads(n)
    cv2.setNumThreads(n)
    try:  # BLAS pools that numpy already started
        from threadpoolctl import threadpool_limits
        threadpool_limits(n)
    except ImportError:
        pass
