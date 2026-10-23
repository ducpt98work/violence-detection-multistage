"""Video decoding, resizing to W_target and nearest-frame resampling onto the analysis grid.

Sec. III-A-1: detection and tracking run on every decoded frame, while the kinematic
quantities are read on a uniform 0.1 s grid, taking at each analysis instant the frame
nearest to it in time. Analysis index k therefore always means time k / analysis_fps.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class FramePacket:
    index: int                      # source frame index
    time: float                     # seconds from the start of the stream
    frame: np.ndarray               # decoded BGR frame at source resolution
    resized: np.ndarray             # BGR frame resized to width W_target
    analysis_index: Optional[int]   # k if this frame is the one nearest to t = k / analysis_fps


class AnalysisSampler:
    """Decides online which source frame is nearest to each analysis instant.

    A frame at time t is the nearest one to grid instant g iff |t - g| <= dt_src / 2; when
    frames arrive in order, the first frame with t + dt_src / 2 >= g is that frame.
    """

    def __init__(self, source_fps: float, analysis_fps: float = 10.0):
        self.half_src = 0.5 / source_fps
        self.dt = 1.0 / analysis_fps
        self.next_k = 0
        if source_fps < analysis_fps:
            log.warning("source fps %.2f is below the analysis rate %.1f; analysis indices will skip",
                        source_fps, analysis_fps)

    def __call__(self, t: float) -> Optional[int]:
        if self.next_k * self.dt > t + self.half_src + 1e-9:
            return None
        k = self.next_k
        while self.next_k * self.dt <= t + self.half_src + 1e-9:
            self.next_k += 1
        return k


# DV video (Movies V/N-V fight clips): OpenCV's FFmpeg backend fails the interlaced yuv411p -> BGR
# conversion ("swscaler: Cannot convert interlaced to progressive frames") and returns black or
# striped garbage frames, depending on which libraries are loaded. ``prepare_data.py scan``
# transcodes such files to H.264 with an external ffmpeg; see stage_a.transcode.
UNRELIABLE_FOURCC = ("dvsd", "dv25", "dv50", "dvh1", "dvhd", "dvsl", "cdvc", "dvcp", "dvpp")


def fourcc_of(cap: cv2.VideoCapture) -> str:
    code = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
    return code.to_bytes(4, "little").decode("latin-1").strip("\x00").lower()


def resize_to_width(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w == width:
        return frame
    height = int(round(h * width / w))
    interp = cv2.INTER_AREA if width < w else cv2.INTER_LINEAR
    return cv2.resize(frame, (width, height), interpolation=interp)


class VideoReader:
    """Iterates over a file, an RTSP/HTTP URL or a webcam index."""

    def __init__(self, source, resize_width: int = 640, analysis_fps: float = 10.0,
                 max_seconds: Optional[float] = None, fps_override: Optional[float] = None):
        self.source = source
        src = int(source) if isinstance(source, int) or str(source).isdigit() else str(source)
        self.is_live = isinstance(src, int) or "://" in str(src)
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise IOError(f"cannot open video source {source!r}")
        self.fourcc = fourcc_of(self.cap)
        if self.fourcc in UNRELIABLE_FOURCC:
            log.warning("%s: fourcc %r is decoded unreliably by OpenCV; transcode it to H.264 first "
                        "(prepare_data.py scan does this for the benchmark)", source, self.fourcc)
        fps = fps_override or self.cap.get(cv2.CAP_PROP_FPS)
        if not fps or not (1.0 <= fps <= 240.0):
            log.warning("invalid fps %r reported for %s; assuming 30", fps, source)
            fps = 30.0
        self.fps = float(fps)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.resize_width = resize_width
        self.analysis_fps = analysis_fps
        self.max_seconds = max_seconds

    @property
    def scale(self) -> float:
        """Factor mapping source-resolution pixels to analysis-frame pixels."""
        return self.resize_width / float(self.width) if self.width else 1.0

    def __iter__(self) -> Iterator[FramePacket]:
        sampler = AnalysisSampler(self.fps, self.analysis_fps)
        index = 0
        while True:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                break
            if self.width == 0:  # some streams report 0 until the first frame
                self.height, self.width = frame.shape[:2]
            t = index / self.fps
            if self.max_seconds is not None and t >= self.max_seconds:
                break
            yield FramePacket(index, t, frame, resize_to_width(frame, self.resize_width), sampler(t))
            index += 1

    def release(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


def read_frames(path, indices, crop_from: str = "original", width: int = 640) -> list:
    """Source frames at the given indices (in order, repeats allowed), at source resolution or
    resized to the analysis width. Indices past the end of the stream give the last frame."""
    wanted, stop = set(indices), max(indices)
    got, last = {}, None
    cap = cv2.VideoCapture(str(path))
    try:
        for i in range(stop + 1):
            ok, frame = cap.read()
            if not ok:
                break
            last = frame
            if i in wanted:
                got[i] = frame
    finally:
        cap.release()
    if last is None:
        raise IOError(f"cannot decode {path}")
    frames = [got.get(i, last) for i in indices]
    return frames if crop_from == "original" else [resize_to_width(f, width) for f in frames]
