"""Transcoding of video files that OpenCV cannot decode reliably (see video.UNRELIABLE_FOURCC).

The external ffmpeg decodes DV correctly; the result is H.264 at CRF 12 (visually lossless),
same frame rate and frame count, no audio, no deinterlacing (frames as a plain decoder gives them).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from .config import resolve_path


def find_ffmpeg(configured: Optional[str] = None) -> Optional[str]:
    """Configured path, then ffmpeg on PATH, then the binary shipped with imageio-ffmpeg."""
    if configured:
        p = resolve_path(configured)
        if p.exists():
            return str(p)
        raise FileNotFoundError(f"tools.ffmpeg points to a missing file: {p}")
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def transcode_h264(src: str | Path, dst: str | Path, ffmpeg: str, crf: int = 12) -> Path:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".tmp" + dst.suffix)
    cmd = [ffmpeg, "-v", "error", "-y", "-i", str(src), "-map", "0:v:0", "-c:v", "libx264",
           "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p", "-an", str(tmp)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    os.replace(tmp, dst)
    return dst
