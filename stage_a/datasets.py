"""Source adapters: find every clip of each dataset with its label and a split-grouping key.

The grouping key keeps clips cut from one released video (RWF-2000 segments ``<id>_<n>``) or one
recorded performance (AVD cam1/N and cam2/N) on the same side of every split.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List

from .config import PROJECT_ROOT, resolve_path

VIDEO_EXT = {".avi", ".mp4", ".mpg", ".mpeg", ".mov", ".mkv", ".wmv", ".m4v"}


@dataclass
class ClipRecord:
    clip_id: str
    source: str
    label: int           # 1 violent, 0 non-violent
    path: str            # relative to the project root when possible
    group: str           # split-grouping key

    def to_dict(self):
        return asdict(self)


def _videos(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXT)


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p.resolve())


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")


def scan_rwf2000(root: Path, cfg) -> List[ClipRecord]:
    """RWF-2000: <root>/**/{train,val}/{Fight,NonFight}/*.avi; ``<youtube id>_<n>`` share a video."""
    out = []
    for p in _videos(root):
        parts = [x.lower() for x in p.parts]
        if "fight" in parts:
            label = 1
        elif "nonfight" in parts:
            label = 0
        else:
            continue
        video = re.sub(r"_\d+$", "", p.stem)
        out.append(ClipRecord(f"rwf2000/{'V' if label else 'NV'}_{_slug(p.stem)}", "rwf2000", label,
                              _rel(p), f"rwf2000/{video}"))
    return out


def scan_rlvs(root: Path, cfg) -> List[ClipRecord]:
    """RLVS: <root>/**/{Violence,NonViolence}/*. The Kaggle zip ships the set twice; take one copy."""
    out = []
    for p in _videos(root):
        if "real life violence situations" in str(p).lower():
            continue
        parent = p.parent.name.lower()
        if parent == "violence":
            label = 1
        elif parent == "nonviolence":
            label = 0
        else:
            continue
        out.append(ClipRecord(f"rlvs/{_slug(p.stem)}", "rlvs", label, _rel(p), f"rlvs/{_slug(p.stem)}"))
    return out


def scan_hockey(root: Path, cfg) -> List[ClipRecord]:
    """Hockey Fight: fi*.avi (fight) / no*.avi. Game identity is not recoverable from file names."""
    out = []
    for p in _videos(root):
        name = p.name.lower()
        if name.startswith("fi"):
            label = 1
        elif name.startswith("no"):
            label = 0
        else:
            continue
        out.append(ClipRecord(f"hockey/{_slug(p.stem)}", "hockey", label, _rel(p), f"hockey/{_slug(p.stem)}"))
    return out


def scan_movies(root: Path, cfg) -> List[ClipRecord]:
    """Movies V/N-V: <root>/**/fights/*, <root>/**/noFights/*."""
    out = []
    for p in _videos(root):
        parent = p.parent.name.lower()
        if parent == "fights":
            label = 1
        elif parent == "nofights":
            label = 0
        else:
            continue
        cid = f"movies/{'V' if label else 'NV'}_{_slug(p.stem)}"
        out.append(ClipRecord(cid, "movies", label, _rel(p), cid))
    return out


def scan_avd(root: Path, cfg) -> List[ClipRecord]:
    """AVD (AIRTLab): violent|non-violent / cam1|cam2 / N.mp4; cam1/N and cam2/N are one performance."""
    out = []
    group_cams = cfg.benchmark.get("group_avd_cameras", True)
    for p in _videos(root):
        cam = p.parent.name.lower()
        cls = p.parent.parent.name.lower()
        if cls == "violent":
            label = 1
        elif cls == "non-violent":
            label = 0
        else:
            continue
        tag = "V" if label else "NV"
        group = f"avd/{tag}_{p.stem}" if group_cams else f"avd/{tag}_{cam}_{p.stem}"
        out.append(ClipRecord(f"avd/{tag}_{cam}_{_slug(p.stem)}", "avd", label, _rel(p), group))
    return out


def scan_ptit(root: Path, cfg) -> List[ClipRecord]:
    """PTIT (private): <root>/{violent,non_violent}/**/* ; the sub-folder is the camera position."""
    out = []
    for p in _videos(root):
        rel = p.relative_to(root).parts
        if rel[0].lower() in ("violent", "violence"):
            label = 1
        elif rel[0].lower() in ("non_violent", "non-violent", "nonviolence"):
            label = 0
        else:
            continue
        cid = f"ptit/{'V' if label else 'NV'}_{_slug('_'.join(rel[1:]))}"
        out.append(ClipRecord(cid, "ptit", label, _rel(p), cid))
    return out


SCANNERS: Dict[str, Callable] = {
    "rwf2000": scan_rwf2000, "rlvs": scan_rlvs, "hockey": scan_hockey,
    "movies": scan_movies, "avd": scan_avd, "ptit": scan_ptit,
}


def scan_sources(cfg) -> Dict[str, List[ClipRecord]]:
    found = {}
    for name, src in cfg.sources.items():
        if not src.get("enabled", True):
            continue
        root = resolve_path(src.root)
        found[name] = SCANNERS[name](root, cfg) if root.exists() else []
    return found


def file_md5(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(resolve_path(path), "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()
