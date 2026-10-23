"""Locations of the intermediate artefacts under ``paths.work_dir`` and ``paths.models_dir``."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import resolve_path


def work_dir(cfg) -> Path:
    return resolve_path(cfg.paths.work_dir)


def models_dir(cfg) -> Path:
    p = resolve_path(cfg.paths.models_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def manifest_path(cfg, name: str) -> Path:
    p = work_dir(cfg) / "manifests" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def cache_path(cfg, clip_id: str) -> Path:
    return work_dir(cfg) / "cache" / f"{clip_id}.npz"


def windows_path(cfg) -> Path:
    p = work_dir(cfg) / "windows" / "windows.csv.gz"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_benchmark(cfg, kept_only: bool = True) -> pd.DataFrame:
    df = pd.read_csv(manifest_path(cfg, "benchmark.csv"), keep_default_na=False,
                     dtype={"split": str, "stageb_split": str, "svm_split": str, "excluded": str})
    return df[df.excluded == ""].copy() if kept_only else df


def load_windows(cfg, clip_ids=None) -> pd.DataFrame:
    df = pd.read_csv(windows_path(cfg), dtype={"members": str})
    return df[df.clip_id.isin(set(clip_ids))].copy() if clip_ids is not None else df
