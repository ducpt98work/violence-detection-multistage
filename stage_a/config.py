"""Configuration loading: YAML -> attribute-accessible dict, with CLI overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Iterable, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "stage_a.yaml"


class Cfg(dict):
    """A dict whose keys are also readable as attributes (recursively)."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value

    def to_dict(self) -> dict:
        return _unwrap(self)


def _wrap(obj):
    if isinstance(obj, dict):
        return Cfg({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def _unwrap(obj):
    if isinstance(obj, dict):
        return {k: _unwrap(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_unwrap(v) for v in obj]
    return obj


def apply_overrides(data: dict, overrides: Iterable[str]) -> dict:
    """Apply ``a.b.c=value`` overrides; the value is parsed as YAML."""
    data = copy.deepcopy(data)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key.path=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = data
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return data


def load_config(path: Optional[str | Path] = None, overrides: Iterable[str] = ()) -> Cfg:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return _wrap(apply_overrides(data, overrides))


def config_from_dict(data: dict, overrides: Iterable[str] = ()) -> Cfg:
    """Rebuild a config from a plain dict (e.g. the one stored in a trained gate bundle)."""
    return _wrap(apply_overrides(data, overrides))


def resolve_path(path: str | Path) -> Path:
    """Paths in the config are relative to the project root."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def display_path(path: str | Path) -> str:
    """A path as written to reports and logs on disk: relative to the project root, or just the
    file name for a file outside it, so no output records the directories of the machine."""
    p = Path(str(path))
    try:
        return p.resolve().relative_to(PROJECT_ROOT).as_posix()
    except (ValueError, OSError):
        return p.name


def display_source(source) -> str:
    """A video source as written to reports: the file name, ``webcam<n>`` or ``<scheme> stream``,
    without directories, user names, passwords or camera addresses."""
    s = str(source)
    if s.isdigit():
        return f"webcam{s}"
    if "://" in s:
        return s.split("://", 1)[0].lower() + " stream"
    return Path(s).name
