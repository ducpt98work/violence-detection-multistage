from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "stage_b.yaml"


def load_config(path=None, overrides=()):
    """Load a YAML config and apply "section.key=value" overrides."""
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for item in overrides:
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    return cfg


def resolve_path(path):
    """Paths in the config are relative to the project root."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path
