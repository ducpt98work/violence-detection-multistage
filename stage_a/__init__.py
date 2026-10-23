"""Stage A of "Real-Time Violence Detection via Kinematic Multi-Stage Gating and ROI-Based
Swin-Temporal Transformer": kinematic multi-stage gating (Sec. III-A of the paper)."""

from .config import PROJECT_ROOT, load_config, resolve_path  # noqa: F401
from .descriptor import FEATURE_NAMES, WindowEngine, WindowSample  # noqa: F401
from .gate import StageAGate, physics_pass  # noqa: F401
