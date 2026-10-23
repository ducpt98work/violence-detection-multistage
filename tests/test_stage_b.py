"""Unit checks of Stage B: model configuration, trigger choice, alert rule, metrics.

Run:  python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from extract_stage_b_clips import flagged_trigger  # noqa: E402
from stage_a.descriptor import WindowSample  # noqa: E402
from stage_a.gate import GateDecision  # noqa: E402
from stage_b.inference import AlertManager, GroupRegistry  # noqa: E402
from stage_b.metrics import binary_metrics, video_decisions  # noqa: E402
from stage_b.model import SwinTemporal, count_parameters  # noqa: E402


def test_model_matches_the_paper_configuration():
    m = SwinTemporal(pretrained=False)
    temporal = count_parameters(m) - count_parameters(m.backbone)
    assert abs(temporal / 1e6 - 14.5) < 0.05
    m.freeze_backbone()
    assert count_parameters(m, trainable_only=True) == temporal
    m.train()
    assert not m.backbone.training and m.temporal.training
    with torch.no_grad():
        assert m(torch.randn(2, 16, 3, 224, 224)).shape == (2,)


def test_variants_build():
    for kw in (dict(temporal_pool="attn"), dict(temporal_pool="cls"), dict(temporal_pool="max"),
               dict(pos_encoding="sinusoidal"), dict(dim=384, ffn_dim=1536, head_hidden=192), dict(depth=4)):
        m = SwinTemporal(pretrained=False, **kw)
        with torch.no_grad():
            assert m.temporal_forward(torch.randn(2, 16, 768)).shape == (2,)


class _Gate:
    """Margin grows with A_max; forwards above A_max = 15."""

    def score(self, X):
        return X[:, 3] - 20.0

    def decide(self, X):
        fwd = X[:, 3] > 15
        return GateDecision(fwd, self.score(X), fwd)


def _window(aidx, a_max):
    f = np.zeros(7)
    f[3] = a_max
    return WindowSample(aidx, aidx / 10, (1, 2), (0, 0, 10, 10), f, 10, 2)


def test_trigger_is_the_most_strongly_flagged_window():
    w, margin, how = flagged_trigger([_window(10, 12), _window(15, 30), _window(20, 25)], _Gate())
    assert (w.aidx, how, margin) == (15, "forwarded", 10.0)
    w, _, how = flagged_trigger([_window(10, 5), _window(15, 12)], _Gate())
    assert (w.aidx, how) == (15, "scored")


def test_alert_needs_three_positives_and_respects_cooldown():
    am = AlertManager()
    assert not am.update("g", 0.9, now=0.0)
    assert not am.update("g", 0.9, now=0.5)
    assert am.update("g", 0.9, now=1.0)          # third consecutive positive
    assert not am.update("g", 0.9, now=1.5)      # absorbed by the open alert
    assert not am.update("g", 0.1, now=40.0)     # cooldown over, history ends negative
    assert am.update("h", 0.9, now=0.0, windows=3)  # one clip merged from three windows


def test_group_keeps_its_id_as_members_change():
    reg = GroupRegistry()
    a = reg.key([1, 2])
    assert reg.key([2, 3]) == a          # shares track 2
    assert reg.key([3, 4, 5]) == a       # shares track 3 with the updated group
    assert reg.key([9]) != a             # no shared track: a new group


def test_soft_band_discount():
    am = AlertManager(persistence=1)
    assert not am.update("g", 0.6, now=0.0, soft=True)   # needs p > 0.5 / 0.75
    assert am.update("g", 0.7, now=0.0, soft=True)


def test_chance_corrected_measures_on_the_paper_confusion_matrix():
    y = np.r_[np.ones(571), np.zeros(572)]
    p = np.r_[np.ones(531), np.zeros(40), np.ones(60), np.zeros(512)]   # TP 531, FN 40, FP 60, TN 512
    m = binary_metrics(y, p)
    assert round(m["mcc"], 3) == 0.826 and round(m["kappa"], 3) == 0.825
    assert round(100 * m["specificity"], 1) == 89.5 and round(100 * m["f1"], 1) == 91.4


def test_metrics_and_or_rule():
    m = binary_metrics([1, 1, 0, 0], [0.9, 0.4, 0.6, 0.1])
    assert m["tp"] == 1 and m["fp"] == 1 and m["f1"] == 0.5 and m["fpr"] == 0.5
    assert video_decisions(["a", "a", "b"], [0.2, 0.8, 0.3]) == {"a": 1, "b": 0}
