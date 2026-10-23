"""Unit checks of the Stage A building blocks against the definitions (and worked examples) of the paper.

Run:  python -m pytest tests -q      (or simply: python tests/test_stage_a.py)
"""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stage_a.clips import count_invocations  # noqa: E402
from stage_a.descriptor import WindowEngine, circular_std_deg, pad_box  # noqa: E402
from stage_a.energy import MotionEnergy, region_stats  # noqa: E402
from stage_a.gate import physics_pass  # noqa: E402
from stage_a.grouping import adaptive_groups  # noqa: E402
from stage_a.video import AnalysisSampler  # noqa: E402


def _box(cx, cy, h, w=None):
    w = w or h * 0.4
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], np.float32)


def test_sampler_nearest_frame_mean_interval():
    for fps in (24.0, 25.0, 29.97, 30.0, 60.0):
        s = AnalysisSampler(fps, 10.0)
        picked = [(k, i / fps) for i in range(int(fps * 20)) if (k := s(i / fps)) is not None]
        ks = [k for k, _ in picked]
        assert ks == list(range(len(ks))), "analysis indices must be consecutive"
        for k, t in picked:  # nearest frame: within half a source frame of k / 10
            assert abs(t - k / 10.0) <= 0.5 / fps + 1e-9
        assert abs(np.diff([t for _, t in picked]).mean() - 0.1) < 0.002


def test_energy_update_rule():
    me = MotionEnergy(15, 1.0, 0.1)
    still = np.zeros((4, 4), np.uint8)
    moved = still.copy()
    moved[0, 0] = 200
    me.update(still)                                # first frame: E_0 = 0
    me.update(moved)                                # pixel (0,0) moves: +alpha
    me.update(still)                                # moves back: +alpha -> 2.0
    assert math.isclose(me.energy()[0, 0], 2.0)
    for _ in range(5):
        me.update(still)                            # static: -0.1 alpha per frame
    assert math.isclose(me.energy()[0, 0], 1.5, abs_tol=1e-6)
    for _ in range(30):
        me.update(still)
    assert me.energy()[0, 0] == 0.0                 # floored at zero
    assert me.energy()[1, 1] == 0.0
    mu, sd = region_stats(np.full((2, 2), 20, np.int32), 0.1, (0, 0, 2, 2))
    assert math.isclose(mu, 2.0) and sd == 0.0


def test_grouping_paper_examples():
    # Fig. grouping (b): heights 394 and 452 px, centres 793.77 px apart, T = 634.5 -> two singletons
    b = np.stack([_box(100, 500, 394), _box(100 + 793.77, 500, 452)])
    assert adaptive_groups(b, 1.5) == [[0], [1]]
    # distant pair: 49.75 px apart, mean height 67.96 -> T = 101.94 -> grouped
    b = np.stack([_box(300, 100, 67.96), _box(300 + 49.75, 100, 67.96)])
    assert adaptive_groups(b, 1.5) == [[0, 1]]
    # transitivity: A-B and B-C linked, A-C too far
    b = np.stack([_box(0, 0, 100), _box(140, 0, 100), _box(280, 0, 100)])
    assert adaptive_groups(b, 1.5) == [[0, 1, 2]]


def test_circular_std():
    assert circular_std_deg([0.3] * 10) < 1e-3
    near = circular_std_deg(np.radians([-179.0, 179.0]))
    assert near < 2.0, near                         # wrap-around: 2 degrees apart, not 358
    assert circular_std_deg(np.radians([0.0, 180.0])) > 300  # opposite directions: unbounded growth


def test_physics_gate():
    X = np.zeros((4, 7))
    X[1, 3] = 15.1        # A_max
    X[2, 5] = 2.1         # mu_heat
    X[3, 3], X[3, 5] = 15.0, 2.0  # both exactly at the threshold: strict inequalities reject
    assert physics_pass(X).tolist() == [False, True, True, False]


def _obs(tid, box, kbox=None):
    return SimpleNamespace(tid=tid, box=box, matched=True, conf=0.9, tsu=0, kbox=box if kbox is None else kbox)


def test_box_source_selection():
    det, kal = _box(100, 100, 100), _box(140, 100, 100)
    for source, expected in (("kalman", kal), ("detection", det)):
        eng = WindowEngine(box_source=source)
        eng.observe(0, [_obs(1, det, kal)])
        assert np.allclose(eng.hist[1][-1].box, expected)


def test_window_engine_kinematics_and_admission():
    eng = WindowEngine(window=10, stride=5, grouping_k=1.5, min_members=2, min_history=10, roi_pad=0.15,
                       analysis_fps=10.0)
    units = np.zeros((360, 640), np.int32)
    out = []
    for k in range(16):
        # person 1 walks right at 10 px per analysis frame with height 100 -> v = 0.1 * 10 = 1 BL/s
        # person 2 stands 80 px away
        groups = eng.observe(k, [_obs(1, _box(200 + 10 * k, 200, 100)), _obs(2, _box(300 + 10 * k * 0, 200, 100))])
        ws = eng.windows(k, units, 0.1, (360, 640))
        out.append((k, groups, ws))
    # windows only on evaluation frames, and only once both members have 10 frames of history
    assert all(not ws for k, _, ws in out if k % 5)
    assert not out[5][2]
    ws = out[10][2]
    assert len(ws) == 1 and ws[0].members == (1, 2)
    v_max, sigma_v, mu_a, a_max, s_int, mu_heat, sigma_heat = ws[0].features
    assert math.isclose(v_max, 1.0, rel_tol=1e-5) and sigma_v < 1e-6
    assert abs(mu_a) < 1e-6 and abs(a_max) < 1e-6
    assert s_int < 1e-3            # the stationary person has no direction; the walker's is constant
    assert mu_heat == 0.0 and sigma_heat == 0.0
    x1, y1, x2, y2 = ws[0].roi     # union of boxes over the window, enlarged by 15%
    assert x1 < 200 + 10 * 1 - 20 and x2 > 300 + 20


def test_pad_box_square_and_clip():
    assert pad_box((10, 10, 30, 50), 0.0, 640, 360, square=True)[2:] != (0, 0)
    x1, y1, x2, y2 = pad_box((600, 300, 640, 360), 0.5, 640, 360, square=True)
    assert 0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 360 and abs((x2 - x1) - (y2 - y1)) <= 1


def test_dv_clips_are_indexed_as_h264_copies():
    """Regression: OpenCV decodes the interlaced DV fight clips of Movies V/N-V as black or striped
    frames; scan must point them to H.264 copies that decode to real frames, frame for frame."""
    import pandas as pd

    from stage_a.config import PROJECT_ROOT
    from stage_a.video import UNRELIABLE_FOURCC, VideoReader

    index = PROJECT_ROOT / "work/manifests/index.csv"
    if not index.exists():
        print("skip: run prepare_data.py scan first")
        return
    df = pd.read_csv(index, keep_default_na=False)
    assert not df.fourcc.isin(UNRELIABLE_FOURCC).any(), "an indexed clip still uses a DV codec"
    dv = df[df.path != df.original_path]
    assert len(dv) and dv.path.str.endswith(".mp4").all()
    row = dv.iloc[0]
    with VideoReader(PROJECT_ROOT / row.path, 640, 10.0) as reader:
        frames = [p.frame for p in reader]
    assert abs(len(frames) - int(row.frames_meta)) <= 1
    assert all(f.max() > 0 for f in frames)
    # garbage decodes are vertical stripes: real frames have far smoother rows than columns
    f = frames[len(frames) // 2].astype(np.float32).mean(axis=2)
    assert np.abs(np.diff(f, axis=1)).mean() < 3 * np.abs(np.diff(f, axis=0)).mean() + 5


def test_clip_merging():
    # one group: triggers 0.5 s apart fall inside one 16-frame span -> 1 invocation; 2 s later -> a new one
    assert count_invocations([(10, [1, 2]), (15, [1, 2]), (20, [2, 1])]) == 2
    assert count_invocations([(10, [1, 2]), (15, [1, 2])]) == 1
    # a different group at the same time needs its own clip
    assert count_invocations([(10, [1, 2]), (10, [5, 6])]) == 2


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
