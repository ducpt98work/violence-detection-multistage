#!/usr/bin/env python
"""Train the Stage A gatekeeper (Sec. III-A-5).

  D_SVM   = windows of the selected clips (stride 5, physics filter disabled), label = clip label
  model   = StandardScaler + SVC(kernel="rbf"), C and gamma by 5-fold grouped CV (group = clip)
            on the training clips
  operating point: the signed-margin threshold is moved toward the violent class until the
            validation clips reach the target window recall (77.3% in the paper)

Writes models/stage_a_gate.joblib (model + threshold + the full config it was trained with)
and a JSON / Markdown report next to it.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.metrics import precision_recall_fscore_support  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from stage_a.config import load_config  # noqa: E402
from stage_a.descriptor import FEATURE_NAMES  # noqa: E402
from stage_a.gate import StageAGate, physics_pass  # noqa: E402
from stage_a.paths import load_benchmark, load_windows, models_dir  # noqa: E402

log = logging.getLogger("train_stage_a")


def group_kfold(groups: np.ndarray, n_folds: int) -> list:
    """GroupKFold's assignment (largest group first, into the lightest fold) with ties broken by
    group id. sklearn's GroupKFold breaks ties with an unstable sort, so its folds, and the
    selected (C, gamma), change with the numpy build."""
    ids, inverse, sizes = np.unique(groups, return_inverse=True, return_counts=True)
    load = np.zeros(n_folds, np.int64)
    fold_of = np.empty(len(ids), np.int64)
    for g in np.argsort(-sizes, kind="stable"):
        k = int(np.argmin(load))
        fold_of[g] = k
        load[k] += sizes[g]
    fold = fold_of[inverse]
    return [(np.flatnonzero(fold != k), np.flatnonzero(fold == k)) for k in range(n_folds)]


def _gamma(v):
    if isinstance(v, str) and v not in ("scale", "auto"):
        return float(v)
    return v


def per_class(y, pred) -> dict:
    p, r, f, n = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    w = n / n.sum()
    return {"non_violent": {"precision": p[0], "recall": r[0], "f1": f[0], "support": int(n[0])},
            "violent": {"precision": p[1], "recall": r[1], "f1": f[1], "support": int(n[1])},
            "weighted": {"precision": float((p * w).sum()), "recall": float((r * w).sum()),
                         "f1": float((f * w).sum())},
            "accuracy": float((y == pred).mean())}


def pick_threshold(scores, y, phys, target, mode) -> tuple[float, float]:
    """Highest threshold whose violent-window recall reaches ``target``. Returns (thr, recall)."""
    pos = y == 1
    eligible = pos & phys if mode == "gate" else pos
    cand = np.sort(scores[eligible])[::-1]
    need = int(math.ceil(target * pos.sum()))
    if need == 0:
        return float(np.inf), 0.0
    if need > len(cand):
        log.warning("target recall %.3f unreachable in %s mode (max %.3f); forwarding everything eligible",
                    target, mode, len(cand) / pos.sum())
        thr = float(cand[-1]) if len(cand) else float("-inf")
    else:
        thr = float(cand[need - 1])
    fwd = (phys if mode == "gate" else np.ones_like(phys)) & (scores >= thr)
    return thr, float(fwd[pos].mean())


def gate_metrics(scores, y, phys, thr) -> dict:
    fwd = phys & (scores >= thr)
    pos = y == 1
    return {"threshold": thr, "forwarded_fraction": float(fwd.mean()),
            "rejected_by_physics": float((~phys).mean()), "rejected_by_svm": float((phys & ~fwd).mean()),
            "violent_recall": float(fwd[pos].mean()) if pos.any() else float("nan"),
            "violent_precision": float(pos[fwd].mean()) if fwd.any() else float("nan")}


def clip_coverage(df, fwd) -> float:
    v = df.assign(fwd=fwd)[df.label == 1].groupby("clip_id").fwd.any()
    return float(v.mean()) if len(v) else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--out", default=None, help="model bundle path (default <models_dir>/stage_a_gate.joblib)")
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config, args.set)
    s = cfg.svm
    out = Path(args.out) if args.out else models_dir(cfg) / "stage_a_gate.joblib"

    bench = load_benchmark(cfg)
    clips = bench[bench.svm_split.isin(["train", "val"])][["clip_id", "source", "label", "svm_split"]]
    if clips.empty:
        raise SystemExit("no D_SVM clips: run prepare_data.py split first")
    win = load_windows(cfg, clips.clip_id).merge(clips, on="clip_id")
    X = win[list(FEATURE_NAMES)].to_numpy(np.float64)
    y = win.label.to_numpy(int)
    phys = physics_pass(X, cfg.gate.physics.a_max, cfg.gate.physics.mu_heat)
    tr = (win.svm_split == "train").to_numpy()
    va = ~tr
    fit_mask = tr & (phys if s.train_on == "physics" else True)
    log.info("D_SVM: %d windows from %d clips (train %d windows / %d clips, val %d / %d); fitting on %d",
             len(win), clips.clip_id.nunique(), tr.sum(), win[tr].clip_id.nunique(), va.sum(),
             win[va].clip_id.nunique(), fit_mask.sum())

    pipe = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf"))])
    grid = {"svm__C": [float(c) for c in s.C_grid], "svm__gamma": [_gamma(g) for g in s.gamma_grid]}
    groups = win.clip_id.to_numpy()
    folds = group_kfold(groups[fit_mask], s.cv_folds)
    search = GridSearchCV(pipe, grid, cv=folds, scoring="accuracy", n_jobs=args.n_jobs,
                          refit=True, return_train_score=False)
    search.fit(X[fit_mask], y[fit_mask])
    best = search.best_estimator_
    log.info("grouped %d-fold CV: best %s, accuracy %.4f", s.cv_folds, search.best_params_, search.best_score_)

    # the optimism a vector-level split would add (adjacent windows share half their frames), and
    # whether such a split would have picked another (C, gamma) cell (the paper: same cell)
    vec_cv_kw = dict(cv=StratifiedKFold(s.cv_folds, shuffle=True, random_state=cfg.benchmark.seed), n_jobs=args.n_jobs)
    vec_cv = cross_val_score(clone(best), X[fit_mask], y[fit_mask], **vec_cv_kw)
    vec_search = GridSearchCV(pipe, grid, scoring="accuracy", refit=False, **vec_cv_kw)
    vec_search.fit(X[fit_mask], y[fit_mask])
    log.info("vector-level %d-fold CV: at the grouped-CV cell %.4f; its own best %s at %.4f",
             s.cv_folds, vec_cv.mean(), vec_search.best_params_, vec_search.best_score_)

    # operating point on the validation clips
    scores_va = best.decision_function(X[va])
    y_va, phys_va = y[va], phys[va]
    thr, rec = pick_threshold(scores_va, y_va, phys_va, s.target_recall, s.calibrate_on)
    log.info("operating threshold %.4f (%s recall %.4f on validation, target %.3f)",
             thr, s.calibrate_on, rec, s.target_recall)

    svm_default = per_class(y_va, (scores_va >= 0).astype(int))
    svm_operating = per_class(y_va, (scores_va >= thr).astype(int))
    gate_default = gate_metrics(scores_va, y_va, phys_va, 0.0)
    gate_operating = gate_metrics(scores_va, y_va, phys_va, thr)
    win_va = win[va].reset_index(drop=True)
    gate_operating["clip_coverage"] = clip_coverage(win_va, phys_va & (scores_va >= thr))
    gate_default["clip_coverage"] = clip_coverage(win_va, phys_va & (scores_va >= 0))

    # recall / precision trade-off of the gate on validation, for choosing another point by hand
    curve = []
    for q in np.linspace(0, 1, 41):
        t = float(np.quantile(scores_va, q))
        m = gate_metrics(scores_va, y_va, phys_va, t)
        curve.append({"threshold": t, **{k: m[k] for k in ("violent_recall", "violent_precision", "forwarded_fraction")}})
    pd.DataFrame(curve).to_csv(out.with_name(out.stem + "_val_curve.csv"), index=False)

    cv_table = pd.DataFrame(search.cv_results_)[["param_svm__C", "param_svm__gamma", "mean_test_score", "std_test_score"]]
    meta = {
        "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "best_params": {k: (v if isinstance(v, str) else float(v)) for k, v in search.best_params_.items()},
        "grouped_cv_accuracy": float(search.best_score_),
        "vector_cv_accuracy": float(vec_cv.mean()),
        "vector_cv_best_params": {k: (v if isinstance(v, str) else float(v)) for k, v in vec_search.best_params_.items()},
        "vector_cv_best_accuracy": float(vec_search.best_score_),
        "n_windows": {"train": int(tr.sum()), "val": int(va.sum()), "fit": int(fit_mask.sum())},
        "n_clips": {"train": int(win[tr].clip_id.nunique()), "val": int(win[va].clip_id.nunique())},
        "windows_by_source": {f"{src}/{'V' if lab else 'NV'}/{split}": int(n) for (src, lab, split), n in
                              win.groupby(["source", "label", "svm_split"]).size().items()},
        "train_on": s.train_on, "calibrate_on": s.calibrate_on, "target_recall": s.target_recall,
        "n_support_vectors": int(best.named_steps["svm"].n_support_.sum()),
    }
    gate = StageAGate(best, thr, cfg.gate.physics.a_max, cfg.gate.physics.mu_heat, cfg.to_dict(), meta)
    gate.save(out)

    report = {"meta": meta,
              "validation": {"svm_at_default_threshold": svm_default, "svm_at_operating_threshold": svm_operating,
                             "gate_at_default_threshold": gate_default, "gate_at_operating_threshold": gate_operating},
              "cv_results": cv_table.astype({"param_svm__C": float, "param_svm__gamma": str}).to_dict("records")}
    with open(out.with_name(out.stem + "_report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    md = [f"# Stage A gate training report ({meta['trained_at']})", "",
          f"- D_SVM windows: train {meta['n_windows']['train']} ({meta['n_clips']['train']} clips), "
          f"val {meta['n_windows']['val']} ({meta['n_clips']['val']} clips); fitted on {meta['n_windows']['fit']} ({s.train_on})",
          f"- Grouped {s.cv_folds}-fold CV: best C={meta['best_params']['svm__C']}, gamma={meta['best_params']['svm__gamma']}, "
          f"accuracy {meta['grouped_cv_accuracy']:.4f} (vector-level CV at this cell: {meta['vector_cv_accuracy']:.4f}; "
          f"vector-level CV's own best: C={meta['vector_cv_best_params']['svm__C']}, "
          f"gamma={meta['vector_cv_best_params']['svm__gamma']}, {meta['vector_cv_best_accuracy']:.4f})",
          f"- Operating threshold on the signed margin: {thr:.4f} ({s.calibrate_on} recall {rec:.4f}, target {s.target_recall})",
          f"- Support vectors: {meta['n_support_vectors']}", "",
          "## SVM per class on the validation windows (operating threshold)", "",
          "| Class | Precision | Recall | F1 |", "|---|---|---|---|"]
    for cls in ("non_violent", "violent", "weighted"):
        m = svm_operating[cls]
        md.append(f"| {cls} | {100 * m['precision']:.1f} | {100 * m['recall']:.1f} | {100 * m['f1']:.1f} |")
    md += ["", "## Whole gate (physics AND SVM) on the validation windows", "",
           "| Threshold | Rej. physics | Rej. SVM | Forwarded | Violent recall | Violent precision | Clip coverage |",
           "|---|---|---|---|---|---|---|"]
    for name, m in (("default (0)", gate_default), ("operating", gate_operating)):
        md.append(f"| {name} {m['threshold']:.3f} | {100 * m['rejected_by_physics']:.1f} | {100 * m['rejected_by_svm']:.1f} | "
                  f"{100 * m['forwarded_fraction']:.1f} | {100 * m['violent_recall']:.1f} | "
                  f"{100 * m['violent_precision']:.1f} | {100 * m['clip_coverage']:.1f} |")
    out.with_name(out.stem + "_report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    log.info("saved %s", out)


if __name__ == "__main__":
    main()
