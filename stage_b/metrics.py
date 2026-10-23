import numpy as np
from sklearn.metrics import roc_auc_score


def binary_metrics(labels, probs, threshold=0.5):
    y = np.asarray(labels).astype(int)
    p = np.asarray(probs)
    pred = (p > threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    n = len(y)
    acc = (tp + tn) / n
    spec = tn / (tn + fp) if tn + fp else 0.0
    # chance-corrected measures and calibration (Sec. IV-E)
    mcc_den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    p_chance = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / n**2
    out = {
        "n": n,
        "acc": acc,
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "specificity": spec,
        "balanced_acc": (rec + spec) / 2,
        "mcc": (tp * tn - fp * fn) / mcc_den if mcc_den else 0.0,
        "kappa": (acc - p_chance) / (1 - p_chance) if p_chance < 1 else 0.0,
        "brier": float(np.mean((p - y) ** 2)),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }
    if 0 < y.sum() < len(y):
        out["auc"] = float(roc_auc_score(y, p))
    return out


def video_decisions(video_ids, probs, threshold=0.5):
    """Disjunctive rule of Section III-B-2: a video is violent if any of its
    forwarded clips is. Videos with no forwarded clip must be added by the
    caller as 0."""
    out = {}
    for vid, p in zip(video_ids, probs):
        out[vid] = max(out.get(vid, 0.0), float(p))
    return {vid: int(p > threshold) for vid, p in out.items()}
