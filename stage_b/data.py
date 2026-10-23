import csv

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from .config import resolve_path

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Stored clips carry one extra analysis frame on each side of the 16-frame
# window (t-9 .. t+8) so temporal jitter can reach a neighbour.
STORED_FRAMES = 18


def read_benchmark(path):
    with open(resolve_path(path), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["label"] = int(r["label"])
    return rows


def clip_file(clip_dir, clip_id):
    return resolve_path(clip_dir) / (clip_id.replace("/", "__") + ".npy")


def stratified_folds(rows, n_folds, seed):
    """Fold index per row, stratified by (source, label)."""
    rng = np.random.default_rng(seed)
    fold = {}
    strata = {}
    for r in rows:
        strata.setdefault((r["source"], r["label"]), []).append(r["clip_id"])
    for key in sorted(strata):
        ids = sorted(strata[key])
        rng.shuffle(ids)
        for i, cid in enumerate(ids):
            fold[cid] = i % n_folds
    return fold


def select(rows, split, holdout=None, cv_fold=None, n_folds=5, seed=42):
    """Rows for one partition.

    split: "train" (backprop), "val" or "test" of the benchmark split.
    holdout: leave-one-dataset-out; "train"/"val" drop that source and "test"
        returns every clip of it.
    cv_fold: 5-fold stratified CV inside the backprop partition; "train" is the
        other four folds and "val" is this fold.
    """
    if holdout:
        if split == "test":
            return [r for r in rows if r["source"] == holdout]
        rows = [r for r in rows if r["source"] != holdout]
    if cv_fold is not None and split in ("train", "val"):
        pool = [r for r in rows if r["split"] == "train"]
        fold = stratified_folds(pool, n_folds, seed)
        keep = (lambda f: f == cv_fold) if split == "val" else (lambda f: f != cv_fold)
        return [r for r in pool if keep(fold[r["clip_id"]])]
    return [r for r in rows if r["split"] == split]


class ClipDataset(Dataset):
    """16-frame ROI clips with the augmentation of the training-configuration table: temporal jitter,
    horizontal flip, scale jitter before the resize, brightness and contrast.
    Spatial and photometric parameters are drawn once per clip."""

    def __init__(self, rows, clip_dir, train=False, num_frames=16, image_size=224,
                 hflip=0.5, scale=(0.8, 1.0), brightness=0.2, contrast=0.2, temporal_jitter=0.05):
        self.rows = rows
        self.clip_dir = clip_dir
        self.train = train
        self.num_frames = num_frames
        self.size = [image_size, image_size]
        self.hflip, self.scale = hflip, scale
        self.brightness, self.contrast = brightness, contrast
        self.temporal_jitter = temporal_jitter

    def __len__(self):
        return len(self.rows)

    def _uniform(self, lo, hi):
        return lo + (hi - lo) * torch.rand(()).item()

    def __getitem__(self, i):
        row = self.rows[i]
        frames = np.load(clip_file(self.clip_dir, row["clip_id"]))
        offset = (len(frames) - self.num_frames) // 2
        idx = torch.arange(offset, offset + self.num_frames)
        if self.train and self.temporal_jitter > 0:
            moved = torch.rand(self.num_frames) < self.temporal_jitter
            step = torch.randint(0, 2, (self.num_frames,)) * 2 - 1
            idx = (idx + moved * step).clamp(0, len(frames) - 1)
        clip = torch.from_numpy(frames[idx.numpy()]).permute(0, 3, 1, 2)

        if self.train:
            side = clip.shape[-1]
            crop = round(side * self._uniform(*self.scale))
            top, left = torch.randint(0, side - crop + 1, (2,)).tolist()
            clip = TF.resized_crop(clip, top, left, crop, crop, self.size, antialias=True)
            if torch.rand(()) < self.hflip:
                clip = TF.hflip(clip)
            clip = clip.float() / 255
            clip = TF.adjust_brightness(clip, self._uniform(1 - self.brightness, 1 + self.brightness))
            clip = TF.adjust_contrast(clip, self._uniform(1 - self.contrast, 1 + self.contrast))
        else:
            clip = TF.resize(clip, self.size, antialias=True).float() / 255
        clip = (clip - IMAGENET_MEAN) / IMAGENET_STD
        return clip, torch.tensor(row["label"], dtype=torch.float32)
