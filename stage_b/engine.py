import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import ClipDataset
from .model import SwinTemporal


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg, pretrained=None):
    m = dict(cfg["model"])
    if pretrained is not None:
        m["pretrained"] = pretrained
    return SwinTemporal(num_frames=cfg["data"]["num_frames"], **m)


def make_loader(cfg, rows, train, batch_size, seed=0):
    d, aug = cfg["data"], cfg["augment"]
    ds = ClipDataset(rows, d["clips"], train=train, num_frames=d["num_frames"],
                     image_size=d["image_size"], **aug)
    return DataLoader(ds, batch_size=batch_size, shuffle=train,
                      generator=torch.Generator().manual_seed(seed) if train else None,
                      num_workers=d["num_workers"], pin_memory=True,
                      persistent_workers=d["num_workers"] > 0)


def run_epoch(model, loader, device, optimizer=None, scaler=None, grad_clip=None, amp=True):
    """One pass over the loader; trains if an optimizer is given.
    Returns (mean loss, probabilities, labels)."""
    training = optimizer is not None
    model.train(training)
    loss_fn = nn.BCEWithLogitsLoss()
    params = [p for g in optimizer.param_groups for p in g["params"]] if training else []
    total, probs, labels = 0.0, [], []
    for clips, y in loader:
        clips = clips.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.set_grad_enabled(training), torch.autocast("cuda", torch.float16, enabled=amp):
            logits = model(clips).float()
            loss = loss_fn(logits, y)
        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(params, grad_clip)
            scaler.step(optimizer)
            scaler.update()
        total += loss.item() * len(y)
        probs.append(torch.sigmoid(logits).detach().cpu())
        labels.append(y.cpu())
    probs, labels = torch.cat(probs).numpy(), torch.cat(labels).numpy()
    return total / len(labels), probs, labels


@torch.no_grad()
def predict(model, loader, device, amp=True):
    _, probs, labels = run_epoch(model, loader, device, amp=amp)
    return probs, labels
