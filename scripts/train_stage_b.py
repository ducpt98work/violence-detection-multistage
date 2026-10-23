"""Train Stage B with the decoupled schedule of Section III-B-4.

Phase 1 adapts Swin-T to the ROI crops with a temporary head (5 epochs,
lr 1e-5) and saves the backbone. Phase 2 freezes it and trains the positional
table, the temporal encoder and the MLP head for 50 epochs.

    # phase 1, once
    python scripts/train_stage_b.py --phase adapt --out runs/adapt
    # phase 2, one run per seed
    python scripts/train_stage_b.py --phase temporal --backbone runs/adapt/backbone.pt --seed 42 --out runs/main/seed42
    # end-to-end ablation ("Training scheme" block of the Stage B ablation)
    python scripts/train_stage_b.py --phase joint --seed 42 --out runs/joint/seed42
    # leave-one-dataset-out fold: un-adapted ImageNet encoder, source held out
    python scripts/train_stage_b.py --phase temporal --backbone imagenet --holdout hockey --out runs/lodo/hockey/seed42
    # one fold of the 5-fold CV on the backprop partition, e.g. for the depth sweep
    python scripts/train_stage_b.py --phase temporal --backbone runs/adapt/backbone.pt --cv-fold 0 --set model.depth=4 --out runs/cv/depth4/fold0
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import timm
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage_a.config import display_path  # noqa: E402
from stage_b.config import load_config  # noqa: E402
from stage_b.data import read_benchmark, select  # noqa: E402
from stage_b.engine import build_model, make_loader, run_epoch, seed_everything  # noqa: E402
from stage_b.metrics import binary_metrics  # noqa: E402
from stage_b.model import FrameHead, count_parameters  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="default configs/stage_b.yaml")
    ap.add_argument("--phase", choices=["adapt", "temporal", "joint"], required=True)
    ap.add_argument("--backbone", default="imagenet",
                    help="adapted backbone from phase 1, or 'imagenet' for the raw checkpoint")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--holdout", help="source left out for a leave-one-dataset-out fold")
    ap.add_argument("--cv-fold", type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.set)
    seed = cfg["seed"] if args.seed is None else args.seed
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.yaml", "w", encoding="utf-8") as f:
        run = {k: (display_path(v) if k in ("config", "backbone", "out") and v and v != "imagenet" else v)
               for k, v in vars(args).items()}
        yaml.safe_dump({**cfg, "seed": seed, "run": run}, f, sort_keys=False)

    seed_everything(seed)
    device = torch.device("cuda")
    rows = read_benchmark(cfg["data"]["benchmark"])
    split = dict(holdout=args.holdout, cv_fold=args.cv_fold, seed=cfg["seed"])
    train_rows, val_rows = select(rows, "train", **split), select(rows, "val", **split)
    print(f"train {len(train_rows)} clips, val {len(val_rows)} clips")

    if args.phase == "adapt":
        sched_cfg = cfg["adapt"]
        backbone = timm.create_model(cfg["model"]["backbone"], pretrained=True, num_classes=0)
        model = FrameHead(backbone, cfg["model"]["head_hidden"])
        if sched_cfg.get("grad_checkpointing"):
            model.backbone.set_grad_checkpointing(True)
        amp = True
    else:
        sched_cfg = cfg["train"]
        model = build_model(cfg)
        if args.phase == "temporal":
            if args.backbone != "imagenet":
                model.backbone.load_state_dict(torch.load(args.backbone, map_location="cpu"))
            model.freeze_backbone()
        elif cfg["adapt"].get("grad_checkpointing"):
            model.backbone.set_grad_checkpointing(True)
        amp = sched_cfg["amp"]
    model.to(device)
    print(f"parameters: {count_parameters(model) / 1e6:.2f} M total, "
          f"{count_parameters(model, trainable_only=True) / 1e6:.2f} M trainable")

    bs = sched_cfg["batch_size"]
    train_loader = make_loader(cfg, train_rows, True, bs, seed)
    val_loader = make_loader(cfg, val_rows, False, bs)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=sched_cfg["lr"], weight_decay=sched_cfg["weight_decay"])
    epochs = sched_cfg["epochs"]
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, sched_cfg["min_lr"])
                 if "min_lr" in sched_cfg else None)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    grad_clip = sched_cfg.get("grad_clip")
    threshold = cfg["train"]["threshold"]

    # the settings in effect, read back from the objects that use them
    opt = optimizer.defaults
    schedule = (f"cosine T_max={scheduler.T_max} eta_min={scheduler.eta_min:g}" if scheduler
                else "constant lr")
    dropout = (f"{model.temporal.layers[0].dropout.p:g} (attention {model.temporal.layers[0].self_attn.dropout:g})"
               if hasattr(model, "temporal") else "none in the temporary head")
    print(f"{args.phase}: {epochs} epochs, {type(optimizer).__name__} lr={opt['lr']:g} "
          f"weight_decay={opt['weight_decay']:g}, batch {bs}, {schedule}, dropout {dropout}, "
          f"AMP {'fp16' if amp else 'off'}, grad clip {grad_clip}, seed {seed}")

    log_path = out / "log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "lr", "train_loss", "train_f1", "val_loss", "val_f1",
                                "val_acc", "seconds", "peak_mem_gb"])
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, epochs + 1):
        start = time.time()
        lr = optimizer.param_groups[0]["lr"]
        tr_loss, tr_p, tr_y = run_epoch(model, train_loader, device, optimizer, scaler, grad_clip, amp)
        with torch.no_grad():
            va_loss, va_p, va_y = run_epoch(model, val_loader, device, amp=amp)
        if scheduler:
            scheduler.step()
        tr, va = binary_metrics(tr_y, tr_p, threshold), binary_metrics(va_y, va_p, threshold)
        secs, mem = time.time() - start, torch.cuda.max_memory_allocated() / 2**30
        print(f"epoch {epoch:3d}  lr {lr:.2e}  train {tr_loss:.4f}/{tr['f1']:.4f}  "
              f"val {va_loss:.4f}/{va['f1']:.4f}  {secs:.0f}s  {mem:.1f} GB", flush=True)
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{lr:.3e}", f"{tr_loss:.5f}", f"{tr['f1']:.5f}",
                                    f"{va_loss:.5f}", f"{va['f1']:.5f}", f"{va['acc']:.5f}",
                                    f"{secs:.1f}", f"{mem:.2f}"])

    if args.phase == "adapt":
        torch.save(model.backbone.state_dict(), out / "backbone.pt")
    else:
        torch.save({"config": {**cfg, "seed": seed}, "model": model.state_dict(), "epoch": epochs},
                   out / "model.pt")
    with open(out / "val_metrics.json", "w") as f:
        json.dump(va, f, indent=2)


if __name__ == "__main__":
    main()
