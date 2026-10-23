#!/usr/bin/env python
"""Run the pipeline on a video file, an RTSP/HTTP stream or a webcam (Algorithms 1 and 2).

Stage A turns frames into forwarded 16-frame ROI clips. With --stage-b, every clip is scored by
the Swin-Temporal recognizer as soon as it is cut, and the alert rule fires once an interaction
group has n_p = 3 consecutive positive decisions, then holds a 30 s cooldown. Without --stage-b
only Stage A runs.

Outputs (in --out-dir):
  annotated.mp4     tracks, interaction groups coloured by gate status, HUD
  clips/clip_NNNN.mp4 / .npy   every forwarded 16-frame ROI clip (the Stage B input; .npy is RGB uint8)
  events.jsonl      one line per forwarded clip: time span, members, ROI, trigger windows, features,
                    and with --stage-b the clip probability and whether it raised an alert
  summary.json      counts and per-component latency

Example:
  python scripts/run_pipeline.py --source data/rwf2000/RWF-2000/val/Fight/xxx.avi \\
      --stage-b runs/main/seed42/model.pt --out-dir outputs/demo
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from stage_a.config import config_from_dict, display_path, display_source, load_config  # noqa: E402
from stage_a.gate import StageAGate  # noqa: E402
from stage_a.paths import models_dir  # noqa: E402

log = logging.getLogger("run_pipeline")

COLORS = {  # BGR
    "single": (160, 160, 160), "warming": (200, 200, 200), "rejected_physics": (80, 200, 80),
    "rejected_svm": (0, 200, 255), "forwarded": (0, 0, 255),
}
LEGEND = [("warming up", "warming"), ("rejected: physics", "rejected_physics"),
          ("rejected: SVM", "rejected_svm"), ("forwarded to Stage B", "forwarded")]


def draw(frame, res, energy_units=None, energy_unit=0.1, fps_txt="", n_clips=0, show_energy=False):
    img = frame.copy()
    if show_energy and energy_units is not None:
        e = np.clip(energy_units.astype(np.float32) * energy_unit / 10.0, 0, 1)
        heat = cv2.applyColorMap((e * 255).astype(np.uint8), cv2.COLORMAP_JET)
        mask = (e > 0)[..., None]
        img = np.where(mask, cv2.addWeighted(img, 0.5, heat, 0.5, 0), img)
    for t in res.tracks:
        x1, y1, x2, y2 = map(int, t.box)
        c = (255, 200, 0) if t.matched else (140, 110, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 1)
        cv2.putText(img, str(t.tid), (x1 + 2, y1 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1, cv2.LINE_AA)
    for g in res.groups:
        if g.status == "single":
            continue
        x1, y1, x2, y2 = map(int, g.box)
        c = COLORS[g.status]
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
        label = f"G{list(g.members)}"
        if g.window is not None:
            f = g.window.features
            label += f" A={f[3]:.1f} mu={f[5]:.1f} S={f[4]:.0f}"
            if not np.isnan(g.score):
                label += f" s={g.score:+.2f}"
        ty = y1 - 4 if y1 > 34 else y1 + 14  # keep the label clear of the HUD bar
        cv2.putText(img, label, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, c, 1, cv2.LINE_AA)
    hud = f"t={res.packet.time:6.2f}s  {fps_txt}  clips={n_clips}"
    cv2.rectangle(img, (0, 0), (img.shape[1], 18), (0, 0, 0), -1)
    cv2.putText(img, hud, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    y = img.shape[0] - 6
    for text, key in reversed(LEGEND):
        cv2.putText(img, text, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLORS[key], 1, cv2.LINE_AA)
        y -= 13
    return img


def save_clip(clip, clip_dir: Path, fps: float) -> dict:
    clip_dir.mkdir(parents=True, exist_ok=True)
    stem = clip_dir / f"clip_{clip.clip_no:04d}"
    np.save(f"{stem}.npy", clip.frames[..., ::-1].copy())  # RGB for the recognizer
    h, w = clip.frames.shape[1:3]
    vw = cv2.VideoWriter(f"{stem}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in clip.frames:
        vw.write(f)
    vw.release()
    meta = clip.to_meta()
    meta.update(npy=f"clips/{stem.name}.npy", mp4=f"clips/{stem.name}.mp4")
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="video path, rtsp/http URL, or webcam index")
    ap.add_argument("--model", default=None, help="gate bundle (default models/stage_a_gate.joblib)")
    ap.add_argument("--config", default=None, help="use this YAML instead of the config stored in the bundle")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    ap.add_argument("--threshold", type=float, default=None, help="override the SVM operating threshold")
    ap.add_argument("--out-dir", default=None, help="default outputs/<source name>")
    ap.add_argument("--gpu", type=int, default=0, help="GPU index, -1 for CPU")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--no-video", action="store_true", help="do not write annotated.mp4")
    ap.add_argument("--no-clips", action="store_true", help="do not write ROI clips")
    ap.add_argument("--show-energy", action="store_true", help="overlay the motion-energy map")
    ap.add_argument("--display", action="store_true", help="show a live window (needs a display)")
    ap.add_argument("--threads", type=int, default=2, help="CPU threads for torch / OpenCV")
    ap.add_argument("--stage-b", default=None, metavar="CHECKPOINT",
                    help="Stage B model (scripts/train_stage_b.py output); omit to run Stage A only")
    ap.add_argument("--persistence", type=int, default=3, help="n_p consecutive positive decisions per alert")
    ap.add_argument("--cooldown", type=float, default=30.0, help="seconds an open alert absorbs further positives")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    from stage_a.frontend import limit_cpu_threads
    from stage_a.runtime import StageARuntime
    from stage_a.video import VideoReader

    limit_cpu_threads(args.threads)

    base = load_config(args.config)
    model = Path(args.model) if args.model else models_dir(base) / "stage_a_gate.joblib"
    gate = StageAGate.load(model)
    if args.threshold is not None:
        gate.threshold = args.threshold
    # the descriptor must match the one the gate was trained on: use the bundle's config by default
    cfg = load_config(args.config, args.set) if args.config else config_from_dict(gate.config, args.set)
    device = "cuda:0" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    src_name = Path(str(args.source)).stem if "://" not in str(args.source) else "stream"
    src_name = f"webcam{args.source}" if str(args.source).isdigit() else src_name
    out_dir = Path(args.out_dir) if args.out_dir else Path("outputs") / src_name
    out_dir.mkdir(parents=True, exist_ok=True)

    recognizer = alerts = None
    if args.stage_b:
        from stage_b.inference import AlertManager, GroupRegistry, Recognizer

        recognizer = Recognizer(args.stage_b, device)
        alerts = AlertManager(persistence=args.persistence, cooldown=args.cooldown)
        groups = GroupRegistry()

    def stage_b(frames):  # Stage A cuts BGR clips; the recognizer takes RGB
        return recognizer(frames[..., ::-1])

    rt = StageARuntime(cfg, gate, device, stage_b=stage_b if recognizer else None)
    reader = VideoReader(args.source, cfg.frontend.resize_width, cfg.frontend.analysis_fps, args.max_seconds)
    log.info("source %s: %dx%d @ %.2f fps -> analysis %d px wide @ %d FPS; threshold %.4f; device %s; stage B %s",
             args.source, reader.width, reader.height, reader.fps, cfg.frontend.resize_width,
             cfg.frontend.analysis_fps, gate.threshold, device, args.stage_b or "off")
    writer = None
    events = open(out_dir / "events.jsonl", "w")
    n_frames = n_analysis = n_windows = n_forwarded = n_clips = n_positive = n_alerts = 0
    t_start = time.perf_counter()
    fps_ema = None

    def emit(clip):
        nonlocal n_clips, n_positive, n_alerts
        meta = save_clip(clip, out_dir / "clips", cfg.frontend.analysis_fps) if not args.no_clips else clip.to_meta()
        msg = ""
        if alerts is not None:
            group = groups.key(clip.members)
            fired = alerts.update(group, clip.stage_b_prob, now=clip.t_end, windows=len(clip.triggers))
            meta.update(group=group, alert=fired)
            n_positive += int(clip.stage_b_prob > alerts.threshold)
            n_alerts += int(fired)
            msg = f" p={clip.stage_b_prob:.3f}" + ("  ALERT" if fired else "")
        events.write(json.dumps(meta) + "\n")
        events.flush()
        n_clips += 1
        log.info("clip %d: t=%.1f-%.1fs members=%s windows=%d%s", clip.clip_no, clip.t_start, clip.t_end,
                 list(clip.members), len(clip.triggers), msg)
    try:
        for pkt in reader:
            t0 = time.perf_counter()
            res = rt.process(pkt)
            dt = time.perf_counter() - t0
            fps_ema = 1.0 / dt if fps_ema is None else 0.9 * fps_ema + 0.1 / max(dt, 1e-6)
            n_frames += 1
            n_analysis += int(res.analysis)
            n_windows += len(res.windows)
            n_forwarded += len(res.forwarded)
            for clip in res.clips:
                emit(clip)
            if not args.no_video or args.display:
                img = draw(pkt.resized, res, rt.energy.units, rt.energy.unit, f"{fps_ema:5.1f} FPS", n_clips,
                           args.show_energy)
                if not args.no_video:
                    if writer is None:
                        h, w = img.shape[:2]
                        writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                                 reader.fps, (w, h))
                    writer.write(img)
                if args.display:
                    cv2.imshow("Stage A", img)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    except KeyboardInterrupt:
        log.info("interrupted")
    finally:
        for clip in rt.flush():
            emit(clip)
        events.close()
        reader.release()
        if writer is not None:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()
    wall = time.perf_counter() - t_start
    summary = {"source": display_source(args.source), "model": display_path(model), "threshold": gate.threshold,
               "frames": n_frames, "analysis_frames": n_analysis, "windows": n_windows,
               "forwarded_windows": n_forwarded, "stage_b_clips": n_clips,
               "stage_b_model": display_path(args.stage_b) if args.stage_b else None,
               "positive_clips": n_positive if args.stage_b else None,
               "alerts": n_alerts if args.stage_b else None,
               "wall_seconds": round(wall, 2), "end_to_end_fps": round(n_frames / max(wall, 1e-9), 2),
               "latency_ms": rt.timer.summary(), "out_dir": display_path(out_dir)}
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
