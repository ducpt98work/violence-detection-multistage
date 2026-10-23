#!/usr/bin/env python
"""Play a video (file, RTSP/HTTP stream or webcam) through both stages and draw the result the way
the paper's qualitative figures do:

  * every tracked person: thin green box with the detector confidence
  * an interaction group: blue box "Group k" while nothing is suspected
  * orange "Group k - checking": Stage A forwarded the group, the Stage B verdict is not in yet
    (a 16-frame clip needs the 7 analysis frames after the trigger, ~0.7 s)
  * red "Group k - Violence (p)": Stage B classified the group's latest clip as violent
  * a red ALERT banner once the alert rule of Algorithm 2 fires (n_p = 3 consecutive positive
    decisions for a group), held for the 30 s cooldown

Without --stage-b only Stage A runs and suspected groups stay orange.

Examples:
  python scripts/demo_video.py --source data/rwf2000/RWF-2000/val/Fight/xxx.avi --stage-b runs/main/seed42/model.pt
  python scripts/demo_video.py --source rtsp://... --stage-b runs/main/seed42/model.pt --save outputs/cam.mp4
  python scripts/demo_video.py --source 0 --stage-b runs/main/seed42/model.pt          # webcam
  python scripts/demo_video.py --source video.mp4 --stage-b model.pt --no-display --save out.mp4   # headless

Keys in the window: space pauses, q or Esc quits.
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

log = logging.getLogger("demo_video")

# BGR, as in the paper's figures
PERSON = (60, 220, 60)
NORMAL = (230, 90, 20)
CHECKING = (0, 150, 255)
VIOLENCE = (30, 30, 220)
WHITE = (255, 255, 255)


def put_label(img, text, x, y, color, scale=0.55, thickness=1):
    """Text on a filled box whose bottom-left corner is (x, y), kept inside the image."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = int(min(max(x, 0), img.shape[1] - tw - 4))
    y = int(min(max(y, th + base + 2), img.shape[0] - 1))
    cv2.rectangle(img, (x, y - th - base - 2), (x + tw + 4, y), color, -1)
    cv2.putText(img, text, (x + 2, y - base), cv2.FONT_HERSHEY_SIMPLEX, scale, WHITE, thickness, cv2.LINE_AA)


def union_box(boxes):
    b = np.asarray(boxes, np.float32)
    return np.array([b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()], np.float32)


class Verdicts:
    """Latest Stage B decision per display group, kept on screen for `hold` seconds."""

    def __init__(self, threshold=0.5, hold=2.0):
        self.threshold = threshold
        self.hold = hold
        self.latest = {}   # group -> (prob, time)

    def set(self, group, prob, now):
        self.latest[group] = (prob, now)

    def get(self, group, now):
        v = self.latest.get(group)
        if v is None or now - v[1] > self.hold:
            return None
        return v[0]


def make_recognizer(checkpoint, device, amp):
    import torch
    from stage_b.inference import Recognizer

    class _Recognizer(Recognizer):
        """The library recognizer with AMP made optional: fp16 is several times slower than fp32 on
        GPUs without tensor cores."""

        def __init__(self, ckpt, dev, use_amp):
            super().__init__(ckpt, dev)
            self.amp = use_amp and dev != "cpu"

        @torch.no_grad()
        def __call__(self, crops):
            x = self.preprocess(crops).to(self.device)
            with torch.autocast("cuda", torch.float16, enabled=self.amp):
                return torch.sigmoid(self.model(x).float()).item()

    return _Recognizer(checkpoint, device, amp)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="video path, rtsp/http URL, or webcam index")
    ap.add_argument("--stage-b", default=None, metavar="CHECKPOINT", help="Stage B model.pt; omit for Stage A only")
    ap.add_argument("--model", default=None, help="Stage A gate bundle (default models/stage_a_gate.joblib)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="Stage A config overrides")
    ap.add_argument("--threshold", type=float, default=0.5, help="Stage B violence threshold")
    ap.add_argument("--persistence", type=int, default=3, help="n_p consecutive positive decisions per alert")
    ap.add_argument("--cooldown", type=float, default=30.0, help="seconds an alert stays open")
    ap.add_argument("--hold", type=float, default=2.0, help="seconds a Stage B verdict stays on a group box")
    ap.add_argument("--save", default=None, help="write the annotated video here (.mp4)")
    ap.add_argument("--events", default=None, help="write one JSON line per Stage B clip here")
    ap.add_argument("--no-display", action="store_true", help="do not open a window (headless)")
    ap.add_argument("--no-realtime", action="store_true", help="show frames as fast as they are processed")
    ap.add_argument("--display-width", type=int, default=1280, help="window / saved video width, 0 = source width")
    ap.add_argument("--fp32", action="store_true", help="run Stage B without AMP (much faster on GPUs without tensor cores)")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--gpu", type=int, default=0, help="GPU index, -1 for CPU")
    ap.add_argument("--threads", type=int, default=2, help="CPU threads for torch / OpenCV")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    from stage_a.frontend import limit_cpu_threads
    from stage_a.runtime import StageARuntime
    from stage_a.video import VideoReader
    from stage_b.inference import AlertManager, GroupRegistry

    limit_cpu_threads(args.threads)
    device = "cuda:0" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    base = load_config()
    gate = StageAGate.load(Path(args.model) if args.model else models_dir(base) / "stage_a_gate.joblib")
    cfg = config_from_dict(gate.config, args.set)   # the descriptor the gate was trained with

    recognizer = make_recognizer(args.stage_b, device, not args.fp32) if args.stage_b else None
    stage_b = (lambda frames: recognizer(frames[..., ::-1])) if recognizer else None  # clips are BGR
    rt = StageARuntime(cfg, gate, device, stage_b=stage_b)
    reader = VideoReader(args.source, cfg.frontend.resize_width, cfg.frontend.analysis_fps, args.max_seconds)
    log.info("source %s: %dx%d @ %.2f fps; Stage B %s (%s); device %s", args.source, reader.width, reader.height,
             reader.fps, args.stage_b or "off", "fp32" if args.fp32 else "AMP fp16", device)

    registry = GroupRegistry()
    verdicts = Verdicts(args.threshold, args.hold)
    alerts = AlertManager(persistence=args.persistence, cooldown=args.cooldown, threshold=args.threshold)
    groups = []          # [(display id, members, status)] from the last analysis frame
    alert_log = []       # [(group, time)]
    writer = None
    events = open(args.events, "w") if args.events else None
    n_frames = n_clips = n_positive = 0
    t_start = time.perf_counter()
    fps_ema = None
    paused = False
    window = "Violence detection (q: quit, space: pause)"

    try:
        for pkt in reader:
            t0 = time.perf_counter()
            res = rt.process(pkt)
            now = pkt.time

            # Stage B verdicts of the clips cut at this frame
            for clip in res.clips:
                gid = registry.key(clip.members)
                n_clips += 1
                if clip.stage_b_prob is None:
                    continue
                verdicts.set(gid, clip.stage_b_prob, now)
                fired = alerts.update(gid, clip.stage_b_prob, now=now, windows=len(clip.triggers))
                n_positive += int(clip.stage_b_prob > args.threshold)
                if fired:
                    alert_log.append((gid, now))
                    log.warning("ALERT group %d at %.1f s (p=%.2f)", gid + 1, now, clip.stage_b_prob)
                if events:
                    events.write(json.dumps({"time": round(now, 2), "group": gid + 1, "members": list(clip.members),
                                             "clip_t": [round(clip.t_start, 2), round(clip.t_end, 2)],
                                             "prob": round(clip.stage_b_prob, 4), "alert": fired}) + "\n")
            if res.analysis:
                groups = [(registry.key(g.members), g.members, g.status) for g in res.groups
                          if len(g.members) >= cfg.descriptor.min_members]

            # draw on the source-resolution frame
            img = pkt.frame.copy()
            s = 1.0 / rt.scale
            boxes = {t.tid: t.box * s for t in res.tracks}
            for t in res.tracks:
                x1, y1, x2, y2 = map(int, boxes[t.tid])
                cv2.rectangle(img, (x1, y1), (x2, y2), PERSON, 2)
                if t.matched and not np.isnan(t.conf):
                    put_label(img, f"{t.conf:.2f}", x1, y1, PERSON, 0.45)
            violent_now = []
            for gid, members, status in groups:
                present = [boxes[m] for m in members if m in boxes]
                if not present:
                    continue
                x1, y1, x2, y2 = map(int, union_box(present))
                prob = verdicts.get(gid, now)
                if prob is not None and prob > args.threshold:
                    color, text, width = VIOLENCE, f"Group {gid + 1} - Violence ({prob:.2f})", 4
                    violent_now.append(gid)
                elif status == "forwarded" and prob is None:
                    color, text, width = CHECKING, f"Group {gid + 1} - checking", 3
                else:
                    color, text, width = NORMAL, f"Group {gid + 1}" + (f" ({prob:.2f})" if prob is not None else ""), 2
                cv2.rectangle(img, (x1, y1), (x2, y2), color, width)
                put_label(img, text, x1, y1 - 2, color, 0.7, 2)

            open_alerts = [g for g, deadline in alerts.open_alerts.items() if deadline >= now]
            if open_alerts:
                bar = "VIOLENCE ALERT - " + ", ".join(f"Group {g + 1}" for g in sorted(open_alerts))
                cv2.rectangle(img, (0, 0), (img.shape[1], 46), VIOLENCE, -1)
                cv2.putText(img, bar, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2, cv2.LINE_AA)
            dt = time.perf_counter() - t0
            fps_ema = 1.0 / max(dt, 1e-6) if fps_ema is None else 0.9 * fps_ema + 0.1 / max(dt, 1e-6)
            hud = f"t={now:6.1f}s  {fps_ema:5.1f} FPS  Stage B clips {n_clips}  violent {n_positive}  alerts {len(alert_log)}"
            put_label(img, hud, 8, img.shape[0] - 8, (40, 40, 40), 0.6, 1)

            if args.display_width and img.shape[1] != args.display_width:
                h = int(round(img.shape[0] * args.display_width / img.shape[1]))
                img = cv2.resize(img, (args.display_width, h), interpolation=cv2.INTER_AREA)
            if args.save:
                if writer is None:
                    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), reader.fps,
                                             (img.shape[1], img.shape[0]))
                writer.write(img)
            if not args.no_display:
                cv2.imshow(window, img)
                wait = 1
                if not args.no_realtime and not reader.is_live:  # play at the source rate when faster than it
                    wait = max(1, int(1000 * (1.0 / reader.fps - (time.perf_counter() - t0))))
                key = cv2.waitKey(wait) & 0xFF
                if key == ord(" "):
                    paused = not paused
                while paused:
                    key = cv2.waitKey(50) & 0xFF
                    if key == ord(" "):
                        paused = False
                    elif key in (ord("q"), 27):
                        break
                if key in (ord("q"), 27):
                    break
            n_frames += 1
    except KeyboardInterrupt:
        log.info("interrupted")
    finally:
        reader.release()
        if writer is not None:
            writer.release()
        if events:
            events.close()
        if not args.no_display:
            cv2.destroyAllWindows()
    wall = time.perf_counter() - t_start
    summary = {"source": display_source(args.source), "frames": n_frames, "stage_b_clips": n_clips,
               "violent_clips": n_positive, "alerts": [{"group": g + 1, "time": round(t, 2)} for g, t in alert_log],
               "wall_seconds": round(wall, 1), "fps": round(n_frames / max(wall, 1e-9), 1),
               "latency_ms": rt.timer.summary(), "saved": display_path(args.save) if args.save else None}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
