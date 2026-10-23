# Real-Time Violence Detection via Kinematic Multi-Stage Gating and ROI-Based Swin–Temporal Transformer

Source code of the paper by N. T. Ha, T. D. Pham, H. T. Nguyen and M. D. Nguyen.

* **Stage A** (`stage_a/`, `configs/stage_a.yaml`): YOLOv8s + DeepSORT, motion-energy map,
  adaptive grouping, 7-D window descriptor, physics gate and RBF-SVM, 16-frame ROI clips.
* **Stage B** (`stage_b/`, `configs/stage_b.yaml`): Swin-T per frame, learned positions,
  2-layer Transformer, MLP head; decoupled training (adaptation, then temporal encoder).

Every constant in the two config files is taken from the paper; comments give the section.

## Scope

This repository implements the proposed pipeline end to end: data preparation, the training and
validation of both stages, their evaluation on the test split, and real-time inference with the
alert rule. It does not include the re-implemented baselines or the auxiliary analyses reported in
the paper (robustness to synthetic corruption, the multi-stream throughput benchmark, the
scene-grouped split and the detector and tracker evaluation on the annotated subset).

## Code structure

Each module states in its docstring the part of the paper it implements.

| Paper | Code |
|---|---|
| Fig. 1, overall pipeline (Stage A gate, Stage B recognizer) | `scripts/run_pipeline.py`, `scripts/demo_video.py` |
| III-A-1, front end: resize, 10-FPS analysis grid, YOLOv8 + DeepSORT | `stage_a/video.py`, `stage_a/frontend.py` |
| III-A-1, adaptive proximity grouping (T_adaptive) | `stage_a/grouping.py` |
| III-A-3, motion-energy map E_t | `stage_a/energy.py` |
| III-A-4, per-person kinematics and the 7-D window descriptor V_raw | `stage_a/descriptor.py` |
| III-A-5, physics threshold and RBF-SVM gate | `stage_a/gate.py`, `scripts/train_stage_a.py` |
| Algorithm 1, online Stage A and the 16-frame ROI clips | `stage_a/runtime.py`, `stage_a/clips.py` |
| III-B, Fig. 6, Swin-T + Temporal Transformer | `stage_b/model.py` |
| III-B-4, decoupled training (adaptation, then temporal encoder) | `stage_b/engine.py`, `stage_b/data.py`, `scripts/train_stage_b.py` |
| Algorithm 2, clip decision, video rule and alerting | `stage_b/inference.py` |
| IV-A, datasets, screening, balancing and the 80/20 split | `stage_a/datasets.py`, `scripts/download_data.py`, `scripts/prepare_data.py` |
| IV, evaluation of Stage A, Stage B and the full pipeline | `scripts/eval_stage_a.py`, `scripts/eval_stage_b.py`, `scripts/eval_pipeline.py` |
| IV-D, Fig. 5, Stage A signals on labelled segments | `scripts/diagnose_video.py` |
| Settings of Tables VI and XXXVIII | `configs/stage_a.yaml`, `configs/stage_b.yaml` |

## Setup

Python 3.10, one CUDA GPU (tested on an RTX 2080 Ti, 11 GB).

```bash
conda create -n violence-detection python=3.10 -y && conda activate violence-detection
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python -m pytest -q tests
```

## Data

| Source | Release | Used in the paper (V / NV) |
|---|---|---|
| RWF-2000 | kaggle.com/datasets/vulamnguyen/rwf2000 | 1000 / 1000 |
| RLVS | kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset | 978 / 978 |
| Hockey Fight | kaggle.com/datasets/yassershrief/hockey-fight-vidoes | 500 / 500 |
| AVD | github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos | 120 / 120 |
| Movies V/N-V | kaggle.com/datasets/naveenk903/movies-fight-detection-dataset | 100 / 100 |
| PTIT | internal, not distributed | 160 / 160 |

`python scripts/download_data.py` fetches the five public sources into `data/`. RLVS and AVD are
screened and balanced 1:1; the benchmark is split 80/20 (4,573 / 1,143, seed 42) and the training
part again 80/20 for Stage B (3,658 / 915). PTIT, if available, goes under
`data/ptit/{violent,non_violent}/` with `sources.ptit.enabled: true`.

## Running

```bash
# Stage A
python scripts/prepare_data.py scan
python scripts/prepare_data.py track --gpu 0
python scripts/prepare_data.py windows
python scripts/prepare_data.py split
python scripts/train_stage_a.py
python scripts/eval_stage_a.py --split test

# Stage B
python scripts/extract_stage_b_clips.py
python scripts/train_stage_b.py --phase adapt --out runs/adapt
python scripts/train_stage_b.py --phase temporal --backbone runs/adapt/backbone.pt --seed 42 --out runs/main/seed42
python scripts/eval_stage_b.py --ckpt "runs/main/seed*/model.pt" --out runs/main/test

# both stages, end to end on the test videos
python scripts/eval_pipeline.py --stage-b "runs/main/seed*/model.pt" --out runs/main/pipeline_test

# both stages on a video, RTSP stream or webcam, with the alert rule
python scripts/run_pipeline.py --source video.mp4 --stage-b runs/main/seed42/model.pt --out-dir outputs/demo

# Stage A signals on labelled time segments of one video (the diagnostic of Sec. IV-D)
python scripts/diagnose_video.py --source video.mp4 --segments walking:20-50 fight:110-140 --stride 1
```

`bash scripts/reproduce.sh` runs the whole protocol (five seeds; `LODO=1` adds leave-one-dataset-out).
Any setting can be overridden with `--set section.key=value`; `train_stage_b.py` also takes
`--phase joint`, `--holdout <source>` and `--cv-fold <k>`.

## Models and weights

No trained weights are distributed with this repository. Two pretrained networks are downloaded
automatically on first use: the YOLOv8s person detector (by Ultralytics, to `models/yolov8s.pt`)
and the ImageNet-1K Swin-T backbone (by `timm`). Everything else, the Stage A gate
(`models/stage_a_gate.joblib`) and the Stage B checkpoints (`runs/main/seed*/model.pt`), is
produced by the training commands above.

## Demo

The demo needs a Stage B checkpoint, so run the training steps above first. It needs an NVIDIA GPU with CUDA and at least 4 GB of VRAM (the demo itself uses under 1 GB); an RTX
card is recommended, on a GPU without tensor cores add `--fp32`. Training Stage B needs about 12 GB.

```bat
scripts\demo.bat path\to\video.mp4
```

A window plays the video with the people tracked (green), interaction groups (blue), groups being
checked by Stage B (orange) and fighting groups in red with a VIOLENCE ALERT banner. Space pauses,
q quits. Add `--save result.mp4` to also save the annotated video.

`demo.bat` runs `scripts/demo_video.py` with the checkpoint `runs/main/seed42/model.pt` written by the
Stage B training step; on Linux
or macOS run `python scripts/demo_video.py --source video.mp4 --stage-b runs/main/seed42/model.pt`.
`python scripts/demo_video.py --help` lists the other options.

Expected results on three public test clips are described in [result_demo/README.md](result_demo/README.md).

## License

The code in this repository is released under the MIT License, see [LICENSE](LICENSE).

Third-party packages keep their own licenses. In particular, the person detector relies on
[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), which is distributed under
AGPL-3.0; using or redistributing the full pipeline therefore also has to respect the terms of
that license. DeepSORT (`deep-sort-realtime`) is MIT-licensed and `timm` is Apache-2.0.

## Citation

If you use this code, please cite the paper:

> N. T. Ha, T. D. Pham, H. T. Nguyen and M. D. Nguyen, "Real-Time Violence Detection via Kinematic
> Multi-Stage Gating and ROI-Based Swin-Temporal Transformer," *Scientific Reports*, under review.

Each release of this repository is archived on Zenodo with its own DOI, shown on the repository
page and in the paper; citation metadata are in [CITATION.cff](CITATION.cff).
