# Demo results

Three clips from the held-out test split, played through both stages with `scripts/demo_video.py`
(the program `scripts\demo.bat` runs) and a Stage B model trained with the code in this repository.
Trained weights are not distributed, so the model has to be trained first (see the main README).

The videos themselves are not included: they belong to the public datasets AVD and RWF-2000, which
we do not redistribute. Each clip is identified below by its path in the original release, so the
results can be reproduced after downloading the datasets with `scripts/download_data.py`.

| Source clip | Ground truth | Result |
|---|---|---|
| AVD `violent/cam1/86.mp4` | violent | first Stage B verdict at 2.2 s, alert at 3.2 s; 10/10 Stage B clips violent; the bystander on the right stays outside the red group |
| AVD `non-violent/cam1/59.mp4` | non-violent | group 1 checked 4 times, Stage B p = 0.00 every time; no alert |
| RWF-2000 `val/Fight/EFv961C5RgY_1.avi` | violent | alert at 3.2 s; a second group turns red at 4.2 s; 4/4 clips violent |

## Reading the output

Green boxes are tracked people, blue boxes interaction groups, orange a group forwarded by Stage A
and waiting for Stage B, and red a group Stage B classifies as violent; the red banner is the alert
rule (n_p = 3, 30 s cooldown). The bottom line shows the time, the processing rate, the Stage B clip
count and the alerts.

## Reproducing a result

```bat
scripts\demo.bat path\to\clip.mp4 --save result.mp4
```

On Linux or macOS:

```bash
python scripts/demo_video.py --source path/to/clip.mp4 --stage-b runs/main/seed42/model.pt --save result.mp4
```

The results above were rendered offline (`--no-display --no-realtime --fp32`); timings and the
processing rate depend on the GPU used.
