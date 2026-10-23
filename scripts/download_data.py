#!/usr/bin/env python
"""Download and unpack the five public sources into the folders configs/stage_a.yaml expects.

Kaggle serves these public datasets without an API key; AVD is cloned from the authors' GitHub
repository. PTIT is internal to the authors' institution and has no download link.

  * the RLVS archive holds the dataset twice; only "Real Life Violence Dataset/" is unpacked
  * 24 RWF-2000 file names exceed the 255-byte limit of most file systems once decoded; they are
    unpacked as longname<hash>_<n>.avi, keeping the <video>_<n> pattern the split grouping relies on

Example:
  python scripts/download_data.py                 # all five, ~17 GB of archives
  python scripts/download_data.py --only hockey movies
"""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KAGGLE = "https://www.kaggle.com/api/v1/datasets/download/"

# name -> (url, folder under data/, archive members to skip)
SOURCES = {
    "rwf2000": (KAGGLE + "vulamnguyen/rwf2000", "rwf2000", ()),
    "rlvs": (KAGGLE + "mohamedmustafa/real-life-violence-situations-dataset", "rlvs",
             ("real life violence situations/",)),
    "hockey": (KAGGLE + "yassershrief/hockey-fight-vidoes", "hockey_fight", ()),
    "movies": (KAGGLE + "naveenk903/movies-fight-detection-dataset", "movies", ()),
    "avd": ("https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos.git",
            "A-Dataset-for-Automatic-Violence-Detection-in-Videos-master", ()),
}
MAX_NAME_BYTES = 200


def fetch(url: str, dest: Path):
    tmp = dest.with_suffix(dest.suffix + ".part")
    start = tmp.stat().st_size if tmp.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-"} if start else {})
    with urllib.request.urlopen(req) as r:
        mode = "ab" if r.status == 206 else "wb"
        done = start if mode == "ab" else 0
        total = int(r.headers.get("Content-Length", 0)) + done
        with open(tmp, mode) as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {dest.name}: {done / 2**20:,.0f} / {total / 2**20:,.0f} MiB", end="", flush=True)
    print()
    tmp.rename(dest)


def safe_name(name: str) -> str:
    if len(name.encode("utf-8")) <= MAX_NAME_BYTES:
        return name
    stem, ext = Path(name).stem, Path(name).suffix
    m = re.match(r"(.*)_(\d+)$", stem)
    video, n = (m.group(1), m.group(2)) if m else (stem, "0")
    return f"longname{hashlib.md5(video.encode('utf-8')).hexdigest()[:10]}_{n}{ext}"


def unpack(archive: Path, out: Path, skip):
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            if info.is_dir() or any(info.filename.lower().startswith(s) for s in skip):
                continue
            parts = info.filename.split("/")
            dest = out.joinpath(*parts[:-1], safe_name(parts[-1]))
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--only", nargs="*", choices=sorted(SOURCES))
    ap.add_argument("--keep-archives", action="store_true")
    args = ap.parse_args()

    data = Path(args.data_dir)
    archives = data / "archives"
    archives.mkdir(parents=True, exist_ok=True)
    for name in args.only or SOURCES:
        url, folder, skip = SOURCES[name]
        out = data / folder
        if out.exists() and any(out.iterdir()):
            print(f"{name}: already in {out}")
            continue
        print(f"{name}: {url}")
        if url.endswith(".git"):
            subprocess.run(["git", "clone", "--depth", "1", url, str(out)], check=True)
            shutil.rmtree(out / ".git", ignore_errors=True)
            continue
        zpath = archives / f"{name}.zip"
        if not zpath.exists():
            fetch(url, zpath)
        unpack(zpath, out, skip)
        if not args.keep_archives:
            zpath.unlink()
    print("PTIT is internal and is not downloaded; see README.")


if __name__ == "__main__":
    sys.exit(main())
