#!/usr/bin/env python3
"""
Stitch per-condition episode videos (from `scripts/evaluate.py --save_video`)
into one side-by-side comparison video, labeled per condition — the
baseline-vs-our-method deliverable for the results website
(see memory/project_video_deliverables).

Pure ffmpeg, no simulator needed — run from any shell with ffmpeg on PATH.

Usage:
    python scripts/make_comparison_video.py \
        --videos task_only=/data/heng/cdp/runs/eval_task_only_pick_egg_0_on_pick_egg/videos/task_only_pick_egg_ep0000_success.mp4 \
                 scalar=/data/heng/cdp/runs/eval_scalar_pick_egg_0_on_pick_egg/videos/scalar_pick_egg_ep0000_fail.mp4 \
                 vector=/data/heng/cdp/runs/eval_vector_pick_egg_0_on_pick_egg/videos/vector_pick_egg_ep0000_success.mp4 \
        --out results/videos/pick_egg_comparison.mp4
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def parse_video_arg(s: str) -> tuple[str, str]:
    label, path = s.split("=", 1)
    return label, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True, type=parse_video_arg,
                     metavar="LABEL=PATH", help="e.g. task_only=/path/a.mp4 scalar=/path/b.mp4 vector=/path/c.mp4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--height", type=int, default=480, help="per-panel output height (px)")
    args = ap.parse_args()

    for label, path in args.videos:
        if not os.path.exists(path):
            sys.exit(f"ERROR: video for {label!r} not found: {path}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    n = len(args.videos)
    inputs = []
    filter_parts = []
    labeled_streams = []
    for i, (label, path) in enumerate(args.videos):
        inputs += ["-i", path]
        filter_parts.append(
            f"[{i}:v]scale=-2:{args.height},"
            f"drawtext=text='{label}':x=10:y=10:fontsize=28:fontcolor=white:"
            f"box=1:boxcolor=black@0.5:boxborderw=6[v{i}]"
        )
        labeled_streams.append(f"[v{i}]")
    filter_complex = ";".join(filter_parts) + ";" + "".join(labeled_streams) + f"hstack=inputs={n}[out]"

    cmd = [
        "ffmpeg", "-y", *inputs,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        args.out,
    ]
    print(f"[make_comparison_video] {' '.join(cmd)}")
    subprocess.check_call(cmd)
    print(f"[make_comparison_video] wrote {args.out}")


if __name__ == "__main__":
    main()
