#!/usr/bin/env python
"""Visualize the full spatiotemporal sample for any D-NeRF subject.

D-NeRF stores one camera view per time step in each split (train=200,
val/test=20 frames), so the "full sample" is a temporal filmstrip. This
script lays out every frame left-to-right, wrapping into rows, with a
per-cell time label. Self-contained: parses ``transforms_*.json`` and
reads PNGs with OpenCV — no torch / no Rust-preprocessing dependency.

Usage
-----
    # Default: trex, train split, all 200 frames in a wrapped filmstrip
    python scripts/visualize_sample.py --subject trex

    # Test split (20 frames) at larger size
    python scripts/visualize_sample.py --subject hook --split test --cell 128

    # Specific output
    python scripts/visualize_sample.py --subject bouncingballs --out runs/balls.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# D-NeRF JSON parsing (mirrors src/mcrah/data/scene.py without the torch dep)
# --------------------------------------------------------------------------- #
def load_frames(data_root: Path, category: str, split: str) -> list[dict]:
    path = Path(data_root) / category / f"transforms_{split}.json"
    with open(path) as f:
        return json.load(f)["frames"]


def time_index_map(frames: list[dict]) -> list[int]:
    """Assign each frame an integer temporal index from its ``time`` value."""
    seen: dict[float, int] = {}
    idx: list[int] = []
    for fr in frames:
        t = round(float(fr["time"]), 6)
        if t not in seen:
            seen[t] = len(seen)
        idx.append(seen[t])
    return idx


def load_image(data_root: Path, category: str, frame: dict) -> np.ndarray:
    """Read a D-NeRF PNG as a (H, W, 3) BGR uint8 array."""
    path = Path(data_root) / category / (frame["file_path"] + ".png")
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read {path}")
    if img.ndim == 2:                       # grayscale -> BGR
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[-1] == 4:                # RGBA -> composite on white
        rgb = img[..., :3]
        alpha = img[..., 3:4].astype(np.float32) / 255.0
        img = (rgb * alpha + 255 * (1.0 - alpha)).astype(np.uint8)
    return img[..., :3]


# --------------------------------------------------------------------------- #
# Wrapped filmstrip
# --------------------------------------------------------------------------- #
def make_filmstrip(
    cells: list[np.ndarray],
    labels: list[str],
    cell: int,
    cols: int,
    pad: int,
    margin: int,
) -> np.ndarray:
    """Lay out frames left-to-right, wrapping into rows, with per-cell labels."""
    n = len(cells)
    rows_n = (n + cols - 1) // cols
    W = margin + cols * (cell + pad)
    H = margin + rows_n * (cell + pad) + 20   # extra for bottom edge
    canvas = np.full((H, W, 3), 255, np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = max(0.35, cell / 400.0)
    thick = 1 if cell < 128 else 2

    for i, (img, lab) in enumerate(zip(cells, labels)):
        r, c = divmod(i, cols)
        x = margin + c * (cell + pad)
        y = margin + r * (cell + pad)
        thumb = cv2.resize(img, (cell, cell), interpolation=cv2.INTER_AREA)
        canvas[y:y + cell, x:x + cell] = thumb
        cv2.rectangle(canvas, (x, y), (x + cell - 1, y + cell - 1),
                      (180, 180, 180), 1)
        # label in a small dark band at top-left of the cell
        (tw, th), _ = cv2.getTextSize(lab, font, fs, thick)
        cv2.rectangle(canvas, (x, y), (x + tw + 6, y + th + 6), (0, 0, 0), -1)
        cv2.putText(canvas, lab, (x + 3, y + th + 3), font, fs,
                    (255, 255, 255), thick, cv2.LINE_AA)
    return canvas


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject", required=True, help="D-NeRF category, e.g. trex")
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--data-root", default="data")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Cap total frames (evenly subsampled). 0 = all.")
    p.add_argument("--cols", type=int, default=20,
                   help="Frames per row in the filmstrip.")
    p.add_argument("--cell", type=int, default=120, help="Per-frame pixel size")
    p.add_argument("--out", default=None, help="Output PNG path")
    args = p.parse_args()

    data_root = Path(args.data_root)
    frames = load_frames(data_root, args.subject, args.split)
    tidx = time_index_map(frames)

    # Optionally subsample frames evenly across the sequence.
    if args.max_frames and len(frames) > args.max_frames:
        sel = np.linspace(0, len(frames) - 1, args.max_frames, dtype=int)
        frames = [frames[i] for i in sel]
        tidx = [tidx[i] for i in sel]

    cells = [load_image(data_root, args.subject, fr) for fr in frames]
    labels = [f"t{t}" for t in tidx]

    grid = make_filmstrip(cells, labels, cell=args.cell, cols=args.cols,
                         pad=4, margin=10)

    out = Path(args.out) if args.out else \
        Path(f"{args.subject}_{args.split}_sample.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid)
    print(f"Saved {out}  ({grid.shape[1]}x{grid.shape[0]})  "
          f"frames={len(cells)} cols={args.cols}")


if __name__ == "__main__":
    main()
