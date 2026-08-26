"""D-NeRF scene dataset.

Handles the verified D-NeRF schema:
    transform_matrix : camera-to-world (OpenGL/NeRF convention, looks down -Z)
    We convert to OpenCV (3DGS) convention by negating columns 1 and 2 of the
    rotation block: ``R_cv = R_nerf * diag(1, -1, -1)``.
    Intrinsics come from ``camera_angle_x`` (a horizontal FOV):
        fx = 0.5 * W / tan(0.5 * camera_angle_x), fy = fx (square pixels),
        cx = cy = 0.5 * (W - 1).

Two readers:
  * :class:`DNeRFProcessedScene` : raw JSON + PNG via numpy/torch (no Rust dep).
  * :class:`ProcessedDNeRFDataset` : Rust-preprocessed .npy tensors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import Config


# --------------------------------------------------------------------------- #
# Camera convention helpers
# --------------------------------------------------------------------------- #
_NERF_TO_CV = torch.tensor(
    [[1.0, 0.0, 0.0, 0.0],
     [0.0, -1.0, 0.0, 0.0],
     [0.0, 0.0, -1.0, 0.0],
     [0.0, 0.0, 0.0, 1.0]],
    dtype=torch.float32,
)


def nerf_to_cv_c2w(c2w_nerf: torch.Tensor) -> torch.Tensor:
    """Convert a (...,4,4) NeRF c2w to OpenCV/3DGS c2w."""
    return c2w_nerf @ _NERF_TO_CV.to(c2w_nerf)


def intrinsics_from_fov(camera_angle_x: float, w: int, h: int) -> torch.Tensor:
    fx = 0.5 * w / np.tan(0.5 * camera_angle_x)
    fy = fx  # D-NeRF uses square pixels
    cx = 0.5 * (w - 1)
    cy = 0.5 * (h - 1)
    return torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32
    )


# --------------------------------------------------------------------------- #
# Sample types
# --------------------------------------------------------------------------- #
@dataclass
class SceneSample:
    category: str
    time: float            # normalized [0,1]
    time_idx: int          # integer temporal index
    image: torch.Tensor    # (3,H,W) float32 in [0,1]
    c2w: torch.Tensor      # (4,4) OpenCV convention
    intrinsics: torch.Tensor  # (3,3)


@dataclass
class SceneMeta:
    category: str
    root: Path
    camera_angle_x: float
    intrinsics: torch.Tensor   # (3,3)
    train: List[dict]
    val: List[dict]
    test: List[dict]

    def split_frames(self, split: str) -> List[dict]:
        return {"train": self.train, "val": self.val, "test": self.test}[split]


# --------------------------------------------------------------------------- #
# Raw D-NeRF reader (no preprocessing dependency)
# --------------------------------------------------------------------------- #
def list_categories(data_root: Path) -> List[str]:
    cats = []
    for p in sorted(Path(data_root).iterdir()):
        if p.is_dir() and (p / "transforms_train.json").exists():
            cats.append(p.name)
    return cats


def _load_transforms(path: Path) -> Tuple[float, List[dict]]:
    with open(path) as f:
        obj = json.load(f)
    return float(obj["camera_angle_x"]), obj["frames"]


def load_scene_meta(data_root: Path, category: str) -> SceneMeta:
    root = Path(data_root) / category
    cax_t, train = _load_transforms(root / "transforms_train.json")
    cax_v, val = _load_transforms(root / "transforms_val.json")
    cax_e, test = _load_transforms(root / "transforms_test.json")
    assert abs(cax_t - cax_v) < 1e-4 and abs(cax_t - cax_e) < 1e-4, \
        f"camera_angle_x mismatch in {category}"
    return SceneMeta(
        category=category,
        root=root,
        camera_angle_x=cax_t,
        intrinsics=None,  # filled after W/H known
        train=train,
        val=val,
        test=test,
    )


def _time_indices(frames: List[dict]) -> List[int]:
    """Map each frame to an integer temporal index from its `time` value."""
    seen: Dict[float, int] = {}
    idx = []
    for f in frames:
        t = round(float(f["time"]), 6)
        if t not in seen:
            seen[t] = len(seen)
        idx.append(seen[t])
    return idx


class DNeRFDataset(Dataset):
    """Raw D-NeRF reader. One item = one (view, time) frame.

    Set ``time_aligned=True`` to return, for each temporal index ``t``, the
    full set of views at that ``t`` (useful for the autoregressive rollout in
    workflow.md Phase 3 Step 7). The default (False) flattens view*time.
    """

    def __init__(
        self,
        cfg: Config,
        split: str = "train",
        time_aligned: bool = False,
        categories: Optional[List[str]] = None,
    ):
        self.cfg = cfg
        self.split = split
        self.time_aligned = time_aligned
        self.W, self.H = cfg.data.image_wh
        cats = categories or list(cfg.data.categories)
        self.scenes: Dict[str, SceneMeta] = {}
        self.items: List[Tuple[str, int]] = []  # (category, frame_pos)
        self._time_indices: Dict[str, List[int]] = {}
        self._intrinsics: Dict[str, torch.Tensor] = {}
        for c in cats:
            meta = load_scene_meta(cfg.data.data_root, c)
            # Compute intrinsics at the RENDER resolution (render_wh), not the
            # native image_wh. The differentiable rasterizer renders at
            # render_wh to bound O(N*H*W) memory, so the projection must match
            # that resolution or Gaussians project off-screen (rules.md Rule 6).
            rW, rH = cfg.data.render_wh
            meta.intrinsics = intrinsics_from_fov(meta.camera_angle_x, rW, rH)
            self.scenes[c] = meta
            self._intrinsics[c] = meta.intrinsics
            frames = meta.split_frames(split)
            self._time_indices[c] = _time_indices(frames)
            for i in range(len(frames)):
                self.items.append((c, i))
        if time_aligned:
            # Re-index by unique time per scene.
            self._time_groups: Dict[str, Dict[int, List[int]]] = {}
            for c, meta in self.scenes.items():
                groups: Dict[int, List[int]] = {}
                tidx = self._time_indices[c]
                for pos, t in enumerate(tidx):
                    groups.setdefault(t, []).append(pos)
                # sort groups by time index
                self._time_groups[c] = dict(sorted(groups.items()))
            # items become (category, time_idx)
            self.items = [
                (c, t)
                for c in self.scenes
                for t in sorted(self._time_groups[c].keys())
            ]

    def __len__(self) -> int:
        return len(self.items)

    def _read_image(self, meta: SceneMeta, frame: dict) -> torch.Tensor:
        path = meta.root / (frame["file_path"] + ".png")
        img = _read_png(path)
        if img.shape[-1] == 4:
            img = img[..., :3]  # drop alpha
        if img.shape[0] != self.H or img.shape[1] != self.W:
            # D-NeRF frames may already be (H,W); this path keeps PIL-free.
            img = _resize_nn(img, self.H, self.W)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        if self.cfg.data.white_background and img.shape[0] == 4:
            alpha = img[3:4]
            img = img[:3] * alpha + (1.0 - alpha)
        return img[:3]

    def _frame(self, category: str, pos: int) -> dict:
        return self.scenes[category].split_frames(self.split)[pos]

    def __getitem__(self, idx: int) -> SceneSample:
        category, key = self.items[idx]
        meta = self.scenes[category]
        K = self._intrinsics[category]
        if self.time_aligned:
            # Return the first view at this time as the representative sample;
            # the collate function can gather the rest for multi-view supervision.
            pos = self._time_groups[category][key][0]
        else:
            pos = key
        frame = self._frame(category, pos)
        c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
        c2w = nerf_to_cv_c2w(c2w)
        img = self._read_image(meta, frame)
        return SceneSample(
            category=category,
            time=float(frame["time"]),
            time_idx=self._time_indices[category][pos],
            image=img,
            c2w=c2w,
            intrinsics=K,
        )

    def temporal_steps(self, category: str) -> List[int]:
        return sorted(set(self._time_indices[category]))


# --------------------------------------------------------------------------- #
# Rust-processed reader (.npy)
# --------------------------------------------------------------------------- #
class ProcessedDNeRFDataset(Dataset):
    """Consumes ``processed/<cat>/`` produced by the Rust ``preprocess`` binary.
    Requires images to have been dumped with ``--images``.
    """

    def __init__(self, cfg: Config, split: str = "train", categories: Optional[List[str]] = None):
        self.cfg = cfg
        self.split = split
        self.W, self.H = cfg.data.image_wh
        self.items: List[Tuple[str, int]] = []
        self._poses: Dict[str, np.ndarray] = {}
        self._times: Dict[str, np.ndarray] = {}
        self._K: Dict[str, torch.Tensor] = {}
        self._img_dir: Dict[str, Path] = {}
        for c in (categories or list(cfg.data.categories)):
            sp = Path(cfg.data.processed_root) / c / split
            if not sp.exists():
                continue
            poses = np.load(sp / "poses.npy")  # (N,16)
            times = np.load(sp / "times.npy")  # (N,)
            self._poses[c] = poses
            self._times[c] = times
            self._img_dir[c] = sp / "images"
            # Intrinsics: recompute from camera_angle_x at RENDER resolution.
            cax = float(np.load(Path(cfg.data.processed_root) / c / "camera_angle_x.npy"))
            rW, rH = cfg.data.render_wh
            self._K[c] = intrinsics_from_fov(cax, rW, rH)
            for i in range(len(poses)):
                self.items.append((c, i))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> SceneSample:
        category, i = self.items[idx]
        pose = self._poses[category][i].reshape(4, 4)
        c2w = torch.from_numpy(pose).float()
        c2w = nerf_to_cv_c2w(c2w)
        img = np.load(self._img_dir[category] / f"{i:04d}.npy")  # HxWxC u8
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        if img.shape[0] == 4:
            img = img[:3]
        return SceneSample(
            category=category,
            time=float(self._times[category][i]),
            time_idx=int(np.searchsorted(np.unique(self._times[category]), self._times[category][i])),
            image=img,
            c2w=c2w,
            intrinsics=self._K[category],
        )


# --------------------------------------------------------------------------- #
# Tiny image helpers (no hard dep on opencv/torchvision for the loader path)
# --------------------------------------------------------------------------- #
def _read_png(path: Path) -> np.ndarray:
    try:
        from PIL import Image  # type: ignore
        return np.array(Image.open(path).convert("RGBA"))
    except Exception:
        # Fallback to torchvision
        from torchvision.io import read_image  # type: ignore
        t = read_image(str(path))  # C,H,W uint8
        arr = t.permute(1, 2, 0).numpy()
        if arr.shape[-1] == 3:
            arr = np.concatenate([arr, 255 * np.ones((*arr.shape[:2], 1), np.uint8)], -1)
        return arr


def _resize_nn(img: np.ndarray, h: int, w: int) -> np.ndarray:
    ys = np.linspace(0, img.shape[0] - 1, h).astype(int)
    xs = np.linspace(0, img.shape[1] - 1, w).astype(int)
    return img[ys][:, xs]
