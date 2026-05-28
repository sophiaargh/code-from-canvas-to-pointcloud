# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
BlendedMVS dataset reader for the *raw* (non-WAI) directory layout:

    {ROOT}/
      scene_0/
        blended_images/{frame:08d}.jpg
        rendered_depth_maps/{frame:08d}.pfm
        cams/{frame:08d}_cam.txt

No scene_meta.json or covisibility maps are required.
"""

import os

import cv2
import numpy as np
import PIL.Image

from mapanything.datasets.base.base_dataset import BaseDataset


# ---------------------------------------------------------------------------
# Low-level file readers (same format as eval_pipeline/evaluator.py)
# ---------------------------------------------------------------------------

def _read_pfm(filepath: str) -> np.ndarray:
    with open(filepath, "rb") as f:
        header = f.readline().decode("latin-1").strip()
        assert header in ("PF", "Pf"), f"Not a PFM file: {header}"
        W, H = map(int, f.readline().decode("latin-1").strip().split())
        scale = float(f.readline().decode("latin-1").strip())
        endian = "<" if scale < 0 else ">"
        data = np.frombuffer(f.read(), dtype=np.dtype(endian + "f"))
    return data.reshape((H, W))[::-1].copy()


def _read_cam(filepath: str):
    """Return (K, E_w2c) — both float64 numpy arrays.

    Camera file layout::

        extrinsic
        r00 r01 r02 t0
        r10 r11 r12 t1
        r20 r21 r22 t2
        0   0   0   1

        intrinsic
        fx  0  cx
        0  fy  cy
        0   0   1

        depth_min interval ...
    """
    with open(filepath) as f:
        lines = [l.strip() for l in f if l.strip()]
    E = np.array([list(map(float, lines[i].split())) for i in range(1, 5)])
    K = np.array([list(map(float, lines[i].split())) for i in range(6, 9)])
    return K.astype(np.float32), E.astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BlendedMVSRaw(BaseDataset):
    """BlendedMVS dataset reading directly from raw JPG/PFM/cam.txt files.

    Args:
        ROOT: Root directory containing per-scene subdirectories.
        split: ``"train"`` or ``"val"``.  Scenes are split 90/10 by index.
        All remaining args are forwarded to :class:`BaseDataset`.
    """

    def __init__(self, *args, ROOT: str, split: str, **kwargs):
        super().__init__(*args, split=split, **kwargs)
        self.ROOT = ROOT
        self.split = split
        self._load_data()
        self.is_metric_scale = False
        self.is_synthetic = False

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self):
        all_scenes = sorted(
            d for d in os.listdir(self.ROOT)
            if os.path.isdir(os.path.join(self.ROOT, d))
        )
        if self.split == "train":
            self.scenes = [s for i, s in enumerate(all_scenes) if i % 10 != 0]
        else:
            self.scenes = [s for i, s in enumerate(all_scenes) if i % 10 == 0]
        self.num_of_scenes = len(self.scenes)

    # ------------------------------------------------------------------
    # View loading
    # ------------------------------------------------------------------

    def _get_views(self, sampled_idx, num_views_to_sample, resolution):
        scene_name = self.scenes[sampled_idx]
        scene_root = os.path.join(self.ROOT, scene_name)

        img_dir = os.path.join(scene_root, "blended_images")
        frame_names = sorted(
            f[:-4] for f in os.listdir(img_dir)
            if f.endswith(".jpg") and not f.endswith("_masked.jpg")
        )
        num_frames = len(frame_names)

        frame_indices = self._rng.choice(
            num_frames, size=num_views_to_sample, replace=(num_frames < num_views_to_sample)
        )

        views = []
        for fi in frame_indices:
            frame_name = frame_names[fi]

            # --- image ---
            img_path = os.path.join(img_dir, f"{frame_name}.jpg")
            image = np.array(PIL.Image.open(img_path).convert("RGB"))  # uint8 HxWx3

            # --- depth (z-depth in metres) ---
            depth_path = os.path.join(scene_root, "rendered_depth_maps", f"{frame_name}.pfm")
            depthmap = _read_pfm(depth_path).astype(np.float32)
            depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0)
            img_h, img_w = image.shape[:2]
            if depthmap.shape[0] != img_h or depthmap.shape[1] != img_w:
                depthmap = cv2.resize(depthmap, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

            # --- camera: intrinsics K and w2c extrinsic E ---
            cam_path = os.path.join(scene_root, "cams", f"{frame_name}_cam.txt")
            K, E_w2c = _read_cam(cam_path)
            c2w = np.linalg.inv(E_w2c).astype(np.float32)

            # --- validity mask from positive depth pixels ---
            non_ambiguous_mask = (depthmap > 0).astype(np.int32)

            # --- crop / resize to training resolution ---
            additional = [non_ambiguous_mask]
            image, depthmap, K, additional = self._crop_resize_if_necessary(
                image=image,
                resolution=resolution,
                depthmap=depthmap,
                intrinsics=K,
                additional_quantities=additional,
            )
            non_ambiguous_mask = additional[0]

            views.append(dict(
                img=image,
                depthmap=depthmap,
                camera_pose=c2w,
                camera_intrinsics=K,
                non_ambiguous_mask=non_ambiguous_mask,
                dataset="BlendedMVS",
                label=scene_name,
                instance=os.path.join("images", frame_name),
            ))

        return views
