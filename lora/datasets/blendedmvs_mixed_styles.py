# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
BlendedMVS dataset variant that mixes original photos with TeleStyle-rendered images
from multiple artistic styles within each training sample.

For each sample, `n_styled` views are randomly chosen and each is replaced with a
styled image drawn from a randomly selected style in `style_names`. The remaining
views keep the original photograph. Falls back silently to the original if the styled
file is missing.

Expected TeleStyle output layout:
    {styled_root}/{style_name}/{scene_name}/blended_images/{frame_name}_result.png
"""

import os

import cv2
import numpy as np
import PIL.Image

from lora.datasets.blendedmvs_raw import BlendedMVSRaw


class BlendedMVSMixedStyles(BlendedMVSRaw):
    """BlendedMVS with a random mix of original and multi-style TeleStyle images.

    Args:
        style_names: List of artistic style names (e.g. ``["impressionism", "engraving"]``).
            Each must match a subdirectory under ``styled_root``.
        n_styled: Number of views per sample to replace with styled images.
        styled_root: Root directory containing per-style TeleStyle output.
        grayscale: If True, convert all view images to grayscale-RGB before returning.
        All other keyword args are forwarded to :class:`BlendedMVSRaw`.
    """

    def __init__(
        self,
        *args,
        style_names: list,
        n_styled: int = 2,
        styled_root: str = "/scratch/izar/silly/BlendedMVS/telestyle_output",
        grayscale: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.style_names = style_names
        self.n_styled = n_styled
        self.styled_root = styled_root
        self.grayscale = grayscale

    def _styled_image_path(self, style_name: str, scene_name: str, frame_name: str) -> str:
        return os.path.join(
            self.styled_root,
            style_name,
            scene_name,
            "blended_images",
            f"{frame_name}_result.png",
        )

    def _load_styled_image(self, path: str, target_wh: tuple) -> PIL.Image.Image | None:
        if not os.path.isfile(path):
            return None
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        w, h = target_wh
        if img.shape[1] != w or img.shape[0] != h:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        return PIL.Image.fromarray(img)

    def _get_views(self, sampled_idx, num_views_to_sample, resolution):
        views = super()._get_views(sampled_idx, num_views_to_sample, resolution)

        # Save original images as numpy uint8 arrays (PIL Images lack .dtype and fail
        # base_dataset's is_good_type() check). Converted back to PIL in __getitem__.
        for view in views:
            view["img_original_np"] = np.array(view["img"])
            view["is_styled"] = False

        # Randomly pick which view indices get a styled replacement.
        n = min(self.n_styled, len(views))
        styled_indices = self._rng.choice(len(views), size=n, replace=False)

        for i in styled_indices:
            view = views[i]
            scene_name = view["label"]
            frame_name = os.path.basename(view["instance"])
            style = self._rng.choice(self.style_names)
            styled_path = self._styled_image_path(style, scene_name, frame_name)
            target_wh = view["img"].size  # PIL (W, H)
            styled_img = self._load_styled_image(styled_path, target_wh)
            if styled_img is not None:
                view["img"] = styled_img
                view["dataset"] = f"BlendedMVS-{style}"
                view["is_styled"] = True

        if self.grayscale:
            for view in views:
                gray = view["img"].convert("L")
                view["img"] = PIL.Image.merge("RGB", (gray, gray, gray))
                orig_pil = PIL.Image.fromarray(view["img_original_np"])
                gray_orig = orig_pil.convert("L")
                view["img_original_np"] = np.array(PIL.Image.merge("RGB", (gray_orig, gray_orig, gray_orig)))

        return views

    def __getitem__(self, idx):
        views = super().__getitem__(idx)
        # Apply the same image transform to the original PIL images now that
        # the base class has set up self.transform (PIL → normalized tensor).
        for view in views:
            view["img_original"] = self.transform(PIL.Image.fromarray(view.pop("img_original_np")))
        return views
