# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
BlendedMVS dataset variant that replaces the RGB images with TeleStyle-rendered
versions while keeping the original depth maps and camera parameters unchanged.

Expected TeleStyle output layout:
    {styled_root}/{style_name}/{scene_name}/blended_images/{frame_name}_result.png

Only frames that were processed by TeleStyle are replaced; for the remaining
frames the dataset falls back to the original photograph silently.
"""

import os

import cv2
import numpy as np
import PIL.Image

from lora.datasets.blendedmvs_raw import BlendedMVSRaw


class BlendedMVSStyled(BlendedMVSRaw):
    """BlendedMVS with RGB images swapped for TeleStyle-rendered equivalents.

    Args:
        style_name: Name of the artistic style, e.g. ``"impressionism"``.
            Must match the subdirectory created by TeleStyle (without extension).
        styled_root: Root directory that contains per-style subdirectories.
            Defaults to ``/scratch/izar/silly/BlendedMVS/telestyle_output``.
        All other keyword args are forwarded to :class:`BlendedMVSRaw`.
    """

    def __init__(
        self,
        *args,
        style_name: str,
        styled_root: str = "/scratch/izar/silly/BlendedMVS/telestyle_output",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.style_name = style_name
        self.styled_root = styled_root

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _styled_image_path(self, scene_name: str, frame_name: str) -> str:
        """Return the expected path for the TeleStyle output of a single frame."""
        return os.path.join(
            self.styled_root,
            self.style_name,
            scene_name,
            "blended_images",
            f"{frame_name}_result.png",
        )

    def _load_styled_image(self, path: str, target_wh: tuple[int, int]) -> PIL.Image.Image | None:
        """Load a styled PNG, resize to *target_wh* (W, H), and return a PIL RGB Image.

        Returns None if the file does not exist or cannot be decoded.
        """
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

    # ------------------------------------------------------------------
    # Override _get_views to swap the RGB image with the styled version
    # ------------------------------------------------------------------

    def _get_views(self, sampled_idx, num_views_to_sample, resolution):
        views = super()._get_views(sampled_idx, num_views_to_sample, resolution)

        for view in views:
            scene_name = view["label"]
            # "instance" is set to os.path.join("images", frame_name) by the parent
            frame_name = os.path.basename(view["instance"])
            styled_path = self._styled_image_path(scene_name, frame_name)
            # view["img"] is a PIL Image at this point (returned by _crop_resize_if_necessary)
            target_wh = view["img"].size  # (W, H) — PIL convention
            styled_img = self._load_styled_image(styled_path, target_wh)
            if styled_img is not None:
                view["img"] = styled_img
                view["dataset"] = f"BlendedMVS-{self.style_name}"

        return views
