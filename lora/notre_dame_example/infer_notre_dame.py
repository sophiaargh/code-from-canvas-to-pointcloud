"""
Run MapAnything inference on the images in this folder and export a PLY point cloud.

Usage:
    python notre_dame_example/infer_notre_dame.py
    python notre_dame_example/infer_notre_dame.py --lora_path /path/to/lora/checkpoint
    python notre_dame_example/infer_notre_dame.py --grayscale
"""

import argparse
import os

import matplotlib.cm as cm
import numpy as np
import torch
import trimesh

from lora.eval.models import get_model, infer, load_with_lora
from mapanything.utils.image import load_images

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", default=SCRIPT_DIR,
                   help="Folder containing input images (default: this script's directory)")
    p.add_argument("--out_path", default=os.path.join(SCRIPT_DIR, "notre_dame.ply"),
                   help="Output PLY file path")
    p.add_argument("--lora_path", default=None,
                   help="Path to LoRA checkpoint directory (omit to use base model)")
    p.add_argument("--checkpoint", default="facebook/map-anything")
    p.add_argument("--n_points", type=int, default=200_000,
                   help="Randomly subsample to this many points (0 = keep all)")
    p.add_argument("--color_by_view", action="store_true",
                   help="Color each view with a unique HSV color instead of the image texture")
    p.add_argument("--grayscale", action="store_true",
                   help="Convert images to grayscale-RGB before inference")
    return p.parse_args()


def main():
    args = get_args()

    image_paths = sorted([
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ])

    if not image_paths:
        raise RuntimeError(f"No images found in {args.input_dir}")

    print(f"Found {len(image_paths)} images:")
    for p in image_paths:
        print(f"  {os.path.basename(p)}")

    if args.lora_path:
        print(f"Loading model with LoRA from {args.lora_path} ...")
        model, device = load_with_lora(args.lora_path, base_checkpoint=args.checkpoint)
    else:
        print(f"Loading base model ({args.checkpoint}) ...")
        model, device = get_model(args.checkpoint)

    views = load_images(image_paths, resolution_set=518, norm_type="dinov2", patch_size=14,
                        grayscale=args.grayscale)

    print("Running inference ...")
    with torch.no_grad():
        predictions = infer(model, views)

    colormap = cm.get_cmap("hsv")
    pts_list, colors_list = [], []
    for view_idx, pred in enumerate(predictions):
        pts = pred["pts3d"].squeeze(0).reshape(-1, 3).cpu().numpy()
        img = pred["img_no_norm"].squeeze(0).reshape(-1, 3).cpu().numpy()

        mask = pred.get("mask")
        valid = mask.squeeze(0).reshape(-1).cpu().numpy().astype(bool) if mask is not None \
            else np.ones(len(pts), dtype=bool)

        conf = pred.get("conf")
        if conf is not None:
            c = conf.squeeze(0).reshape(-1).cpu().numpy()
            valid = valid & (c > np.percentile(c, 50))

        valid = valid & np.isfinite(pts).all(axis=1)

        if args.color_by_view:
            view_color = np.array(colormap(view_idx / len(predictions))[:3])
            colors = (np.full((valid.sum(), 3), view_color) * 255).astype(np.uint8)
            print(f"  View {view_idx}: {valid.sum():,} points, color RGB{tuple(np.round(view_color, 2))}")
        else:
            colors = (img[valid] * 255).astype(np.uint8)

        pts_list.append(pts[valid])
        colors_list.append(colors)

    all_pts = np.concatenate(pts_list)
    all_colors = np.concatenate(colors_list)

    n = args.n_points
    if n > 0 and len(all_pts) > n:
        total = len(all_pts)
        rng = np.random.default_rng(0)
        idx = rng.choice(total, size=n, replace=False)
        all_pts = all_pts[idx]
        all_colors = all_colors[idx]
        print(f"Subsampled {total:,} → {n:,} points")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)
    trimesh.PointCloud(all_pts, colors=all_colors).export(args.out_path)
    print(f"Saved {len(all_pts):,} points → {args.out_path}")


if __name__ == "__main__":
    main()
