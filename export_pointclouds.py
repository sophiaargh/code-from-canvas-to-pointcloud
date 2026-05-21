"""
Export point cloud PLY files from MapAnything inference for visual quality assessment.

Runs inference for a given condition on a set of scenes and saves one PLY per scene
to ply_exports/{condition}/{scene_name}.ply, ready for Open3D visualization.

Usage:
    python export_pointclouds.py --condition photographs
    python export_pointclouds.py --condition mixed_baseline --styled_root /scratch/...
    python export_pointclouds.py --condition mixed_lora --lora_path /scratch/.../final
"""

import argparse
import hashlib
import os
import re

import numpy as np
import torch
import trimesh

from eval_pipeline.models import get_model, infer, load_with_lora
from mapanything.utils.colmap_export import voxel_downsample_point_cloud
from mapanything.utils.image import load_images

DEFAULT_SCENES = [
    # high f-score (best reconstructions)
    "scene_15", "scene_33", "scene_51", "scene_63", "scene_100", "scene_22",
    # mid-range
    "scene_1",  "scene_27", "scene_16", "scene_26",  "scene_40", "scene_23",
    # low f-score (hard cases)
    "scene_36", "scene_38", "scene_0",  "scene_13",
]

_ALL_STYLES = ["engraving", "impressionism", "oil_painting", "watercolor"]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True,
                   choices=["photographs", "mixed_baseline", "mixed_lora"])
    p.add_argument("--data_dir", default="/scratch/izar/silly/BlendedMVS/renamed")
    p.add_argument("--styled_root", default="/scratch/izar/silly/BlendedMVS/telestyle_output")
    p.add_argument("--lora_path",
                   default="/scratch/izar/silly/lora_checkpoints/mixed_styles_gray/final")
    p.add_argument("--style_names", nargs="+", default=_ALL_STYLES)
    p.add_argument("--n_styled", type=int, default=4)
    p.add_argument("--scenes", nargs="+", default=None)
    p.add_argument("--out_dir", default="ply_exports")
    p.add_argument("--voxel_fraction", type=float, default=0.01)
    p.add_argument("--checkpoint", default="facebook/map-anything")
    p.add_argument("--grayscale", action="store_true",
                   help="Convert all images to grayscale-RGB before model inference")
    return p.parse_args()


def _select_views(scene_dir):
    """Replicate eval_pipeline/evaluator.py view-selection logic exactly."""
    blended_dir = os.path.join(scene_dir, "blended_images")
    depth_dir = os.path.join(scene_dir, "rendered_depth_maps")
    if not os.path.isdir(blended_dir) or not os.path.isdir(depth_dir):
        return []
    available = []
    for fn in os.listdir(blended_dir):
        m = re.match(r"(\d{8})\.jpg$", fn)
        if not m:
            continue
        vid = int(m.group(1))
        if os.path.exists(os.path.join(depth_dir, f"{vid:08d}.pfm")):
            available.append(vid)
    available = sorted(set(available))
    stride = max(1, len(available) // 8)
    selected = available[::stride][:8]
    return available, selected


def _get_available_styles(vid, scene_name, styled_root, style_names):
    """Return styles that have a rendered counterpart for this (scene, vid)."""
    return [
        style for style in style_names
        if os.path.isfile(os.path.join(
            styled_root, style, scene_name, "blended_images", f"{vid:08d}_result.png"
        ))
    ]


def get_image_paths(scene_dir, scene_name, condition, styled_root, style_names, n_styled):
    available, selected = _select_views(scene_dir)
    if not selected:
        print(f"  [skip] {scene_name}: no views with depth found")
        return None

    blended_dir = os.path.join(scene_dir, "blended_images")

    if condition == "photographs":
        return [os.path.join(blended_dir, f"{v:08d}.jpg") for v in selected]

    # Mixed conditions: n_styled styled views from random styles + rest original.
    # Mirrors evaluator.py logic exactly (same deterministic seed).
    vid_to_styles = {v: _get_available_styles(v, scene_name, styled_root, style_names)
                     for v in selected}
    styled_candidates = [v for v, styles in vid_to_styles.items() if styles]

    # Fallback: scan all available frames if not enough candidates in selected.
    if len(styled_candidates) < n_styled:
        for v in available:
            if v in vid_to_styles:
                continue
            styles = _get_available_styles(v, scene_name, styled_root, style_names)
            if styles:
                styled_candidates.append(v)
                vid_to_styles[v] = styles
                non_styled = [x for x in reversed(selected) if x not in styled_candidates]
                if non_styled:
                    selected.remove(non_styled[0])
                selected.append(v)
            if len(styled_candidates) >= n_styled:
                break

    if not styled_candidates:
        print(f"  [skip] {scene_name}: no styled frames found")
        return None

    n = min(n_styled, len(styled_candidates))
    chosen = styled_candidates[:n]
    scene_seed = int(hashlib.md5(scene_name.encode()).hexdigest(), 16) % (2 ** 32)
    rng = np.random.default_rng(scene_seed)
    chosen_style = {v: rng.choice(vid_to_styles[v]) for v in chosen}

    paths = []
    for v in selected:
        if v in chosen_style:
            paths.append(os.path.join(
                styled_root, chosen_style[v], scene_name,
                "blended_images", f"{v:08d}_result.png",
            ))
        else:
            paths.append(os.path.join(blended_dir, f"{v:08d}.jpg"))
    style_summary = ", ".join(f"{v:08d}:{chosen_style[v]}" for v in chosen)
    print(f"  mixed: {n} styled ({style_summary}) + {len(selected)-n} original")
    return paths


def export_scene(model, image_paths, out_path, voxel_fraction, grayscale=False):
    views = load_images(image_paths, resolution_set=518, norm_type="dinov2", patch_size=14,
                        grayscale=grayscale)
    with torch.no_grad():
        predictions = infer(model, views)

    pts_list, colors_list = [], []
    for pred in predictions:
        pts = pred["pts3d"].squeeze(0).reshape(-1, 3).cpu().numpy()
        img = pred["img_no_norm"].squeeze(0).reshape(-1, 3).cpu().numpy()
        colors = (img * 255).astype(np.uint8)

        mask = pred.get("mask")
        if mask is not None:
            valid = mask.squeeze(0).reshape(-1).cpu().numpy().astype(bool)
        else:
            valid = np.ones(len(pts), dtype=bool)

        conf = pred.get("conf")
        if conf is not None:
            c = conf.squeeze(0).reshape(-1).cpu().numpy()
            valid = valid & (c > np.percentile(c, 50))

        valid = valid & np.isfinite(pts).all(axis=1)
        pts_list.append(pts[valid])
        colors_list.append(colors[valid])

    all_pts = np.concatenate(pts_list)
    all_colors = np.concatenate(colors_list)
    pts_ds, colors_ds = voxel_downsample_point_cloud(all_pts, all_colors, voxel_fraction)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    trimesh.PointCloud(pts_ds, colors=colors_ds).export(out_path)
    print(f"  Saved {len(pts_ds)} points → {out_path}")


def main():
    args = get_args()

    if args.condition == "mixed_lora":
        model, device = load_with_lora(args.lora_path, base_checkpoint=args.checkpoint)
    else:
        model, device = get_model(args.checkpoint)

    scenes = args.scenes or DEFAULT_SCENES
    print(f"Condition: {args.condition}  |  {len(scenes)} scenes")

    for scene_name in scenes:
        scene_dir = os.path.join(args.data_dir, scene_name)
        out_path = os.path.join(args.out_dir, args.condition, f"{scene_name}.ply")
        if os.path.exists(out_path):
            print(f"  [skip] {out_path} already exists")
            continue
        print(f"Exporting {scene_name} ...", flush=True)
        image_paths = get_image_paths(scene_dir, scene_name, args.condition, args.styled_root,
                                      args.style_names, args.n_styled)
        if image_paths is None:
            continue
        export_scene(model, image_paths, out_path, args.voxel_fraction, grayscale=args.grayscale)

    print("Done.")


if __name__ == "__main__":
    main()
