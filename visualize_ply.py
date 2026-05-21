"""
Visualize and compare point clouds exported by export_pointclouds.py using Open3D.

Shows three conditions side-by-side for each scene:
  left: photographs (upper bound)  |  middle: baseline + styled  |  right: LoRA + styled

Setup (one-time):
  pip install open3d

Transfer PLY files from cluster (run locally):
  scp -r qsandoz@izar.epfl.ch:/home/qsandoz/visual-intelligence/ply_exports ./ply_exports

Optionally transfer metrics CSV for quality labels:
  scp qsandoz@izar.epfl.ch:/home/qsandoz/visual-intelligence/evaluation_results/mixed_lora_gray.csv ./mixed_lora_gray.csv

Usage:
  python visualize_ply.py                                         # all default scenes
  python visualize_ply.py --scene scene_15                        # single scene
  python visualize_ply.py --scene scene_15 scene_33 scene_51
  python visualize_ply.py --ply_dir ~/downloads/ply_exports
  python visualize_ply.py --results_csv ./mixed_lora_gray.csv     # show quality metrics

Navigation: press [q] in the Open3D window to close and advance to the next scene.
"""

import argparse
import copy
import csv
import os
import sys

import numpy as np

try:
    import open3d as o3d
except ImportError:
    sys.exit("open3d is not installed. Run: pip install open3d")

CONDITIONS = ["photographs", "mixed_baseline", "mixed_lora"]
CONDITION_LABELS = {
    "photographs":    "photos (upper bound)",
    "mixed_baseline": "baseline (4 styled + 4 original, grayscale)",
    "mixed_lora":     "LoRA (4 styled + 4 original, grayscale)",
}

DEFAULT_SCENES = [
    # high f-score (best reconstructions)
    "scene_15", "scene_33", "scene_51", "scene_63", "scene_100", "scene_22",
    # mid-range
    "scene_1",  "scene_27", "scene_16", "scene_26", "scene_40", "scene_23",
    # low f-score (hard cases)
    "scene_36", "scene_38", "scene_0",  "scene_13",
]

# Thresholds for quality tier labels (based on LoRA f-score distribution)
_TIER_HIGH   = 0.15   # top performers
_TIER_MID    = 0.05   # mid-range


def load_metrics(csv_path):
    """Load per-scene LoRA metrics from CSV. Returns dict scene -> {metric: float}."""
    if not csv_path or not os.path.isfile(csv_path):
        return {}
    metrics = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["scene"].startswith("scene_"):
                metrics[row["scene"]] = {k: float(v) for k, v in row.items() if k != "scene" and k != "baseline"}
    return metrics


def quality_tier(fscore):
    if fscore >= _TIER_HIGH:
        return "good"
    if fscore >= _TIER_MID:
        return "mid"
    return "hard"


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ply_dir", default="ply_exports")
    p.add_argument("--results_csv", default=None,
                   help="Optional path to mixed_lora_gray.csv for quality tier display")
    p.add_argument("--scene", nargs="+", default=None,
                   help="Scene(s) to visualize (default: all default scenes)")
    p.add_argument("--condition", default=None, choices=CONDITIONS,
                   help="Show all scenes for a single condition instead of comparing")
    return p.parse_args()


def load_pcd(path):
    if not os.path.isfile(path):
        return None
    pcd = o3d.io.read_point_cloud(path)
    if not pcd.has_points():
        print(f"  [warn] empty point cloud: {path}")
        return None
    return pcd


def visualize_scene_compare(scene_name, ply_dir, metrics):
    pcds = []
    labels = []
    for cond in CONDITIONS:
        path = os.path.join(ply_dir, cond, f"{scene_name}.ply")
        pcd = load_pcd(path)
        if pcd is None:
            print(f"  [missing] {path}")
        pcds.append(pcd)
        labels.append(CONDITION_LABELS[cond])

    available = [(pcd, label) for pcd, label in zip(pcds, labels) if pcd is not None]
    if not available:
        print(f"No PLY files found for {scene_name}, skipping.")
        return

    m = metrics.get(scene_name)
    if m:
        tier   = quality_tier(m["fscore"])
        mstr   = (f"fscore={m['fscore']:.3f}  AbsRel={m['AbsRel']:.3f}  "
                  f"chamfer={m['chamfer']:.3f}")
        header = f"{scene_name}  [{tier}]  {mstr}"
    else:
        tier   = "?"
        header = scene_name

    # Offset each cloud along X so they appear side-by-side
    bbox = available[0][0].get_axis_aligned_bounding_box()
    offset = (bbox.max_bound[0] - bbox.min_bound[0]) * 1.3

    to_draw = []
    print(f"\n{header}")
    for i, (pcd, label) in enumerate(available):
        shifted = copy.deepcopy(pcd)
        if i > 0:
            shifted.translate([i * offset, 0, 0])
        to_draw.append(shifted)
        n = np.asarray(pcd.points).shape[0]
        print(f"  [{i}] {label:45s}  ({n} pts)")

    title = f"{scene_name} [{tier}]  {mstr if m else ''}  |  " + "  |  ".join(
        f"[{i}] {label}" for i, (_, label) in enumerate(available)
    )
    o3d.visualization.draw_geometries(
        to_draw,
        window_name=title,
        width=1800,
        height=800,
        point_show_normal=False,
    )


def visualize_condition_scenes(condition, scenes, ply_dir, metrics):
    label = CONDITION_LABELS[condition]
    for scene_name in scenes:
        path = os.path.join(ply_dir, condition, f"{scene_name}.ply")
        pcd = load_pcd(path)
        if pcd is None:
            print(f"  [missing] {path}")
            continue
        m = metrics.get(scene_name)
        n = np.asarray(pcd.points).shape[0]
        if m:
            tier = quality_tier(m["fscore"])
            mstr = f"fscore={m['fscore']:.3f}  AbsRel={m['AbsRel']:.3f}"
            print(f"{scene_name} [{tier}]  {mstr}  ({n} pts)")
            title = f"{scene_name} [{tier}]  {mstr} — {label}"
        else:
            print(f"{scene_name}  ({n} pts)")
            title = f"{scene_name} — {label}"
        o3d.visualization.draw_geometries(
            [pcd],
            window_name=title,
            width=1200,
            height=800,
        )


def main():
    args = get_args()
    ply_dir = args.ply_dir
    scenes  = args.scene or DEFAULT_SCENES
    metrics = load_metrics(args.results_csv)
    if args.results_csv and not metrics:
        print(f"[warn] Could not load metrics from {args.results_csv} — quality tiers will not be shown")

    if args.condition:
        visualize_condition_scenes(args.condition, scenes, ply_dir, metrics)
    else:
        for scene in scenes:
            visualize_scene_compare(scene, ply_dir, metrics)


if __name__ == "__main__":
    main()
