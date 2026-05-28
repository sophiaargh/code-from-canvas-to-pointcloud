"""
Visualize the three Notre-Dame point clouds side-by-side using Open3D.

Default mode: two consecutive windows
  Window 1 — texture colors:   base | LoRA | LoRA consistency
  Window 2 — view colors:      base | LoRA | LoRA consistency

Setup (one-time):
  pip install open3d

Transfer PLY files from cluster (run locally):
  scp -r qsandoz@izar.epfl.ch:/home/qsandoz/visual-intelligence/notre_dame_example/*.ply ./notre_dame_example/

Usage:
  python visualize_notre_dame.py                        # both windows (texture then view-colored)
  python visualize_notre_dame.py --only texture         # texture window only
  python visualize_notre_dame.py --only viewcolors      # view-colored window only
  python visualize_notre_dame.py --single base          # single cloud
  python visualize_notre_dame.py --dir /path/to/plys    # custom folder

Navigation: press [q] to close a window and open the next one.
"""

import argparse
import copy
import os
import sys

import numpy as np

try:
    import open3d as o3d
except ImportError:
    sys.exit("open3d is not installed. Run: pip install open3d")

TEXTURE_CLOUDS = [
    ("base",             "notre_dame_base.ply",             "Base model (no LoRA)"),
    ("lora",             "notre_dame_lora.ply",             "LoRA (mixed styles, grayscale)"),
    ("lora_consistency", "notre_dame_lora_consistency.ply", "LoRA consistency (step 2500)"),
]

VIEWCOLOR_CLOUDS = [
    ("base_viewcolors",             "notre_dame_base_viewcolors.ply",             "Base model — colored by view"),
    ("lora_viewcolors",             "notre_dame_lora_viewcolors.ply",             "LoRA — colored by view"),
    ("lora_consistency_viewcolors", "notre_dame_lora_consistency_viewcolors.ply", "LoRA consistency — colored by view"),
]

ALL_CLOUDS = TEXTURE_CLOUDS + VIEWCOLOR_CLOUDS


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default=os.path.dirname(os.path.abspath(__file__)),
                   help="Folder containing the PLY files (default: this script's directory)")
    p.add_argument("--single", choices=[key for key, _, _ in ALL_CLOUDS], default=None,
                   help="Show only one point cloud")
    p.add_argument("--only", choices=["texture", "viewcolors"], default=None,
                   help="Show only one of the two comparison windows")
    return p.parse_args()


def load_pcd(path):
    if not os.path.isfile(path):
        print(f"  [missing] {path}")
        return None
    pcd = o3d.io.read_point_cloud(path)
    if not pcd.has_points():
        print(f"  [empty]   {path}")
        return None
    return pcd


def show_comparison(cloud_defs, ply_dir, title_prefix):
    loaded = [(load_pcd(os.path.join(ply_dir, fn)), label) for _, fn, label in cloud_defs]
    available = [(pcd, label) for pcd, label in loaded if pcd is not None]
    if not available:
        print(f"  [skip] no PLY files found for '{title_prefix}'")
        return

    bbox = available[0][0].get_axis_aligned_bounding_box()
    offset = (bbox.max_bound[0] - bbox.min_bound[0]) * 1.3

    to_draw = []
    print(f"\n{title_prefix}:")
    for i, (pcd, label) in enumerate(available):
        shifted = copy.deepcopy(pcd)
        if i > 0:
            shifted.translate([i * offset, 0, 0])
        to_draw.append(shifted)
        n = np.asarray(pcd.points).shape[0]
        print(f"  [{i}] {label:50s}  ({n:,} pts)")

    title = f"{title_prefix}  |  " + "  |  ".join(
        f"[{i}] {label}" for i, (_, label) in enumerate(available)
    )
    o3d.visualization.draw_geometries(to_draw, window_name=title, width=1800, height=800)


def main():
    args = get_args()

    if args.single:
        key, filename, label = next(t for t in ALL_CLOUDS if t[0] == args.single)
        pcd = load_pcd(os.path.join(args.dir, filename))
        if pcd is None:
            sys.exit(1)
        n = np.asarray(pcd.points).shape[0]
        print(f"{label}  ({n:,} pts)")
        o3d.visualization.draw_geometries([pcd], window_name=label, width=1200, height=800)
        return

    if args.only != "viewcolors":
        show_comparison(TEXTURE_CLOUDS, args.dir, "Notre-Dame — texture colors")

    if args.only != "texture":
        show_comparison(VIEWCOLOR_CLOUDS, args.dir, "Notre-Dame — colored by view")


if __name__ == "__main__":
    main()
