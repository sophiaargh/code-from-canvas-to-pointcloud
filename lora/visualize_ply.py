"""
Visualize point clouds exported by export_pointclouds.py using Open3D GUI.

Shows one point cloud at a time with Prev/Next navigation.
Order: for each scene → baseline, LoRA, LoRA+consistency.

Setup (one-time):
  pip install open3d

Transfer PLY files from cluster (run locally):
  scp -r qsandoz@izar.epfl.ch:/home/qsandoz/visual-intelligence/ply_exports ./ply_exports

Optionally transfer metrics CSV for quality labels:
  scp qsandoz@izar.epfl.ch:/home/qsandoz/visual-intelligence/evaluation_results/mixed_lora_gray.csv ./mixed_lora_gray.csv

Usage:
  python visualize_ply.py                                      # all default scenes
  python visualize_ply.py --scene scene_15                     # single scene
  python visualize_ply.py --scene scene_15 scene_33 scene_51
  python visualize_ply.py --ply_dir ~/downloads/ply_exports
  python visualize_ply.py --results_csv ./mixed_lora_gray.csv  # show quality metrics
"""

import argparse
import csv
import os
import sys

import numpy as np

try:
    import open3d as o3d
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering
except ImportError:
    sys.exit("open3d is not installed. Run: pip install open3d")

CONDITIONS = ["mixed_baseline", "mixed_lora", "mixed_lora_const"]
CONDITION_LABELS = {
    "mixed_baseline":   "Baseline (4 styled + 4 original, grayscale)",
    "mixed_lora":       "LoRA final (4 styled + 4 original, grayscale)",
    "mixed_lora_const": "LoRA + consistency loss (step 2500, grayscale)",
}

DEFAULT_SCENES = [
    "scene_15", "scene_33", "scene_51", "scene_63", "scene_100", "scene_22",
    "scene_1",  "scene_27", "scene_16", "scene_26", "scene_40", "scene_23",
    "scene_36", "scene_38", "scene_0",  "scene_13",
]

_TIER_HIGH = 0.15
_TIER_MID  = 0.05


def load_metrics(csv_path):
    if not csv_path or not os.path.isfile(csv_path):
        return {}
    metrics = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["scene"].startswith("scene_"):
                metrics[row["scene"]] = {
                    k: float(v) for k, v in row.items()
                    if k not in ("scene", "baseline")
                }
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
                   help="Restrict to a single condition")
    return p.parse_args()


def load_pcd(path):
    if not os.path.isfile(path):
        return None
    pcd = o3d.io.read_point_cloud(path)
    if not pcd.has_points():
        print(f"  [warn] empty point cloud: {path}")
        return None
    return pcd


def build_items(scenes, conditions, ply_dir, metrics):
    items = []
    for scene in scenes:
        for cond in conditions:
            path = os.path.join(ply_dir, cond, f"{scene}.ply")
            pcd = load_pcd(path)
            if pcd is None:
                print(f"  [missing] {path}")
                continue
            m = metrics.get(scene)
            if m:
                tier = quality_tier(m["fscore"])
                info = (f"[{tier}]  fscore={m['fscore']:.3f}  "
                        f"AbsRel={m['AbsRel']:.3f}  chamfer={m['chamfer']:.3f}")
            else:
                info = ""
            n_pts = len(np.asarray(pcd.points))
            items.append((pcd, scene, cond, info, n_pts))
    return items


class Viewer:
    PANEL_WIDTH_EM = 22

    def __init__(self, items):
        self.items = items
        self.current = 0

        app = gui.Application.instance
        self.window = app.create_window("Point Cloud Viewer", 1500, 900)
        w = self.window
        em = w.theme.font_size

        # --- Scene widget ---
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)
        self._scene.scene.set_background([0.12, 0.12, 0.12, 1.0])

        self._mat = rendering.MaterialRecord()
        self._mat.shader = "defaultUnlit"
        self._mat.point_size = 2.5

        # --- Control panel ---
        margin = gui.Margins(em, em, em, em)
        self._panel = gui.Vert(int(0.5 * em), margin)

        # Info labels
        self._lbl_progress  = gui.Label("")
        self._lbl_scene     = gui.Label("")
        self._lbl_condition = gui.Label("")
        self._lbl_info      = gui.Label("")
        self._lbl_npts      = gui.Label("")
        for lbl in (self._lbl_progress, self._lbl_scene,
                    self._lbl_condition, self._lbl_info, self._lbl_npts):
            self._panel.add_child(lbl)

        self._panel.add_fixed(int(em))

        # Navigation buttons
        nav = gui.Horiz(int(0.5 * em))
        btn_prev = gui.Button("← Prev")
        btn_prev.horizontal_padding_em = 1.0
        btn_prev.set_on_clicked(self._on_prev)
        btn_next = gui.Button("Next →")
        btn_next.horizontal_padding_em = 1.0
        btn_next.set_on_clicked(self._on_next)
        nav.add_child(btn_prev)
        nav.add_stretch()
        nav.add_child(btn_next)
        self._panel.add_child(nav)

        w.add_child(self._scene)
        w.add_child(self._panel)
        w.set_on_layout(self._on_layout)

        self._load_current()

    def _on_layout(self, ctx):
        r = self.window.content_rect
        em = self.window.theme.font_size
        panel_w = int(self.PANEL_WIDTH_EM * em)
        self._scene.frame = gui.Rect(r.x, r.y, r.width - panel_w, r.height)
        self._panel.frame = gui.Rect(r.get_right() - panel_w, r.y, panel_w, r.height)

    def _load_current(self):
        pcd, scene_name, cond, info, n_pts = self.items[self.current]
        total = len(self.items)
        cond_idx = CONDITIONS.index(cond) + 1 if cond in CONDITIONS else "?"

        self._scene.scene.clear_geometry()
        self._scene.scene.add_geometry("pcd", pcd, self._mat)

        self._lbl_progress.text  = f"{self.current + 1} / {total}"
        self._lbl_scene.text     = f"Scene: {scene_name}"
        self._lbl_condition.text = f"[{cond_idx}/3] {CONDITION_LABELS.get(cond, cond)}"
        self._lbl_info.text      = info
        self._lbl_npts.text      = f"{n_pts:,} points"

        bounds = pcd.get_axis_aligned_bounding_box()
        self._scene.setup_camera(60, bounds, bounds.get_center())

    def _on_next(self):
        if self.current < len(self.items) - 1:
            self.current += 1
            self._load_current()

    def _on_prev(self):
        if self.current > 0:
            self.current -= 1
            self._load_current()


def main():
    args = get_args()
    scenes     = args.scene or DEFAULT_SCENES
    conditions = [args.condition] if args.condition else CONDITIONS
    metrics    = load_metrics(args.results_csv)

    if args.results_csv and not metrics:
        print(f"[warn] Could not load metrics from {args.results_csv}")

    items = build_items(scenes, conditions, args.ply_dir, metrics)
    if not items:
        sys.exit("No PLY files found. Check --ply_dir and that export_pointclouds.py has been run.")

    print(f"Loaded {len(items)} point clouds ({len(scenes)} scenes × {len(conditions)} conditions)")

    app = gui.Application.instance
    app.initialize()
    Viewer(items)
    app.run()


if __name__ == "__main__":
    main()
