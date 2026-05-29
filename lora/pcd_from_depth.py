import argparse
import hashlib
import os
import re

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import cKDTree

from mapanything.utils.image import load_images
from lora.eval.evaluator import read_pfm
from lora.eval.models import get_model, load_with_lora, infer

DEFAULT_STYLES = ["engraving", "impressionism", "oil_painting", "watercolor"]


# ---------------------------------------------------------------------------
# Camera / point-cloud helpers (from evaluator)
# ---------------------------------------------------------------------------

def read_cam(filepath):
    with open(filepath) as f:
        lines = [l.strip() for l in f if l.strip()]
    E = np.array([list(map(float, lines[i].split())) for i in range(1, 5)])
    K = np.array([list(map(float, lines[i].split())) for i in range(6, 9)])
    return K, E


def depth_to_world_pcd(depth, K, E):
    """Unproject a depth map to a world-space point cloud."""
    mask = (depth > 0) & np.isfinite(depth)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = np.arange(depth.shape[1], dtype=np.float32)
    v = np.arange(depth.shape[0], dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    x = (uu - cx) * depth / fx
    y = (vv - cy) * depth / fy
    z = depth
    pts_cam   = np.stack([x, y, z], axis=-1)[mask]
    R, t      = E[:3, :3], E[:3, 3]
    pts_world = (R.T @ (pts_cam - t).T).T
    return pts_world


def icp_align(pred_pts, gt_pts, n_iters=50):
    """Rigid + isotropic-scale ICP to bring pred into GT's coordinate frame."""
    pred_aligned = pred_pts - pred_pts.mean(0) + gt_pts.mean(0)
    pred_scale = np.linalg.norm(pred_aligned.std(0))
    gt_scale   = np.linalg.norm(gt_pts.std(0))
    pred_aligned = pred_aligned * (gt_scale / (pred_scale + 1e-8))
    for _ in range(n_iters):
        _, idx     = cKDTree(gt_pts).query(pred_aligned)
        gt_matched = gt_pts[idx]
        pc = pred_aligned.mean(0);  gc = gt_matched.mean(0)
        H  = (pred_aligned - pc).T @ (gt_matched - gc)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1;  R = Vt.T @ U.T
        t        = gc - R @ pc
        pred_new = (R @ pred_aligned.T).T + t
        if np.abs(pred_new - pred_aligned).max() < 1e-7:
            break
        pred_aligned = pred_new
    return pred_aligned


def compute_pcd_metrics(pts_pred, pts_gt, max_pts=50_000, threshold_pct=0.02):
    """Chamfer distance, precision, recall, F-score after ICP alignment.

    threshold_pct: fraction of GT scene scale used as the acceptance radius
                   (matches the 2 % rule from the evaluator).
    Returns a dict with keys: chamfer, precision, recall, fscore.
    Returns None if either cloud is empty.
    """
    if len(pts_pred) == 0 or len(pts_gt) == 0:
        return None

    def _sub(pts):
        if len(pts) > max_pts:
            return pts[np.random.choice(len(pts), max_pts, replace=False)]
        return pts

    pred_s = _sub(pts_pred)
    gt_s   = _sub(pts_gt)

    pred_aligned = icp_align(pred_s, gt_s)

    scene_scale = np.linalg.norm(gt_s.std(axis=0))
    threshold   = threshold_pct * scene_scale

    tgt  = cKDTree(gt_s)
    tprd = cKDTree(pred_aligned)
    d_p2g, _ = tgt.query(pred_aligned)
    d_g2p, _ = tprd.query(gt_s)

    chamfer   = float(d_p2g.mean() + d_g2p.mean())
    precision = float((d_p2g < threshold).mean())
    recall    = float((d_g2p < threshold).mean())
    fscore    = 2 * precision * recall / (precision + recall + 1e-8)
    return {"chamfer": chamfer, "precision": precision,
            "recall":  recall,  "fscore":   fscore}


def _scale_align_depth(pred, gt, mask):
    pred_v, gt_v = pred[mask], gt[mask]
    scale = (gt_v * pred_v).sum() / (pred_v * pred_v).sum()
    return pred * scale


def _resize_if_needed(pred, target_shape):
    if pred.shape != target_shape:
        from skimage.transform import resize
        pred = resize(pred, target_shape, anti_aliasing=True, preserve_range=True)
    return pred


# ---------------------------------------------------------------------------
# Scene / view helpers
# ---------------------------------------------------------------------------

def build_mixed_view_list(scene_dir, styled_root, style_names, n_styled=4, n_original=4):
    """Return (image_paths, view_ids, input_labels) for a fixed 8-view mixed batch."""
    blended_dir = os.path.join(scene_dir, "blended_images")
    depth_dir   = os.path.join(scene_dir, "rendered_depth_maps")
    scene_name  = os.path.basename(scene_dir)

    available = []
    for fn in sorted(os.listdir(blended_dir)):
        m = re.match(r"(\d{8})(?:_result)?\.(jpg|png)$", fn)
        if not m:
            continue
        vid = int(m.group(1))
        if os.path.exists(os.path.join(depth_dir, f"{vid:08d}.pfm")):
            available.append((vid, fn))

    available = sorted(available, key=lambda x: x[0])
    if "renamed" in scene_dir:
        available = [(vid, fn) for vid, fn in available if vid >= 5 and vid % 5 == 0]
    available = available[:8]

    def styled_path(vid, style):
        return os.path.join(styled_root, style, scene_name,
                            "blended_images", f"{vid:08d}_result.png")

    vid_to_styles = {
        vid: [s for s in style_names if os.path.isfile(styled_path(vid, s))]
        for vid, _ in available
    }
    styled_candidates = [vid for vid, _ in available if vid_to_styles[vid]]

    seed = int(hashlib.md5(scene_name.encode()).hexdigest(), 16) % (2 ** 32)
    rng  = np.random.default_rng(seed)

    n = min(n_styled, len(styled_candidates), len(style_names))
    chosen_vids  = styled_candidates[:n]
    styles_pool  = list(style_names[:n])
    rng.shuffle(styles_pool)
    chosen_style = {vid: styles_pool[i] for i, vid in enumerate(chosen_vids)}

    photo_vids = [vid for vid, _ in available if vid not in chosen_style][:n_original]

    image_paths, view_ids, input_labels = [], [], []
    for vid in chosen_vids:
        style = chosen_style[vid]
        image_paths.append(styled_path(vid, style))
        view_ids.append(vid)
        input_labels.append(style)
    for vid in photo_vids:
        fn = next(fn for v, fn in available if v == vid)
        image_paths.append(os.path.join(blended_dir, fn))
        view_ids.append(vid)
        input_labels.append("photo")

    return image_paths, view_ids, input_labels


# ---------------------------------------------------------------------------
# Point-cloud construction
# ---------------------------------------------------------------------------

def build_pointcloud_from_predictions(pred_depths_scene, scene_dir, view_ids,
                                      gt_scale_align=True):
    """Build a world-space point cloud from predicted depth maps.

    For each view:
      1. Load the camera intrinsics / extrinsics.
      2. Optionally scale-align the predicted depth to the GT depth so that
         the per-view clouds land in a consistent metric scale.
      3. Unproject to world space via depth_to_world_pcd.

    Returns
    -------
    pts_pred : (N, 3) float32  — predicted point cloud (all views merged)
    pts_gt   : (N, 3) float32  — GT point cloud for reference
    """
    pts_pred_parts = []
    pts_gt_parts   = []

    depth_dir = os.path.join(scene_dir, "rendered_depth_maps")
    cam_dir   = os.path.join(scene_dir, "cams")

    for i, vid in enumerate(view_ids):
        cam_path   = os.path.join(cam_dir,   f"{vid:08d}_cam.txt")
        depth_path = os.path.join(depth_dir, f"{vid:08d}.pfm")

        if not os.path.exists(cam_path):
            continue

        K, E = read_cam(cam_path)
        gt   = read_pfm(depth_path)
        pred = _resize_if_needed(pred_depths_scene[i], gt.shape)

        if gt_scale_align:
            mask = (gt > 0) & np.isfinite(gt)
            if mask.sum() > 0:
                pred = _scale_align_depth(pred, gt, mask)

        pts_pred_parts.append(depth_to_world_pcd(pred, K, E))
        pts_gt_parts.append(depth_to_world_pcd(gt,   K, E))

    pts_pred = np.concatenate(pts_pred_parts, axis=0) if pts_pred_parts else np.empty((0, 3))
    pts_gt   = np.concatenate(pts_gt_parts,   axis=0) if pts_gt_parts   else np.empty((0, 3))
    return pts_pred.astype(np.float32), pts_gt.astype(np.float32)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def subsample_pts(pts, max_pts=100_000):
    if len(pts) > max_pts:
        idx = np.random.choice(len(pts), max_pts, replace=False)
        return pts[idx]
    return pts


def visualize_pointcloud(pts_pred_per_model, model_labels, pts_gt,
                         metrics_per_model, scene_name, out_path, max_pts=100_000):
    """Save a 3-row figure comparing GT vs predicted point clouds.

    Row 0: top-down  (X-Z plane)
    Row 1: side      (X-Y plane, Y flipped so up = up)
    Row 2: metrics   (Chamfer / Precision / Recall / F-score text panels)

    metrics_per_model: list of dicts (one per model) from compute_pcd_metrics,
                       or None entries for models that failed.
    """
    gt_sub = subsample_pts(pts_gt, max_pts)

    n_cols = len(pts_pred_per_model) + 1
    fig, axes = plt.subplots(3, n_cols, figsize=(5 * n_cols, 11), facecolor="#0e0e0e",
                             gridspec_kw={"height_ratios": [3, 3, 1]})

    def _scatter(ax, pts, color, title, xi, yi, flip_y=False):
        ys = -pts[:, yi] if flip_y else pts[:, yi]
        ax.scatter(pts[:, xi], ys, s=0.3, c=color, alpha=0.4, rasterized=True)
        ax.set_title(title, color="#aaa", fontsize=9)
        ax.set_facecolor("#0e0e0e")
        ax.tick_params(colors="#555")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        ax.set_aspect("equal", adjustable="datalim")

    def _metrics_panel(ax, m, label):
        ax.set_facecolor("#1a1a1a")
        ax.axis("off")
        if m is None:
            ax.text(0.5, 0.5, "N/A", color="#888", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10)
            return
        lines = [
            label,
            f"Chamfer   {m['chamfer']:.4f}",
            f"Precision {m['precision']:.4f}",
            f"Recall    {m['recall']:.4f}",
            f"F-score   {m['fscore']:.4f}",
        ]
        colors = ["#fff", "#aaa", "#aaa", "#aaa",
                  "#a5d6a7" if m["fscore"] >= 0.5 else "#ef9a9a"]
        for row_i, (line, color) in enumerate(zip(lines, colors)):
            ax.text(0.05, 0.85 - row_i * 0.18, line, color=color,
                    transform=ax.transAxes, fontsize=8.5,
                    fontfamily="monospace", va="top")

    # --- GT column ---
    _scatter(axes[0, 0], gt_sub, "#4fc3f7", "GT — top view  (X-Z)", xi=0, yi=2)
    _scatter(axes[1, 0], gt_sub, "#4fc3f7", "GT — side view (X-Y)", xi=0, yi=1, flip_y=True)
    axes[2, 0].set_facecolor("#1a1a1a")
    axes[2, 0].axis("off")
    axes[2, 0].text(0.5, 0.5, "GT", color="#4fc3f7", ha="center", va="center",
                    transform=axes[2, 0].transAxes, fontsize=11, fontweight="bold")

    # --- model columns ---
    for col, (pts_pred, label, m) in enumerate(
            zip(pts_pred_per_model, model_labels, metrics_per_model), start=1):
        pred_sub = subsample_pts(pts_pred, max_pts)
        _scatter(axes[0, col], pred_sub, "#ff8a65",
                 f"{label} — top view  (X-Z)", xi=0, yi=2)
        _scatter(axes[1, col], pred_sub, "#ff8a65",
                 f"{label} — side view (X-Y)", xi=0, yi=1, flip_y=True)
        _metrics_panel(axes[2, col], m, label)

    plt.suptitle(f"Point clouds — {scene_name}",
                 color="white", fontsize=12, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0e0e0e")
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Build and compare point clouds from depth predictions of 3 models "
                    "on a mixed 8-view scene input (4 styled + 4 original photographs).")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--scene_dir", help="Path to a single BlendedMVS scene directory")
    group.add_argument("--data_dir",  help="Path to BlendedMVS root; iterates over scene_* dirs")
    p.add_argument("--max_scenes",     type=int, default=20)
    p.add_argument("--styled_root",    required=True)
    p.add_argument("--style_names",    nargs="+", default=DEFAULT_STYLES)
    p.add_argument("--checkpoint",     default="facebook/map-anything")
    p.add_argument("--lora_path_1",    required=True)
    p.add_argument("--lora_path_2",    required=True)
    p.add_argument("--label_baseline", default="baseline")
    p.add_argument("--label_1",        default="lora_mixed_gray")
    p.add_argument("--label_2",        default="lora_consistency")
    p.add_argument("--out_dir",        default="lora/results/pointcloud_visualizations")
    p.add_argument("--grayscale",      action="store_true")
    p.add_argument("--no_scale_align", action="store_true",
                   help="Skip GT-based scale alignment of predicted depths before unprojection")
    p.add_argument("--max_pts",        type=int, default=100_000,
                   help="Max points per cloud shown in scatter plots (default: 100k)")
    p.add_argument("--icp_pts",        type=int, default=50_000,
                   help="Max points used for ICP + metric computation (default: 50k)")
    args = p.parse_args()

    import time
    t0 = time.time()

    def log(msg, flush=True):
        elapsed = time.time() - t0
        print(f"[{elapsed:6.0f}s] {msg}", flush=flush)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device: {device}")

    model_specs = [
        (args.label_baseline, None),
        (args.label_1,        args.lora_path_1),
        (args.label_2,        args.lora_path_2),
    ]
    labels = [label for label, _ in model_specs]

    n_styled   = len(args.style_names)
    n_original = 8 - n_styled
    gt_scale_align = not args.no_scale_align

    # --- build scene list ---
    if args.scene_dir:
        scene_dirs = [args.scene_dir]
    else:
        all_scenes = sorted(
            d for d in os.listdir(args.data_dir)
            if d.startswith("scene_")
            and os.path.isdir(os.path.join(args.data_dir, d))
        )
        def _count_images(scene):
            blended = os.path.join(args.data_dir, scene, "blended_images")
            return len([f for f in os.listdir(blended) if os.path.isfile(os.path.join(blended, f))])
        all_scenes = [s for s in all_scenes if _count_images(s) < 300]
        all_scenes = all_scenes[:args.max_scenes]
        scene_dirs = [os.path.join(args.data_dir, s) for s in all_scenes]

    # --- prepare per-scene inputs ---
    log(f"[1/3] preparing inputs for {len(scene_dirs)} scenes...")
    scene_data = []
    for i, scene_dir in enumerate(scene_dirs, 1):
        image_paths, view_ids, input_labels = build_mixed_view_list(
            scene_dir, args.styled_root, args.style_names,
            n_styled=n_styled, n_original=n_original,
        )
        if not view_ids:
            log(f"  [{i}/{len(scene_dirs)}] {os.path.basename(scene_dir)}: skip (no valid views)")
            continue
        styled_count = sum(l != "photo" for l in input_labels)
        log(f"  [{i}/{len(scene_dirs)}] {os.path.basename(scene_dir)}: "
            f"{len(image_paths)} views ({styled_count} styled, {len(image_paths)-styled_count} photo)")
        views = load_images(image_paths, resolution_set=518, norm_type="dinov2",
                            patch_size=14, grayscale=args.grayscale)
        scene_data.append((scene_dir, view_ids, input_labels, views))

    if not scene_data:
        log("no valid scenes found, exiting.")
        return
    log(f"  inputs ready for {len(scene_data)} scenes.")

    # --- one pass per model: load → infer all scenes → unload ---
    # pred_depths[model_idx][scene_idx] = (N_views, H, W) numpy array
    pred_depths = [None] * len(model_specs)

    for m_idx, (label, lora_path) in enumerate(model_specs):
        log(f"[2/3] model {m_idx+1}/{len(model_specs)}: loading {label}...")
        if lora_path:
            model, _ = load_with_lora(lora_path, base_checkpoint=args.checkpoint, device=device)
        else:
            model, _ = get_model(args.checkpoint, device=device)
        log(f"  {label} loaded.")

        scene_depths = []
        for s_idx, (scene_dir, view_ids, input_labels, views) in enumerate(scene_data, 1):
            log(f"  inferring scene {s_idx}/{len(scene_data)}: {os.path.basename(scene_dir)}...")
            with torch.no_grad():
                preds = infer(model, views)
            depth = np.stack(
                [p["depth_along_ray"].squeeze(0)[..., 0].cpu().numpy() for p in preds],
                axis=0,
            )  # (N_views, H, W)
            scene_depths.append(depth)

        pred_depths[m_idx] = scene_depths
        del model
        torch.cuda.empty_cache()
        log(f"  {label}: all scenes done, model unloaded.")

    # --- build point clouds, compute metrics, save figures ---
    log(f"[3/3] building point clouds and saving figures for {len(scene_data)} scenes...")

    # Accumulate rows for CSV summary: list of dicts
    csv_rows = []

    for s_idx, (scene_dir, view_ids, input_labels, _) in enumerate(scene_data):
        scene_name = os.path.basename(scene_dir)
        log(f"  [{s_idx+1}/{len(scene_data)}] {scene_name}: building point clouds...")

        pts_pred_per_model = []
        metrics_per_model  = []
        pts_gt = None

        for m_idx, (label, _) in enumerate(model_specs):
            pts_pred, pts_gt_m = build_pointcloud_from_predictions(
                pred_depths[m_idx][s_idx],
                scene_dir,
                view_ids,
                gt_scale_align=gt_scale_align,
            )
            pts_pred_per_model.append(pts_pred)
            if pts_gt is None:
                pts_gt = pts_gt_m

            log(f"    {label}: computing Chamfer / F-score "
                f"({len(pts_pred):,} pred pts, {len(pts_gt_m):,} GT pts)...")
            m = compute_pcd_metrics(pts_pred, pts_gt_m, max_pts=args.icp_pts)
            metrics_per_model.append(m)

            if m is not None:
                log(f"    {label}: chamfer={m['chamfer']:.4f}  "
                    f"precision={m['precision']:.4f}  recall={m['recall']:.4f}  "
                    f"fscore={m['fscore']:.4f}")
                csv_rows.append({
                    "scene":     scene_name,
                    "model":     label,
                    "chamfer":   f"{m['chamfer']:.6f}",
                    "precision": f"{m['precision']:.6f}",
                    "recall":    f"{m['recall']:.6f}",
                    "fscore":    f"{m['fscore']:.6f}",
                })
            else:
                log(f"    {label}: metrics unavailable (empty cloud)")

        # out_path = os.path.join(args.out_dir, scene_name, "pointcloud_comparison.png")
        # log(f"  saving {out_path}")
        # visualize_pointcloud(
        #     pts_pred_per_model, labels, pts_gt,
        #     metrics_per_model=metrics_per_model,
        #     scene_name=scene_name,
        #     out_path=out_path,
        #     max_pts=args.max_pts,
        # )

    # --- write CSV summary ---
    if csv_rows:
        csv_path = os.path.join(args.out_dir, "metrics_summary.csv")
        os.makedirs(args.out_dir, exist_ok=True)
        fieldnames = ["scene", "model", "chamfer", "precision", "recall", "fscore"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        log(f"metrics saved to {csv_path}")

        # Print per-model averages
        for label in labels:
            rows = [r for r in csv_rows if r["model"] == label]
            if not rows:
                continue
            avg_chamfer   = np.mean([float(r["chamfer"])   for r in rows])
            avg_precision = np.mean([float(r["precision"]) for r in rows])
            avg_recall    = np.mean([float(r["recall"])    for r in rows])
            avg_fscore    = np.mean([float(r["fscore"])    for r in rows])
            log(f"  [{label}] avg over {len(rows)} scenes — "
                f"chamfer={avg_chamfer:.4f}  precision={avg_precision:.4f}  "
                f"recall={avg_recall:.4f}  fscore={avg_fscore:.4f}")

    log(f"done. figures saved to {args.out_dir}/")


if __name__ == "__main__":
    main()