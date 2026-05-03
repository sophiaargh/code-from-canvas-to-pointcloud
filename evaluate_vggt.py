import argparse
import os
import csv
import torch
import numpy as np
from scipy.spatial import cKDTree
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


def get_args():
    parser = argparse.ArgumentParser(
        description="Processes scenes through VGGT. Evaluates AbsRel, RMSE, Chamfer distance "
                    "and F-score compared to GT. Saves results to CSV."
    )
    parser.add_argument("--data_dir",       type=str, default="/scratch/izar/silly/BlendedMVS")
    parser.add_argument("--baseline_name",  type=str, default="photographs", help="Experiment name (used as output CSV filename)")
    parser.add_argument("--max_scenes",     type=int, default=None, help="Limit number of scenes (useful for quick tests)")
    parser.add_argument("--view_ids",       type=int, nargs="+", default=[0, 10, 20, 30, 40])
    parser.add_argument("--max_pts",        type=int, default=50_000, help="Max points to subsample per cloud for Chamfer/F-score")
    return parser.parse_args()


# file readers

def read_pfm(filepath):
    with open(filepath, 'rb') as f:
        header = f.readline().decode('latin-1').strip()
        assert header in ('PF', 'Pf'), f"Not a PFM file: {header}"
        W, H   = map(int, f.readline().decode('latin-1').strip().split())
        scale  = float(f.readline().decode('latin-1').strip())
        endian = '<' if scale < 0 else '>'
        data   = np.frombuffer(f.read(), dtype=np.dtype(endian + 'f'))
    return data.reshape((H, W))[::-1].copy()


def read_cam(filepath):
    """Parse BlendedMVS _cam.txt → (K 3x3, E 4x4)"""
    with open(filepath) as f:
        lines = [l.strip() for l in f if l.strip()]
    E = np.array([list(map(float, lines[i].split())) for i in range(1, 5)])  # 4×4
    K = np.array([list(map(float, lines[i].split())) for i in range(6, 9)])  # 3×3
    return K, E


# geometry helpers

def depth_to_world_pcd(depth, K, E):
    """Unproject a depth map to a world-space point cloud.

    depth : (H, W)  — GT depth (invalid pixels are 0 or nan)
    K     : (3, 3)  — camera intrinsics
    E     : (4, 4)  — world→camera extrinsics
    returns (N, 3)  — points in world space
    """
    mask = (depth > 0) & np.isfinite(depth)
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]

    u = np.arange(depth.shape[1], dtype=np.float32)
    v = np.arange(depth.shape[0], dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    x = (uu - cx) * depth / fx
    y = (vv - cy) * depth / fy
    z = depth

    pts_cam   = np.stack([x, y, z], axis=-1)[mask]   # (N, 3)
    R, t      = E[:3, :3], E[:3, 3]
    pts_world = (R.T @ (pts_cam - t).T).T             # (N, 3)
    return pts_world


# depth metrics

def compute_depth_error(pred_depth_all_views, gt_path, view_idx):
    """
    pred_depth_all_views : (V, H, W) numpy array from VGGT
    gt_path              : path to .pfm GT depth for this view
    view_idx             : which view index (0-based within the V views passed to VGGT)
    returns dict with AbsRel and RMSE
    """
    gt   = read_pfm(gt_path)
    pred = pred_depth_all_views[view_idx]

    # resize pred to match GT if VGGT resized the input images
    if pred.shape != gt.shape:
        from skimage.transform import resize
        pred = resize(pred, gt.shape, anti_aliasing=True, preserve_range=True)

    mask = (gt > 0) & np.isfinite(gt)
    if mask.sum() == 0:
        print(f"  [warn] no valid GT pixels for view {view_idx}, skipping")
        return None

    # affine-invariant scale alignment: least-squares scale only
    pred_v = pred[mask]
    gt_v   = gt[mask]
    scale  = (gt_v * pred_v).sum() / (pred_v * pred_v).sum()
    pred_aligned = pred * scale

    abs_rel = float(np.mean(np.abs(pred_aligned[mask] - gt_v) / gt_v))
    rmse    = float(np.sqrt(np.mean((pred_aligned[mask] - gt_v) ** 2)))
    return {"AbsRel": abs_rel, "RMSE": rmse}


# point-cloud metrics

def icp_align(pred_pts, gt_pts, n_iters=50):
    """Centroid + ICP alignment. Returns aligned pred_pts."""
    pred_aligned = pred_pts - pred_pts.mean(0) + gt_pts.mean(0)

    # scale alignment — match overall spread
    pred_scale = np.linalg.norm(pred_aligned.std(0))
    gt_scale   = np.linalg.norm(gt_pts.std(0))
    pred_aligned = pred_aligned * (gt_scale / pred_scale)

    for _ in range(n_iters):
        _, idx      = cKDTree(gt_pts).query(pred_aligned)
        gt_matched  = gt_pts[idx]
        pc          = pred_aligned.mean(0);  gc = gt_matched.mean(0)
        H           = (pred_aligned - pc).T @ (gt_matched - gc)
        U, _, Vt    = np.linalg.svd(H)
        R           = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1;  R = Vt.T @ U.T
        t           = gc - R @ pc
        pred_new    = (R @ pred_aligned.T).T + t
        if np.abs(pred_new - pred_aligned).max() < 1e-7:
            break
        pred_aligned = pred_new
    
    return pred_aligned


def evaluate_pointcloud(predictions, scene_dir, view_ids, max_pts=50_000):
    # build GT cloud from all views
    gt_parts = []
    last_gt_depth = None
    for vid in view_ids:
        depth_path = os.path.join(scene_dir, "rendered_depth_maps", f"{vid:08d}.pfm")
        cam_path   = os.path.join(scene_dir, "cams",                f"{vid:08d}_cam.txt")
        gt_depth   = read_pfm(depth_path)
        K, E       = read_cam(cam_path)
        gt_parts.append(depth_to_world_pcd(gt_depth, K, E))
        last_gt_depth = gt_depth

    gt_pts = np.concatenate(gt_parts, axis=0)

    # predicted cloud from VGGT world_points
    pred_pts = predictions["world_points"].squeeze(0).reshape(-1, 3).cpu().numpy()
    conf     = predictions["world_points_conf"].squeeze(0).reshape(-1).cpu().numpy()
    pred_pts = pred_pts[conf > np.percentile(conf, 50)]  # top-50% confidence

    # subsample
    def subsample(pts):
        if len(pts) > max_pts:
            return pts[np.random.choice(len(pts), max_pts, replace=False)]
        return pts

    pred_pts = subsample(pred_pts)
    gt_pts   = subsample(gt_pts)
    print(f"  pcd  pred={len(pred_pts):,}  gt={len(gt_pts):,}")

    # align then evaluate
    pred_aligned = icp_align(pred_pts, gt_pts)

    valid = last_gt_depth[last_gt_depth > 0]
    scene_scale = np.linalg.norm(gt_pts.std(axis=0))  # overall spread of GT cloud
    threshold = 0.02 * scene_scale  # 2% of scene scale
    print(f"threshold: {threshold:.6f}  |  scene_scale: {scene_scale:.6f}")
    
    tgt  = cKDTree(gt_pts)
    tprd = cKDTree(pred_aligned)
    d_p2g, _ = tgt.query(pred_aligned)
    d_g2p, _ = tprd.query(gt_pts)
    print("d_p2g mean", d_p2g.mean())
    print("pred, gt std", pred_aligned.std(0), gt_pts.std(0))

    # for pct in [0.01, 0.02, 0.05, 0.10]:
    #     t = pct * scene_scale
    #     p = float((d_p2g < t).mean())
    #     r = float((d_g2p < t).mean())
    #     f = 2*p*r/(p+r+1e-8)
    #     print(f"  {pct*100:.0f}% → threshold={t:.4f}  F={f:.4f}")
        
    chamfer   = float(d_p2g.mean() + d_g2p.mean())
    precision = float((d_p2g < threshold).mean())
    recall    = float((d_g2p < threshold).mean())
    fscore    = 2 * precision * recall / (precision + recall + 1e-8)

    return {"chamfer": chamfer, "precision": precision,
            "recall": recall,   "fscore": fscore}


def main(args):
    scenes = sorted(
        d for d in os.listdir(args.data_dir)
        if d.startswith("scene") and os.path.isdir(os.path.join(args.data_dir, d))
    )
    print(f"Found {len(scenes)} scenes")
    if args.max_scenes:
        scenes = scenes[:args.max_scenes]
        print(f"Limiting to {len(scenes)} scenes")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = (torch.bfloat16
              if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
              else torch.float16)
    print(f"Device: {device}  |  dtype: {dtype}")

    print("Loading VGGT...", flush=True)
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()

    view_ids = args.view_ids
    print(f"View IDs: {view_ids}", flush=True)

    all_rows = []  # one dict per scene for CSV

    for scene in scenes:
        scene_dir = os.path.join(args.data_dir, scene)
        print(f"\n{'─'*60}\nScene: {scene}")

        image_paths    = [os.path.join(scene_dir, "blended_images", f"{vid:08d}.jpg")
                          for vid in view_ids]
        gt_depth_paths = [os.path.join(scene_dir, "rendered_depth_maps", f"{vid:08d}.pfm")
                          for vid in view_ids]

        # check all files exist
        missing = [p for p in image_paths + gt_depth_paths if not os.path.exists(p)]
        if missing:
            print(f"  [skip] missing files: {missing}")
            continue

        images = load_and_preprocess_images(image_paths).to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)

        # depth metrics
        pred_depth = predictions["depth"].squeeze(0)[..., 0].cpu().numpy()  # (V, H, W)

        per_view = []
        for i, (vid, gt_path) in enumerate(zip(view_ids, gt_depth_paths)):
            m = compute_depth_error(pred_depth, gt_path, view_idx=i)
            if m is not None:
                per_view.append(m)
                print(f"  view {vid:08d}  AbsRel={m['AbsRel']:.4f}  RMSE={m['RMSE']:.4f}")

        if not per_view:
            print("  [skip] no valid depth views")
            continue

        abs_rel_scene = float(np.mean([m["AbsRel"] for m in per_view]))
        rmse_scene    = float(np.mean([m["RMSE"]   for m in per_view]))

        # point-cloud metrics
        pcd_metrics = evaluate_pointcloud(predictions, scene_dir, view_ids, args.max_pts)

        print(f"scene AbsRel={abs_rel_scene:.4f}  RMSE={rmse_scene:.4f}")
        print(f"Chamfer={pcd_metrics['chamfer']:.6f}  "
              f"F-score={pcd_metrics['fscore']:.4f}  "
              f"(P={pcd_metrics['precision']:.4f}  R={pcd_metrics['recall']:.4f})")

        all_rows.append({
            "scene":     scene,
            "baseline":  args.baseline_name,
            "AbsRel":    round(abs_rel_scene, 6),
            "RMSE":      round(rmse_scene, 6),
            "chamfer":   round(pcd_metrics["chamfer"],   6),
            "fscore":    round(pcd_metrics["fscore"],    6),
            "precision": round(pcd_metrics["precision"], 6),
            "recall":    round(pcd_metrics["recall"],    6),
        })
# TODO put the median too
    # summary
    if not all_rows:
        print("\nNo scenes evaluated successfully.")
        return

    print(f"\n{'═'*60}")
    print(f"Summary over {len(all_rows)} scenes:", flush=True)
    for metric in ["AbsRel", "RMSE", "chamfer", "fscore"]:
        vals = [r[metric] for r in all_rows]
        print(f"  {metric:10s}  mean={np.mean(vals):.4f}  std={np.std(vals):.4f}")

    # save CSV
    out_path = os.path.join("evaluation_results", f"{args.baseline_name}.csv")
    fieldnames = ["scene", "baseline", "AbsRel", "RMSE", "chamfer", "fscore", "precision", "recall"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
        # summary row
        writer.writerow({
            "scene":     "MEAN",
            "baseline":  args.baseline_name,
            "AbsRel":    round(np.mean([r["AbsRel"]    for r in all_rows]), 6),
            "RMSE":      round(np.mean([r["RMSE"]      for r in all_rows]), 6),
            "chamfer":   round(np.mean([r["chamfer"]   for r in all_rows]), 6),
            "fscore":    round(np.mean([r["fscore"]    for r in all_rows]), 6),
            "precision": round(np.mean([r["precision"] for r in all_rows]), 6),
            "recall":    round(np.mean([r["recall"]    for r in all_rows]), 6),
        })

    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    args = get_args()
    print("Arguments:", args)
    main(args)