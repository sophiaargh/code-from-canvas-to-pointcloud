import os
import csv
import numpy as np
from scipy.spatial import cKDTree
import torch
from mapanything.utils.image import load_images
from .models import infer
import re


# --- file readers / geometry (adapted small helpers) ---

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
    with open(filepath) as f:
        lines = [l.strip() for l in f if l.strip()]
    E = np.array([list(map(float, lines[i].split())) for i in range(1, 5)])
    K = np.array([list(map(float, lines[i].split())) for i in range(6, 9)])
    return K, E


def depth_to_world_pcd(depth, K, E):
    mask = (depth > 0) & np.isfinite(depth)
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
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


# --- depth metric ---

def compute_depth_error(pred_depth_all_views, gt_path, view_idx):
    gt = read_pfm(gt_path)
    pred = pred_depth_all_views[view_idx]
    if pred.shape != gt.shape:
        from skimage.transform import resize
        pred = resize(pred, gt.shape, anti_aliasing=True, preserve_range=True)
    mask = (gt > 0) & np.isfinite(gt)
    if mask.sum() == 0:
        return None
    pred_v = pred[mask]
    gt_v   = gt[mask]
    scale  = (gt_v * pred_v).sum() / (pred_v * pred_v).sum()
    pred_aligned = pred * scale
    abs_rel = float(np.mean(np.abs(pred_aligned[mask] - gt_v) / gt_v))
    rmse    = float(np.sqrt(np.mean((pred_aligned[mask] - gt_v) ** 2)))
    return {"AbsRel": abs_rel, "RMSE": rmse}


# --- point-cloud helpers ---

def icp_align(pred_pts, gt_pts, n_iters=50):
    pred_aligned = pred_pts - pred_pts.mean(0) + gt_pts.mean(0)
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
    pred_parts = []
    for pred in predictions:
        pts = pred["pts3d"].squeeze(0).reshape(-1, 3).cpu().numpy()
        valid = np.isfinite(pts).all(axis=1)
        conf = pred.get("conf")
        if conf is not None:
            conf = conf.squeeze(0).reshape(-1).cpu().numpy()
            valid &= conf > np.percentile(conf, 50)
        pred_parts.append(pts[valid])
    pred_pts = np.concatenate(pred_parts, axis=0)
    def subsample(pts):
        if len(pts) > max_pts:
            return pts[np.random.choice(len(pts), max_pts, replace=False)]
        return pts
    pred_pts = subsample(pred_pts)
    gt_pts   = subsample(gt_pts)
    pred_aligned = icp_align(pred_pts, gt_pts)
    scene_scale = np.linalg.norm(gt_pts.std(axis=0))
    threshold = 0.02 * scene_scale
    tgt  = cKDTree(gt_pts)
    tprd = cKDTree(pred_aligned)
    d_p2g, _ = tgt.query(pred_aligned)
    d_g2p, _ = tprd.query(gt_pts)
    chamfer   = float(d_p2g.mean() + d_g2p.mean())
    precision = float((d_p2g < threshold).mean())
    recall    = float((d_g2p < threshold).mean())
    fscore    = 2 * precision * recall / (precision + recall + 1e-8)
    return {"chamfer": chamfer, "precision": precision,
            "recall": recall,   "fscore": fscore}


# --- Evaluator class ---

class Evaluator:
    def __init__(self, model, device, baseline_name="baseline", max_pts=50_000, out_dir="evaluation_results", max_scenes=None):
        self.model = model
        self.device = device
        self.baseline_name = baseline_name
        self.max_pts = max_pts
        self.out_dir = out_dir
        self.max_scenes = max_scenes
        os.makedirs(self.out_dir, exist_ok=True)

    def evaluate_scene(self, scene_dir):
        blended_dir = os.path.join(scene_dir, "blended_images")
        depth_dir = os.path.join(scene_dir, "rendered_depth_maps")
        if not os.path.isdir(blended_dir) or not os.path.isdir(depth_dir):
            print("Could not find scene !")
            return None

        # gather available ids that have both image and depth
        available = []

        for fn in os.listdir(blended_dir):
            m = re.match(r"(\d{8})(?:_result)?\.(jpg|png)$", fn)
            if not m:
                continue

            vid = int(m.group(1))
            depth_path = os.path.join(depth_dir, f"{vid:08d}.pfm")

            if os.path.exists(depth_path):
                available.append((vid, fn))

        available = sorted(available, key=lambda x: x[0])
        if not available:
            print("Could not find scene !")
            return None
        
        # If using the original renamed dataset, subsample every 5th frame.
        # Otherwise the dataset is already subsampled.
        if "renamed" in scene_dir:
            selected = [(vid, fn) for vid, fn in available if vid >= 5 and (vid % 5) == 0]
        else:
            selected = available

        selected = selected[:8]

        if not selected:
            print(f"No valid selected views in {scene_dir}")
            return None
    
        image_paths = [
            os.path.join(blended_dir, fn)
            for vid, fn in selected
        ]

        gt_depth_paths = [
            os.path.join(depth_dir, f"{vid:08d}.pfm")
            for vid, fn in selected
        ]

        selected_ids = [vid for vid, fn in selected]
        views = load_images(image_paths, resolution_set=518, norm_type="dinov2", patch_size=14)
        with torch.no_grad():
            predictions = infer(self.model, views)

        pred_depth = np.stack(
            [pred["depth_along_ray"].squeeze(0)[..., 0].cpu().numpy() for pred in predictions],
            axis=0,
        )
        per_view = []
        for i, gt_path in enumerate(gt_depth_paths):
            m = compute_depth_error(pred_depth, gt_path, view_idx=i)
            if m is not None:
                per_view.append(m)
        if not per_view:
            print("not per_view")
            return None
        abs_rel_scene = float(np.mean([m["AbsRel"] for m in per_view]))
        rmse_scene    = float(np.mean([m["RMSE"]   for m in per_view]))
        pcd_metrics = evaluate_pointcloud(predictions, scene_dir, selected_ids, self.max_pts)

        row = {
            "scene": os.path.basename(scene_dir),
            "baseline": self.baseline_name,
            "AbsRel": round(abs_rel_scene, 6),
            "RMSE": round(rmse_scene, 6),
            "chamfer": round(pcd_metrics["chamfer"], 6),
            "fscore": round(pcd_metrics["fscore"], 6),
            "precision": round(pcd_metrics["precision"], 6),
            "recall": round(pcd_metrics["recall"], 6),
        }
        return row

    def run(self, data_dir):
        scene_dirs = [d for d in os.listdir(data_dir)
              if d.startswith("scene") and os.path.isdir(os.path.join(data_dir, d))]

        # count valid image files in blended_images 
        def _count_blended_images(scene):
            blended = os.path.join(data_dir, scene, "blended_images")
            return len([name for name in os.listdir(blended) if os.path.isfile(os.path.join(blended, name))])

        # keep only scenes with fewer than 150 images (300 because there are the masked images in the original folder) if original or 30 images if stylized
        
        if "photographs" in self.baseline_name:
            scenes = sorted([s for s in scene_dirs if _count_blended_images(s) < 300])
        else:
            scenes = sorted([s for s in scene_dirs if _count_blended_images(s) < 30])

        print(f"Original number of scenes: {len(scene_dirs)}. Number of scenes kept: {len(scenes)}")

        if self.max_scenes:
            scenes = scenes[:self.max_scenes]
            print(f"Max scenes specified. Evaluating {len(scenes)} scenes")
        
        all_rows = []
        for scene in scenes:
            scene_dir = os.path.join(data_dir, scene)
            print(f"Evaluating {scene}", flush=True)
            row = self.evaluate_scene(scene_dir)
            if row is None:
                print(f"  [skip] {scene} (missing or invalid)")
                continue
            all_rows.append(row)
            print(f"  -> done: {scene}")
        if not all_rows:
            print("No scenes evaluated successfully.")
            return
        out_path = os.path.join(self.out_dir, f"{self.baseline_name}.csv")
        fieldnames = ["scene", "baseline", "AbsRel", "RMSE", "chamfer", "fscore", "precision", "recall"]
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
            writer.writerow({
                "scene": "MEAN",
                "baseline": self.baseline_name,
                "AbsRel": round(np.mean([r["AbsRel"] for r in all_rows]), 6),
                "RMSE": round(np.mean([r["RMSE"] for r in all_rows]), 6),
                "chamfer": round(np.mean([r["chamfer"] for r in all_rows]), 6),
                "fscore": round(np.mean([r["fscore"] for r in all_rows]), 6),
                "precision": round(np.mean([r["precision"] for r in all_rows]), 6),
                "recall": round(np.mean([r["recall"] for r in all_rows]), 6),
            })
            writer.writerow({
                "scene": "MEDIAN",
                "baseline": self.baseline_name,
                "AbsRel": round(np.median([r["AbsRel"] for r in all_rows]), 6),
                "RMSE": round(np.median([r["RMSE"] for r in all_rows]), 6),
                "chamfer": round(np.median([r["chamfer"] for r in all_rows]), 6),
                "fscore": round(np.median([r["fscore"] for r in all_rows]), 6),
                "precision": round(np.median([r["precision"] for r in all_rows]), 6),
                "recall": round(np.median([r["recall"] for r in all_rows]), 6),
            })
        print(f"Results saved to: {out_path}")
