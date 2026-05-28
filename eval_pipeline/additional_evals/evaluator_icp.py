import os
import csv
import numpy as np
from scipy.spatial import cKDTree
import torch
from mapanything.utils.image import load_images
from ..models import infer
from plyfile import PlyData


def read_ply_points(ply_path):
    """Read vertex positions from a PLY file using plyfile.
    Returns an (N,3) numpy array of XYZ coordinates.
    """
    ply = PlyData.read(ply_path)
    try:
        vert = ply["vertex"]
    except (KeyError, ValueError):
        return np.zeros((0, 3), dtype=np.float64)

    names = vert.data.dtype.names
    for n in ('x', 'y', 'z'):
        if n not in names:
            return np.zeros((0, 3), dtype=np.float64)
    x = vert.data['x'].astype(np.float64)
    y = vert.data['y'].astype(np.float64)
    z = vert.data['z'].astype(np.float64)
    pts = np.column_stack((x, y, z))
    return pts


def icp_align(pred_pts, gt_pts, n_iters=50):
    """Run ICP and return (aligned_pts, transform_dict).
    
    The returned transform encodes the full pipeline:
      1. center pred onto gt mean
      2. scale to match gt std
      3. iterative R, t refinement
    Saving it lets you replay the exact same alignment on a different
    pointcloud that lives in the same coordinate frame.
    """
    # Step 1: initial centering + scale normalisation
    pred_mean = pred_pts.mean(0)
    gt_mean   = gt_pts.mean(0)

    pred_centered = pred_pts - pred_mean
    pred_scale    = np.linalg.norm(pred_centered.std(0))
    gt_scale      = np.linalg.norm(gt_pts.std(0))
    scale_factor  = gt_scale / (pred_scale + 1e-8)

    pred_aligned = pred_centered * scale_factor + gt_mean   # same as original logic

    # Step 2: iterative closest point
    R_accum = np.eye(3)
    t_accum = np.zeros(3)

    for _ in range(n_iters):
        _, idx = cKDTree(gt_pts).query(pred_aligned)
        gt_matched = gt_pts[idx]

        pc = pred_aligned.mean(0)
        gc = gt_matched.mean(0)
        H  = (pred_aligned - pc).T @ (gt_matched - gc)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        t = gc - R @ pc

        pred_new = (R @ pred_aligned.T).T + t
        # Accumulate into a single (R, t) expressed in the *scaled* frame
        R_accum = R @ R_accum
        t_accum = R @ t_accum + t

        if np.abs(pred_new - pred_aligned).max() < 1e-7:
            break
        pred_aligned = pred_new

    transform = {
        "pred_mean":    pred_mean,      # subtract from new pred before applying
        "scale_factor": scale_factor,   # multiply after centering
        "gt_mean":      gt_mean,        # add after scaling (re-centering on GT)
        "R":            R_accum,        # final rotation  (3×3)
        "t":            t_accum,        # final translation (3,)
    }
    return pred_aligned, transform


def apply_transform(pred_pts, transform):
    """Replay a saved ICP transform on a new pointcloud."""
    pts = (pred_pts - transform["pred_mean"]) * transform["scale_factor"] + transform["gt_mean"]
    pts = (transform["R"] @ pts.T).T + transform["t"]
    return pts


def evaluate_pointcloud(predictions, gt_ply_path, max_pts=50_000,
                        saved_transform=None):
    """Evaluate predicted pointclouds against a GT PLY pointcloud.

    Parameters
    ----------
    saved_transform : dict or None
        When None  → run ICP and return the computed transform alongside metrics.
        When given → skip ICP and apply this pre-computed transform instead.

    Returns
    -------
    metrics : dict  (chamfer, precision, recall, fscore)
    transform : dict or None
        The ICP transform used.  None when saved_transform was supplied
        (it's already known to the caller).
    """
    gt_pts = read_ply_points(gt_ply_path)
    if gt_pts.size == 0:
        raise ValueError(f"No vertices found in GT PLY: {gt_ply_path}")

    pred_parts = []
    for pred in predictions:
        pts   = pred["pts3d"].squeeze(0).reshape(-1, 3).cpu().numpy()
        valid = np.isfinite(pts).all(axis=1)
        conf  = pred.get("conf")
        if conf is not None:
            conf   = conf.squeeze(0).reshape(-1).cpu().numpy()
            valid &= conf > np.percentile(conf, 50)
        pred_parts.append(pts[valid])
    if not pred_parts:
        raise ValueError("No predicted points found")
    pred_pts = np.concatenate(pred_parts, axis=0)

    def subsample(pts):
        if len(pts) > max_pts:
            return pts[np.random.choice(len(pts), max_pts, replace=False)]
        return pts

    pred_pts = subsample(pred_pts)
    gt_pts   = subsample(gt_pts)

    if saved_transform is None:
        pred_aligned, used_transform = icp_align(pred_pts, gt_pts)
    else:
        pred_aligned  = apply_transform(pred_pts, saved_transform)
        used_transform = None   # caller already has it

    scene_scale = np.linalg.norm(gt_pts.std(axis=0))
    threshold   = 0.02 * scene_scale
    tgt  = cKDTree(gt_pts)
    tprd = cKDTree(pred_aligned)
    d_p2g, _ = tgt.query(pred_aligned)
    d_g2p, _ = tprd.query(gt_pts)
    chamfer   = float(d_p2g.mean() + d_g2p.mean())
    precision = float((d_p2g < threshold).mean())
    recall    = float((d_g2p < threshold).mean())
    fscore    = 2 * precision * recall / (precision + recall + 1e-8)

    metrics = {
        "chamfer": chamfer, "precision": precision,
        "recall": recall,   "fscore": fscore,
    }
    return metrics, used_transform


# --- Evaluator class ---

class ICPSaveEvaluator:
    def __init__(self, model, device, baseline_name="baseline", max_pts=50_000,
                 out_dir="evaluation_results",
                 photo_transforms_dir=None):
        """
        Parameters
        ----------
        photo_transforms_dir : str or None
            Directory where photograph ICP transforms are stored as .npz files
            (one per scene, named  <scene>.npz).
            • None  → always compute ICP from scratch (original behaviour).
            • Path  → if baseline_name == "photographs", compute ICP and *save*
                      the transform here; otherwise *load* the saved transform
                      and skip ICP.
        """
        self.model               = model
        self.device              = device
        self.baseline_name       = baseline_name
        self.max_pts             = max_pts
        self.out_dir             = out_dir
        self.photo_transforms_dir = photo_transforms_dir
        os.makedirs(self.out_dir, exist_ok=True)
        if photo_transforms_dir:
            os.makedirs(photo_transforms_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Transform persistence helpers
    # ------------------------------------------------------------------

    def _transform_path(self, scene_name):
        return os.path.join(self.photo_transforms_dir, f"{scene_name}.npz")

    def _save_transform(self, scene_name, transform):
        np.savez(self._transform_path(scene_name), **transform)

    def _load_transform(self, scene_name):
        path = self._transform_path(scene_name)
        if not os.path.exists(path):
            return None
        d = np.load(path)
        return {k: d[k] for k in d.files}

    # ------------------------------------------------------------------

    def evaluate_scene(self, cleaned_scan_dir, points_scan_dir, scene_name):
        if not os.path.isdir(cleaned_scan_dir) or not os.path.isdir(points_scan_dir):
            print("Could not find scene (cleaned/points scan missing) !")
            return None

        image_files = [
            f for f in sorted(os.listdir(cleaned_scan_dir))
            if f.lower().endswith((".jpg", ".png"))
        ]
        if not image_files:
            print(f"No images found in {cleaned_scan_dir}")
            return None

        if self.baseline_name == "photographs":
            selected_files = image_files[::5][:8]
        else:
            selected_files = image_files[:8]
        image_paths = [os.path.join(cleaned_scan_dir, fn) for fn in selected_files]

        ply_path = None
        for fn in sorted(os.listdir(points_scan_dir)):
            if fn.lower().endswith(".ply"):
                ply_path = os.path.join(points_scan_dir, fn)
                break
        if ply_path is None:
            print(f"No GT PLY found under {points_scan_dir}")
            return None

        views = load_images(image_paths, resolution_set=518, norm_type="dinov2", patch_size=14)
        with torch.no_grad():
            predictions = infer(self.model, views)


        # Decide which ICP transform to use
        saved_transform = None
        if self.photo_transforms_dir:
            if self.baseline_name == "photographs":
                # Run ICP, then persist the transform for other baselines
                pcd_metrics, new_transform = evaluate_pointcloud(
                    predictions, ply_path, self.max_pts, saved_transform=None
                )
                self._save_transform(scene_name, new_transform)
                print(f"  [ICP] transform saved for {scene_name}")
            else:
                # Load the photograph transform; fall back to fresh ICP if missing
                saved_transform = self._load_transform(scene_name)
                if saved_transform is None:
                    print(
                        f"  [warn] No saved transform for {scene_name}; "
                        "falling back to per-style ICP"
                    )
                pcd_metrics, _ = evaluate_pointcloud(
                    predictions, ply_path, self.max_pts,
                    saved_transform=saved_transform
                )
        else:
            # Original behaviour: always compute ICP from scratch
            pcd_metrics, _ = evaluate_pointcloud(
                predictions, ply_path, self.max_pts
            )

        row = {
            "scene":     scene_name,
            "baseline":  self.baseline_name,
            "chamfer":   round(pcd_metrics["chamfer"],   6),
            "fscore":    round(pcd_metrics["fscore"],    6),
            "precision": round(pcd_metrics["precision"], 6),
            "recall":    round(pcd_metrics["recall"],    6),
        }
        return row

    def run(self, data_dir, max_scenes=None):
        cleaned_root = os.path.join(data_dir, "Cleaned")
        if self.baseline_name != "photographs":
            cleaned_root = os.path.join(
                data_dir, "telestyle_output", self.baseline_name
            )
        points_root = os.path.join(data_dir, "Points")
        if not os.path.isdir(cleaned_root) or not os.path.isdir(points_root):
            print("Could not find Cleaned/ or Points/ under data_dir")
            return

        cleaned_scans = {
            name for name in os.listdir(cleaned_root)
            if name.startswith("scan") and os.path.isdir(os.path.join(cleaned_root, name))
        }
        points_scans = {
            name for name in os.listdir(points_root)
            if name.startswith("scan") and os.path.isdir(os.path.join(points_root, name))
        }

        scenes = sorted(cleaned_scans & points_scans)
        print(f"Found {len(scenes)} matching scan folders")

        if max_scenes:
            scenes = scenes[:max_scenes]
            print(f"Max scenes specified. Evaluating {len(scenes)} scenes")

        all_rows = []
        for scene in scenes:
            cleaned_scan_dir = os.path.join(cleaned_root, scene)
            points_scan_dir  = os.path.join(points_root,  scene)
            print(f"Evaluating {scene}", flush=True)
            row = self.evaluate_scene(cleaned_scan_dir, points_scan_dir, scene)
            if row is None:
                print(f"  [skip] {scene} (missing or invalid)")
                continue
            all_rows.append(row)
            print(f"  -> done: {scene}")

        if not all_rows:
            print("No scenes evaluated successfully.")
            return

        out_path   = os.path.join(self.out_dir, f"{self.baseline_name}_DTU_same_ICP.csv")
        fieldnames = ["scene", "baseline", "chamfer", "fscore", "precision", "recall"]
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
            writer.writerow({
                "scene": "MEAN", "baseline": self.baseline_name,
                "chamfer":   round(np.mean([r["chamfer"]   for r in all_rows]), 6),
                "fscore":    round(np.mean([r["fscore"]    for r in all_rows]), 6),
                "precision": round(np.mean([r["precision"] for r in all_rows]), 6),
                "recall":    round(np.mean([r["recall"]    for r in all_rows]), 6),
            })
            writer.writerow({
                "scene": "MEDIAN", "baseline": self.baseline_name,
                "chamfer":   round(np.median([r["chamfer"]   for r in all_rows]), 6),
                "fscore":    round(np.median([r["fscore"]    for r in all_rows]), 6),
                "precision": round(np.median([r["precision"] for r in all_rows]), 6),
                "recall":    round(np.median([r["recall"]    for r in all_rows]), 6),
            })
        print(f"Results saved to: {out_path}")