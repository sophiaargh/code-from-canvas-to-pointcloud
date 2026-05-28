import os
import csv
import numpy as np
from scipy.spatial import cKDTree
import torch
from mapanything.utils.image import load_images
from ..models import infer
from plyfile import PlyData
import open3d as o3d
import copy


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
    pred_aligned = pred_pts - pred_pts.mean(0) + gt_pts.mean(0)
    pred_scale = np.linalg.norm(pred_aligned.std(0))
    gt_scale = np.linalg.norm(gt_pts.std(0))
    pred_aligned = pred_aligned * (gt_scale / (pred_scale + 1e-8))
    for _ in range(n_iters):
        _, idx = cKDTree(gt_pts).query(pred_aligned)
        gt_matched = gt_pts[idx]
        pc = pred_aligned.mean(0)
        gc = gt_matched.mean(0)
        H = (pred_aligned - pc).T @ (gt_matched - gc)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        t = gc - R @ pc
        pred_new = (R @ pred_aligned.T).T + t
        if np.abs(pred_new - pred_aligned).max() < 1e-7:
            break
        pred_aligned = pred_new
    return pred_aligned

def evaluate_pointcloud(predictions, gt_ply_path, scene_name, out_dir, max_pts=50_000,):
    """Evaluate predicted pointclouds against a GT PLY pointcloud.
    Returns chamfer, precision, recall, fscore.
    """
    gt_pts = read_ply_points(gt_ply_path)
    if gt_pts.size == 0:
        raise ValueError(f"No vertices found in GT PLY: {gt_ply_path}")

    pred_parts = []
    for pred in predictions:
        pts = pred["pts3d"].squeeze(0).reshape(-1, 3).cpu().numpy()
        valid = np.isfinite(pts).all(axis=1)
        conf = pred.get("conf")
        if conf is not None:
            conf = conf.squeeze(0).reshape(-1).cpu().numpy()
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
    gt_pts = subsample(gt_pts)

    pred_aligned = icp_align(pred_pts, gt_pts)
    scene_scale = np.linalg.norm(gt_pts.std(axis=0))
    threshold = 0.02 * scene_scale
    tgt = cKDTree(gt_pts)
    tprd = cKDTree(pred_aligned)
    d_p2g, _ = tgt.query(pred_aligned)
    d_g2p, _ = tprd.query(gt_pts)
    chamfer = float(d_p2g.mean() + d_g2p.mean())
    precision = float((d_p2g < threshold).mean())
    recall = float((d_g2p < threshold).mean())
    fscore = 2 * precision * recall / (precision + recall + 1e-8)

    visualize_aligned_pcd(pred_aligned, gt_ply_path, scene_name, out_dir)

    return {"chamfer": chamfer, "precision": precision, "recall": recall, "fscore": fscore}

def visualize_aligned_pcd(pred_aligned, gt_ply_path, scene_name, out_dir):
        print("Loading pointcloud in open3d")
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pred_aligned.astype(np.float64))

        gt_pts = read_ply_points(gt_ply_path)
        gt_pcd = o3d.geometry.PointCloud()
        gt_pcd.points = o3d.utility.Vector3dVector(gt_pts.astype(np.float64)) 
        # Paint GT point cloud a solid color (e.g. red)
        pcd.paint_uniform_color([0, 0.5, 1])   # blue
        gt_pcd.paint_uniform_color([1, 0, 0])  # RGB, values in [0, 1]

        render = o3d.visualization.rendering.OffscreenRenderer(640, 480)
        render.scene.add_geometry("point_cloud", pcd, o3d.visualization.rendering.MaterialRecord())
        render.scene.add_geometry("point_cloud_gt", gt_pcd, o3d.visualization.rendering.MaterialRecord())

        aabb = pcd.get_axis_aligned_bounding_box()
        extent = aabb.get_max_extent()
        distance = extent * 0.8
        center = pcd.get_center()

        views = [
            ("top", [0, 0, distance], [0, 1, 0]),
            ("front", [0, -distance, 0], [0, 0, 1]),
            ("back", [0, distance, 0], [0, 0, 1]),
            ("left", [-distance, 0, 0], [0, 0, 1]),
            ("right", [distance, 0, 0], [0, 0, 1]),
        ]


        for name, eye, up in views:
            render.scene.camera.look_at(center, center + np.array(eye), up)
            img = render.render_to_image()
            
            o3d.io.write_image(os.path.join(out_dir, f"{scene_name}_icp_{name}.png"), img)


        
        
def visualize_pcd(predictions, scene_name, out_dir, gt_ply_path):
        print("Loading pointcloud in open3d")
        
        all_points = []
        all_colors = []
        for pred in predictions:  # one dict per view
            pts3d = pred["pts3d"]
            img = pred["img_no_norm"]
            mask = pred.get("mask", None)

            if isinstance(pts3d, torch.Tensor):
                pts3d = pts3d.detach().float().cpu().numpy()
            if isinstance(img, torch.Tensor):
                img = img.detach().float().cpu().numpy()
            if isinstance(mask, torch.Tensor):
                mask = mask.detach().cpu().numpy()

            # remove batch dim if present (B,H,W,C)->(H,W,C), usually B=1
            if pts3d.ndim == 4:
                pts3d = pts3d[0]
            if img.ndim == 4:
                img = img[0]
            if mask is not None and mask.ndim == 4:
                mask = mask[0]

            points = pts3d.reshape(-1, 3)
            colors = img.reshape(-1, 3)

            valid = np.isfinite(points).all(axis=1)
            if mask is not None:
                valid = valid & (mask.reshape(-1) > 0)

            points = points[valid]
            colors = colors[valid]

            if colors.max() > 1.0:
                colors = colors / 255.0
            colors = np.clip(colors, 0.0, 1.0)

            all_points.append(points)
            all_colors.append(colors)

        points_merged = np.concatenate(all_points, axis=0)
        colors_merged = np.concatenate(all_colors, axis=0)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_merged.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors_merged.astype(np.float64))
        gt_pts = read_ply_points(gt_ply_path)
        gt_pcd = o3d.geometry.PointCloud()
        gt_pcd.points = o3d.utility.Vector3dVector(gt_pts.astype(np.float64)) 
        # Paint GT point cloud a solid color 
        pcd.paint_uniform_color([0, 0.5, 1])   # light blue
        gt_pcd.paint_uniform_color([1, 0, 0])  # red

        
        print(f"PCD center: {pcd.get_center()}")
        print(f"GT center: {gt_pcd.get_center()}")

        pcd_centered = copy.deepcopy(pcd)
        gt_pcd_centered = copy.deepcopy(gt_pcd)

        pcd_centered.translate(-pcd.get_center())
        gt_pcd_centered.translate(-gt_pcd.get_center())

        print(f"PCD center after: {pcd_centered.get_center()}")
        print(f"GT center after: {gt_pcd_centered.get_center()}")

        pcd_extent = pcd_centered.get_axis_aligned_bounding_box().get_max_extent()
        gt_extent = gt_pcd_centered.get_axis_aligned_bounding_box().get_max_extent()
        spacing = max(pcd_extent, gt_extent) * 0.6

        print(f"PCD extent: {pcd_extent:.2f}")
        print(f"GT extent: {gt_extent:.2f}")
        print(f"Spacing: {spacing:.2f}")

        pcd_centered.translate([-spacing, 0, 0])
        gt_pcd_centered.translate([ spacing, 0, 0])

        print(f"PCD center after spacing: {pcd_centered.get_center()}")
        print(f"GT center after spacing: {gt_pcd_centered.get_center()}")

        # Camera target = midpoint between the two cloud centers, not AABB center
        mid_center = (np.array(pcd_centered.get_center()) + np.array(gt_pcd_centered.get_center())) / 2
        extent = max(pcd_extent, gt_extent) + spacing * 2  # total scene width
        distance = extent * 1.5

        print(f"Mid center: {mid_center}")
        print(f"Scene extent: {extent:.2f}, distance: {distance:.2f}")

        render = o3d.visualization.rendering.OffscreenRenderer(1280, 960)

        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = 8.0

        render.scene.add_geometry("point_cloud", pcd_centered, mat)
        render.scene.add_geometry("point_cloud_gt", gt_pcd_centered, mat)

        combined = pcd_centered + gt_pcd_centered
        aabb = combined.get_axis_aligned_bounding_box()
        mid_center = aabb.get_center()
        extent = aabb.get_max_extent()
        distance = extent * 1.5

        print(f"mid_center: {mid_center}, extent: {extent:.2f}, distance: {distance:.2f}")

        # Intrinsic matrix with correct near/far for our scale
        fov_deg = 60.0
        width, height = 1280, 960
        near = distance * 0.01   # near plane: 1% of camera distance
        far  = distance * 10.0   # far plane: 10x camera distance — everything is inside

        print(f"near: {near:.2f}, far: {far:.2f}")

        views = [
            ("top",            [ 0,  0,  1], [0, 1, 0]),
            ("front",          [ 0, -1,  0], [0, 0, 1]),
            ("side",           [-1,  0,  0], [0, 0, 1]),
            ("iso_front_left", [-1, -1,  1], [0, 0, 1]),
        ]

        for name, direction, up in views:
            direction = np.array(direction, dtype=np.float64)
            direction = direction / np.linalg.norm(direction)
            eye_pos = mid_center + direction * distance

            render.scene.camera.look_at(mid_center, eye_pos, up)
            render.scene.camera.set_projection(
                fov_deg,
                width / height,  # aspect ratio
                near,
                far,
                o3d.visualization.rendering.Camera.FovType.Vertical
            )

            img = render.render_to_image()
            path = os.path.join(out_dir, f"{scene_name}_aligned_{name}.png")
            o3d.io.write_image(path, img)
            print(f"Saved {name}: eye={eye_pos.round(1)}")


class DTUEvaluator:
    def __init__(self, model, device, baseline_name="baseline", max_pts=50_000, out_dir="evaluation_results"):
        self.model = model
        self.device = device
        self.baseline_name = baseline_name
        self.max_pts = max_pts
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def evaluate_scene(self, cleaned_scan_dir, points_scan_dir, scene_name):
        if not os.path.isdir(cleaned_scan_dir) or not os.path.isdir(points_scan_dir):
            print("Could not find scene (cleaned/points scan missing) !")
            return None

        image_files = [f for f in sorted(os.listdir(cleaned_scan_dir)) if f.lower().endswith((".jpg", ".png"))]
        if not image_files:
            print(f"No images found in {cleaned_scan_dir}")
            return None

        # select up to 8 images; photographs uses every 5th frame
        if self.baseline_name == "photographs":
            selected_files = image_files[::5][:8]
        else:
            selected_files = image_files[:8]
        image_paths = [os.path.join(cleaned_scan_dir, fn) for fn in selected_files]

        # find GT PLY in the matching points scan folder
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

        visualize_pcd(predictions, scene_name, self.out_dir, ply_path)

        # Only evaluate pointcloud metrics against GT PLY
        pcd_metrics = evaluate_pointcloud(predictions, ply_path, scene_name, self.out_dir, self.max_pts)

        row = {
            "scene": scene_name,
            "baseline": self.baseline_name,
            "chamfer": round(pcd_metrics["chamfer"], 6),
            "fscore": round(pcd_metrics["fscore"], 6),
            "precision": round(pcd_metrics["precision"], 6),
            "recall": round(pcd_metrics["recall"], 6),
        }
        return row
    

    def run(self, data_dir, max_scenes=None):
        cleaned_root = os.path.join(data_dir, "Cleaned")
        if self.baseline_name != "photographs":
            cleaned_root = os.path.join(data_dir, "telestyle_output", self.baseline_name)
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
            points_scan_dir = os.path.join(points_root, scene)
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
        out_path = os.path.join(self.out_dir, f"{self.baseline_name}_DTU.csv")
        fieldnames = ["scene", "baseline", "chamfer", "fscore", "precision", "recall"]
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
            writer.writerow({
                "scene": "MEAN",
                "baseline": self.baseline_name,
                "chamfer": round(np.mean([r["chamfer"] for r in all_rows]), 6),
                "fscore": round(np.mean([r["fscore"] for r in all_rows]), 6),
                "precision": round(np.mean([r["precision"] for r in all_rows]), 6),
                "recall": round(np.mean([r["recall"] for r in all_rows]), 6),
            })
            writer.writerow({
                "scene": "MEDIAN",
                "baseline": self.baseline_name,
                "chamfer": round(np.median([r["chamfer"] for r in all_rows]), 6),
                "fscore": round(np.median([r["fscore"] for r in all_rows]), 6),
                "precision": round(np.median([r["precision"] for r in all_rows]), 6),
                "recall": round(np.median([r["recall"] for r in all_rows]), 6),
            })
        print(f"Results saved to: {out_path}")
