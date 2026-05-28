import os
import csv
import numpy as np
import torch
from mapanything.utils.image import load_images
from ..models import infer
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

# --- Evaluator class ---

class StyleDegradeEvaluator:
    def __init__(self, model, device, number_stylized, max_pts=50_000, out_dir="evaluation_results"):
        self.model = model
        self.device = device
        self.max_pts = max_pts
        self.out_dir = out_dir
        self.number_stylized = number_stylized
        os.makedirs(self.out_dir, exist_ok=True)

    def _collect_views(self, image_dir, depth_dir, subsample_every_5=False):
        if not os.path.isdir(image_dir) or not os.path.isdir(depth_dir):
            return []

        available = []
        for fn in os.listdir(image_dir):
            m = re.match(r"(\d{8})(?:_result)?\.(jpg|png)$", fn)
            if not m:
                continue

            vid = int(m.group(1))
            if subsample_every_5 and not (vid >= 5 and (vid % 5) == 0):
                continue

            depth_path = os.path.join(depth_dir, f"{vid:08d}.pfm")
            if os.path.exists(depth_path):
                available.append((vid, os.path.join(image_dir, fn), depth_path))

        return sorted(available, key=lambda x: x[0])

    def evaluate_scene(self, data_dir, scene):
        renamed_scene_dir = os.path.join(data_dir, "renamed", scene)
        stylized_scene_dir = os.path.join(data_dir, "telestyle_output", "watercolor", scene)

        split_layout = os.path.isdir(renamed_scene_dir) and os.path.isdir(stylized_scene_dir)

        renamed_images_dir = os.path.join(renamed_scene_dir, "blended_images")
        stylized_images_dir = os.path.join(stylized_scene_dir, "blended_images")

        depth_dir = os.path.join(renamed_scene_dir, "rendered_depth_maps")
        if not os.path.isdir(depth_dir):
            depth_dir = os.path.join(stylized_scene_dir, "rendered_depth_maps")

        stylized_target = 0 if self.number_stylized is None else max(0, min(8, int(self.number_stylized)))
        photograph_target = 8 - stylized_target

        stylized_views = self._collect_views(stylized_images_dir, depth_dir)
        photograph_views = self._collect_views(renamed_images_dir, depth_dir, subsample_every_5=True)

        photograph_views = photograph_views[:photograph_target]
        photograph_ids = {view[0] for view in photograph_views}
        last_photograph_id = max(photograph_ids) if photograph_ids else -1

        stylized_views = [
            view for view in stylized_views
            if view[0] > last_photograph_id and view[0] not in photograph_ids
        ]

        selected = photograph_views + stylized_views[:stylized_target]
        selected = sorted(selected, key=lambda x: x[0])
        

        if not selected:
            print(f"No valid selected views in {scene}")
            return None

        image_paths = [view[1] for view in selected]

        print(image_paths)

        gt_depth_paths = [view[2] for view in selected]

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
        

        row = {
            "scene": scene,
            "AbsRel": round(abs_rel_scene, 6),
            "RMSE": round(rmse_scene, 6),
        }
        return row
    
    def run(self, data_dir):
        renamed_root = os.path.join(data_dir, "renamed")
        stylized_root = os.path.join(data_dir, "telestyle_output", "watercolor")

        # count valid image files in blended_images 
        def _count_blended_images(scene, directory):
            blended = os.path.join(directory, scene, "blended_images")
            if not os.path.isdir(blended):
                return 0
            return len([name for name in os.listdir(blended) if os.path.isfile(os.path.join(blended, name))])

        renamed_scenes = {
            d for d in os.listdir(renamed_root)
            if d.startswith("scene") and os.path.isdir(os.path.join(renamed_root, d))
        }
        renamed_scenes = sorted([s for s in renamed_scenes if _count_blended_images(s, renamed_root) < 300])

        stylized_scenes = {
            d for d in os.listdir(stylized_root)
            if d.startswith("scene") and os.path.isdir(os.path.join(stylized_root, d))
        }
        stylized_scenes = sorted([s for s in stylized_scenes if _count_blended_images(s, stylized_root) < 30])

        scene_dirs = sorted(set(renamed_scenes) & set(stylized_scenes))
        scenes = scene_dirs
        print(f"number of scenes {len(scenes)}")

        all_rows = []
        for scene in scenes:
            print(f"Evaluating {scene}", flush=True)
            row = self.evaluate_scene(data_dir, scene)
            if row is None:
                print(f"  [skip] {scene} (missing or invalid)")
                continue
            all_rows.append(row)
            print(f"  -> done: {scene}")
        if not all_rows:
            print("No scenes evaluated successfully.")
            return
        out_path = os.path.join(self.out_dir, f"multi_style_{self.number_stylized}.csv")
        fieldnames = ["scene", "AbsRel", "RMSE"]
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            #writer.writerows(all_rows)
            writer.writerow({
                "scene": "MEAN",
                "AbsRel": round(np.mean([r["AbsRel"] for r in all_rows]), 6),
                "RMSE": round(np.mean([r["RMSE"] for r in all_rows]), 6),
            })
            writer.writerow({
                "scene": "MEDIAN",
                "AbsRel": round(np.median([r["AbsRel"] for r in all_rows]), 6),
                "RMSE": round(np.median([r["RMSE"] for r in all_rows]), 6),
            })
        print(f"Results saved to: {out_path}")
