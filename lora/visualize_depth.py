import argparse
import hashlib
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mapanything.utils.image import load_images
from lora.eval.evaluator import read_pfm
from lora.eval.models import get_model, load_with_lora, infer

DEFAULT_STYLES = ["engraving", "impressionism", "oil_painting", "watercolor"]


def _scale_align(pred, gt, mask):
    pred_v, gt_v = pred[mask], gt[mask]
    scale = (gt_v * pred_v).sum() / (pred_v * pred_v).sum()
    return pred * scale


def _depth_metrics(pred_aligned, gt, mask):
    abs_rel = float(np.mean(np.abs(pred_aligned[mask] - gt[mask]) / gt[mask]))
    rmse    = float(np.sqrt(np.mean((pred_aligned[mask] - gt[mask]) ** 2)))
    return abs_rel, rmse


def _resize_if_needed(pred, target_shape):
    if pred.shape != target_shape:
        from skimage.transform import resize
        pred = resize(pred, target_shape, anti_aliasing=True, preserve_range=True)
    return pred


def build_mixed_view_list(scene_dir, styled_root, style_names, n_styled=4, n_original=4):
    """Return (image_paths, view_ids, input_labels) for a fixed 8-view mixed batch.

    Selects n_styled views that each have a styled counterpart, assigns one style
    per view (deterministic via MD5 scene seed), then fills the remaining slots
    with original photograph views.  Total views = n_styled + n_original.

    input_labels: per-view tag shown in the suptitle (e.g. 'photo' or style name).
    """
    blended_dir = os.path.join(scene_dir, "blended_images")
    depth_dir   = os.path.join(scene_dir, "rendered_depth_maps")
    scene_name  = os.path.basename(scene_dir)

    # Gather all views that have both a photo and a depth map
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

    available = available[:8]  # cap at 8 candidates

    # Determine which views have styled counterparts (at least one style)
    def styled_path(vid, style):
        return os.path.join(styled_root, style, scene_name,
                            "blended_images", f"{vid:08d}_result.png")

    vid_to_styles = {
        vid: [s for s in style_names if os.path.isfile(styled_path(vid, s))]
        for vid, _ in available
    }

    styled_candidates = [vid for vid, _ in available if vid_to_styles[vid]]

    # Deterministic per-scene RNG to assign one style per view
    seed = int(hashlib.md5(scene_name.encode()).hexdigest(), 16) % (2 ** 32)
    rng  = np.random.default_rng(seed)

    n = min(n_styled, len(styled_candidates), len(style_names))
    # Pick n views and assign each a distinct style when possible
    chosen_vids  = styled_candidates[:n]
    styles_pool  = list(style_names[:n])
    rng.shuffle(styles_pool)
    chosen_style = {vid: styles_pool[i] for i, vid in enumerate(chosen_vids)}

    # Fill remaining slots with original-photo views not already chosen
    photo_vids = [vid for vid, _ in available if vid not in chosen_style][:n_original]

    # Build ordered list: styled views first, then photo views
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


def visualize_view(gt_depth, pred_depths, model_labels, view_idx, input_label, out_path):
    """Save a 2-row × (1 + N_models)-col comparison figure for one view.

    Row 0: GT depth | model predictions (magma)
    Row 1: blank    | absolute error maps (hot)

    input_label: 'photo' or a style name — shown in the figure title.
    """
    gt   = gt_depth
    mask = (gt > 0) & np.isfinite(gt)

    n_cols = len(pred_depths) + 1
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 8), facecolor="#0e0e0e")

    def _style_ax(ax, data, cmap, title):
        im = ax.imshow(data, cmap=cmap)
        ax.set_title(title, color="#aaa", fontsize=9)
        ax.axis("off")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.yaxis.set_tick_params(color="#aaa")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="#aaa")

    _style_ax(axes[0, 0], gt, "magma", "GT depth")
    axes[1, 0].axis("off")
    axes[1, 0].set_facecolor("#0e0e0e")

    for col, (pred_raw, label) in enumerate(zip(pred_depths, model_labels), start=1):
        pred = _resize_if_needed(pred_raw, gt.shape)
        if mask.sum() == 0:
            axes[0, col].axis("off")
            axes[1, col].axis("off")
            continue

        pred_aligned  = _scale_align(pred, gt, mask)
        abs_rel, rmse = _depth_metrics(pred_aligned, gt, mask)

        error        = np.abs(pred_aligned - gt)
        error[~mask] = np.nan

        _style_ax(axes[0, col], pred_aligned, "magma",
                  f"{label}\nAbsRel={abs_rel:.4f}  RMSE={rmse:.4f}")
        _style_ax(axes[1, col], error, "hot", f"{label} — abs error")

    plt.suptitle(f"View {view_idx:02d}  [{input_label}]",
                 color="white", fontsize=12, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0e0e0e")
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    p = argparse.ArgumentParser(
        description="Compare depth predictions of 3 models on a mixed 8-view scene input "
                    "(4 styled + 4 original photographs).")
    # scene selection: one of --scene_dir (single) or --data_dir (multiple)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--scene_dir", help="Path to a single BlendedMVS scene directory")
    group.add_argument("--data_dir",  help="Path to BlendedMVS root; iterates over scene_* dirs")
    p.add_argument("--max_scenes",   type=int, default=20,
                   help="Max number of scenes to process when using --data_dir (default: 20)")
    p.add_argument("--styled_root",  required=True,
                   help="Root of TeleStyle output (e.g. /scratch/.../telestyle_output)")
    p.add_argument("--style_names",  nargs="+", default=DEFAULT_STYLES,
                   help="Style subdirectory names; one per styled view (default: 4 styles)")
    p.add_argument("--checkpoint",   default="facebook/map-anything")
    p.add_argument("--lora_path_1",  required=True,
                   help="LoRA adapter path (e.g. mixed_styles_gray/final)")
    p.add_argument("--lora_path_2",  required=True,
                   help="LoRA adapter path (e.g. mixed_styles_gray_consistency/step_002500)")
    p.add_argument("--label_baseline", default="baseline")
    p.add_argument("--label_1",        default="lora_mixed_gray")
    p.add_argument("--label_2",        default="lora_consistency")
    p.add_argument("--out_dir",        default="depth_visualizations")
    p.add_argument("--grayscale",      action="store_true")
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

    # --- prepare per-scene inputs (images + metadata) ---
    log(f"[1/3] preparing inputs for {len(scene_dirs)} scenes...")
    scene_data = []  # list of (scene_dir, view_ids, input_labels, views)
    for i, scene_dir in enumerate(scene_dirs, 1):
        image_paths, view_ids, input_labels = build_mixed_view_list(
            scene_dir, args.styled_root, args.style_names,
            n_styled=n_styled, n_original=n_original,
        )
        if not view_ids:
            log(f"  [{i}/{len(scene_dirs)}] {os.path.basename(scene_dir)}: skip (no valid views)")
            continue
        styled_count = sum(l != 'photo' for l in input_labels)
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

    # --- save figures ---
    n_figures = sum(len(view_ids) for _, view_ids, _, _ in scene_data)
    log(f"[3/3] saving {n_figures} figures...")
    fig_idx = 0
    for s_idx, (scene_dir, view_ids, input_labels, _) in enumerate(scene_data):
        scene_name = os.path.basename(scene_dir)
        depth_dir  = os.path.join(scene_dir, "rendered_depth_maps")
        for i, (vid, input_label) in enumerate(zip(view_ids, input_labels)):
            fig_idx += 1
            gt    = read_pfm(os.path.join(depth_dir, f"{vid:08d}.pfm"))
            preds = [pred_depths[m][s_idx][i] for m in range(len(model_specs))]
            out_path = os.path.join(args.out_dir, scene_name, f"view{i:02d}_{input_label}.png")
            log(f"  [{fig_idx}/{n_figures}] {scene_name}/view{i:02d}_{input_label}.png")
            visualize_view(gt, preds, labels, view_idx=i,
                           input_label=input_label, out_path=out_path)

    log(f"done. {n_figures} figures saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
