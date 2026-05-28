"""
Evaluates the ControlNet hyperparameter grid produced by
TeleStyle/controlnet_tune_grid.py.

For every (strength, scale, style, frame) combination it computes the cosine
similarity in DINOv2 space between the destylized image and the original photo.
Prints a 2-D heatmap (strengths × scales) averaged over styles and frames, and
writes a CSV with per-image rows.

Usage:
    python eval_controlnet_grid.py \
        --data-root /scratch/izar/silly/BlendedMVS \
        --tuning-dir ~/destylize_tuning \
        --scene scene_0000 \
        --out evaluation_results/controlnet_grid.csv
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


@torch.no_grad()
def embed(model, processor, images: list, device, batch_size: int = 32) -> torch.Tensor:
    out = []
    for i in range(0, len(images), batch_size):
        inputs = processor(images=images[i:i + batch_size], return_tensors="pt").to(device)
        cls = model(**inputs).last_hidden_state[:, 0]
        out.append(F.normalize(cls, dim=-1).cpu())
    return torch.cat(out, dim=0)


# Matches: "00000001_result_s0.35_c0.80"
FNAME_RE = re.compile(r'^(.+)_result_s([\d.]+)_c([\d.]+)$')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/scratch/izar/silly/BlendedMVS")
    parser.add_argument("--tuning-dir", required=True,
                        help="Root output of controlnet_tune_grid.py (contains {style}/ subdirs)")
    parser.add_argument("--scene", required=True,
                        help="Scene name to evaluate, e.g. scene_0000")
    parser.add_argument("--styles", nargs="+",
                        default=["watercolor", "oil_painting", "impressionism", "engraving"])
    parser.add_argument("--out", default="evaluation_results/controlnet_grid.csv")
    parser.add_argument("--model", default="facebook/dinov2-large")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    tuning_dir = Path(args.tuning_dir).expanduser()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {args.model} ...")
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float16).to(device)
    model.eval()

    orig_dir = data_root / "renamed" / args.scene / "blended_images"
    orig_by_stem = {p.stem: p for p in sorted(orig_dir.glob("*.jpg"))
                    if not p.stem.endswith("masked")}

    # Collect per-style frame sets first so we can restrict to the intersection
    frames_per_style: dict[str, set] = {}
    entries_per_style: dict[str, list] = {}
    for style in args.styles:
        style_dir = tuning_dir / style
        if not style_dir.exists():
            print(f"[SKIP] {style_dir} not found")
            continue
        stems = set()
        entries = []
        for tuned_img in sorted(style_dir.glob("*_result_s*_c*.png")):
            m = FNAME_RE.match(tuned_img.stem)
            if m:
                stems.add(m.group(1))
                entries.append(tuned_img)
        frames_per_style[style] = stems
        entries_per_style[style] = entries

    # Only score frames present in every style — ensures a fair, balanced average
    common_frames = set.intersection(*frames_per_style.values()) if frames_per_style else set()
    print(f"Frames per style: { {s: len(f) for s, f in frames_per_style.items()} }")
    print(f"Common frames (used for scoring): {len(common_frames)}")

    rows = []
    for style, entries in entries_per_style.items():
        for tuned_img in entries:
            m = FNAME_RE.match(tuned_img.stem)
            if not m:
                continue
            orig_stem, strength, scale = m.group(1), float(m.group(2)), float(m.group(3))

            if orig_stem not in common_frames or orig_stem not in orig_by_stem:
                continue

            orig_img = Image.open(orig_by_stem[orig_stem]).convert("RGB")
            var_img  = Image.open(tuned_img).convert("RGB")

            orig_emb = embed(model, processor, [orig_img], device)
            var_emb  = embed(model, processor, [var_img],  device)
            sim = F.cosine_similarity(orig_emb, var_emb).item()

            rows.append({
                "style":    style,
                "frame":    orig_stem,
                "strength": strength,
                "scale":    scale,
                "cosine_sim": round(sim, 6),
            })

        print(f"  {style}: {sum(1 for r in rows if r['style'] == style)} entries")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["style", "frame", "strength", "scale", "cosine_sim"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows → {out_path}")

    if not rows:
        return

    strengths = sorted(set(r["strength"] for r in rows))
    scales    = sorted(set(r["scale"]    for r in rows))

    agg: dict[tuple, list] = defaultdict(list)
    for r in rows:
        agg[(r["strength"], r["scale"])].append(r["cosine_sim"])

    col_w = 8
    col_label = "str/scale"
    header = f"{col_label:<10}" + "".join(f"{c:>{col_w}.2f}" for c in scales)
    print(f"\nMean DINO cosine similarity  ({len(common_frames)} frames × {len(frames_per_style)} styles, balanced)\n"
          f"  ↑ higher = more similar to original photo\n")
    print(header)
    print("─" * len(header))
    for s in strengths:
        row_str = f"{s:<10.2f}"
        for c in scales:
            vals = agg[(s, c)]
            row_str += f"{sum(vals)/len(vals):>{col_w}.4f}" if vals else f"{'—':>{col_w}}"
        print(row_str)

    best = max(agg, key=lambda k: sum(agg[k]) / len(agg[k]))
    best_sim = sum(agg[best]) / len(agg[best])
    print(f"\nBest cell: strength={best[0]:.2f}, scale={best[1]:.2f}  (mean sim={best_sim:.4f})")


if __name__ == "__main__":
    main()
