"""
For a set of scenes, sweeps the full (strength × scale) grid and computes
DINO cosine similarity to the original for both:
  - stylized images (telestyle output)  — one value per frame, independent of grid
  - ControlNet destylized images        — one value per (strength, scale, frame)

Generates ControlNet images on the fly if they don't exist yet.
Prints a heatmap of medians matching the single-scene grid format, with the
stylized baseline as a reference row.

Usage:
    python eval_destylize_comparison.py \
        --data-root /scratch/izar/silly/BlendedMVS \
        --scenes scene_0 scene_1 scene_2 ... \
        --max-frames 8 \
        --out evaluation_results/destylize_comparison_grid.csv
"""
import argparse
import csv
import statistics
import sys
import torch
import torch.nn.functional as F
from collections import defaultdict
from itertools import product
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

sys.path.insert(0, str(Path(__file__).parent / "TeleStyle"))
from controlnet_destylize_inference import ControlNetDestylizeInference


STYLES    = ["watercolor", "oil_painting", "impressionism", "engraving"]
STRENGTHS = [0.20, 0.35, 0.50, 0.65]
SCALES    = [0.60, 0.80, 1.00, 1.30]


@torch.no_grad()
def embed_batch(model, processor, images, device, batch_size=16):
    out = []
    for i in range(0, len(images), batch_size):
        inputs = processor(images=images[i:i + batch_size], return_tensors="pt").to(device)
        cls = model(**inputs).last_hidden_state[:, 0]
        out.append(F.normalize(cls, dim=-1).cpu())
    return torch.cat(out, dim=0)


def pick_frames(frames, max_frames):
    if len(frames) <= max_frames:
        return frames
    step = len(frames) / max_frames
    return [frames[int(i * step)] for i in range(max_frames)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/scratch/izar/silly/BlendedMVS")
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--strengths", nargs="+", type=float, default=STRENGTHS)
    parser.add_argument("--scales",    nargs="+", type=float, default=SCALES)
    parser.add_argument("--out", default="evaluation_results/destylize_comparison_grid.csv")
    parser.add_argument("--model", default="facebook/dinov2-large")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    orig_root = data_root / "renamed"
    styl_root = data_root / "telestyle_output"
    out_path  = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model} ...")
    processor = AutoImageProcessor.from_pretrained(args.model)
    dino = AutoModel.from_pretrained(args.model, torch_dtype=torch.float16).to(device)
    dino.eval()

    cn_engine = None
    rows = []       # one row per (scene, style, frame, strength, scale)
    styl_rows = []  # one row per (scene, style, frame) — baseline

    for scene in args.scenes:
        orig_img_dir = orig_root / scene / "blended_images"
        orig_frames  = sorted(f for f in orig_img_dir.glob("*.jpg")
                              if not f.stem.endswith("masked"))
        orig_frames  = pick_frames(orig_frames, args.max_frames)

        if not orig_frames:
            print(f"[SKIP] {scene}: no original frames found")
            continue

        print(f"\n{'═'*60}\nScene: {scene} — {len(orig_frames)} frames\n{'═'*60}")

        for style in STYLES:
            styl_img_dir = styl_root / style / scene / "blended_images"

            # Embed originals and stylized images once per (scene, style)
            orig_embs, styl_embs, valid_frames = [], [], []
            for orig_path in orig_frames:
                stem      = orig_path.stem
                styl_path = styl_img_dir / f"{stem}_result.png"
                if not styl_path.exists():
                    print(f"  [MISS stylized] {style}/{scene}/{stem}")
                    continue
                orig_embs.append(Image.open(orig_path).convert("RGB"))
                styl_embs.append(Image.open(styl_path).convert("RGB"))
                valid_frames.append(orig_path)

            if not valid_frames:
                continue

            all_imgs  = orig_embs + styl_embs
            all_embs  = embed_batch(dino, processor, all_imgs, device)
            o_embs    = all_embs[:len(valid_frames)]
            s_embs    = all_embs[len(valid_frames):]
            styl_sims = F.cosine_similarity(o_embs, s_embs).tolist()

            for i, orig_path in enumerate(valid_frames):
                styl_rows.append({
                    "scene": scene, "style": style, "frame": orig_path.stem,
                    "styl_sim": round(styl_sims[i], 6),
                })

            styl_mean = sum(styl_sims) / len(styl_sims)
            print(f"  {style:<15} stylized={styl_mean:.4f}")

            # Sweep grid for this (scene, style)
            for strength, scale in product(args.strengths, args.scales):
                destyl_root = (data_root / "telestyle_output" /
                               f"destylized_controlnet_s{strength:.2f}_c{scale:.2f}")

                cn_sims = []
                for i, orig_path in enumerate(valid_frames):
                    stem        = orig_path.stem
                    styl_path   = styl_img_dir / f"{stem}_result.png"
                    destyl_path = destyl_root / style / scene / "blended_images" / f"{stem}_result.png"

                    if not destyl_path.exists():
                        if cn_engine is None:
                            print("  Loading ControlNet engine...")
                            cn_engine = ControlNetDestylizeInference()
                        destyl_path.parent.mkdir(parents=True, exist_ok=True)
                        styl_img = Image.open(styl_path).convert("RGB")
                        with torch.no_grad():
                            result = cn_engine.inference(
                                styl_img, seed=123,
                                strength=strength,
                                controlnet_conditioning_scale=scale,
                            )
                        result.save(destyl_path)

                    destyl_img = Image.open(destyl_path).convert("RGB")
                    d_emb = embed_batch(dino, processor, [destyl_img], device)
                    cn_sims.append(F.cosine_similarity(o_embs[i:i+1], d_emb).item())

                    rows.append({
                        "scene": scene, "style": style, "frame": orig_path.stem,
                        "strength": strength, "scale": scale,
                        "styl_sim": round(styl_sims[i], 6),
                        "cn_sim":   round(cn_sims[-1], 6),
                    })

                cn_mean = sum(cn_sims) / len(cn_sims)
                print(f"    s={strength:.2f} c={scale:.2f} → cn={cn_mean:.4f}  Δ={cn_mean-styl_mean:+.4f}")

    # Save CSV
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scene","style","frame","strength","scale","styl_sim","cn_sim"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows → {out_path}")

    # Print heatmap with medians
    if not rows:
        return

    strengths = sorted(set(r["strength"] for r in rows))
    scales    = sorted(set(r["scale"]    for r in rows))

    grid_med  = defaultdict(list)
    styl_all  = [r["styl_sim"] for r in styl_rows]
    for r in rows:
        grid_med[(r["strength"], r["scale"])].append(r["cn_sim"])

    n_scenes = len(set(r["scene"] for r in rows))
    n_styles = len(set(r["style"] for r in rows))
    col_w = 8
    header = f"{'str/scale':<10}" + "".join(f"{c:>{col_w}.2f}" for c in scales)

    print(f"\nMedian DINO cosine similarity  ({n_scenes} scenes × {n_styles} styles, balanced)")
    print(f"  ↑ higher = more similar to original photo\n")
    print(header)
    print("─" * len(header))

    styl_median = statistics.median(styl_all)
    print(f"{'stylized':<10}" + "".join(f"{styl_median:>{col_w}.4f}" for _ in scales)
          + f"   ← baseline (n={len(styl_all)})")
    for s in strengths:
        row_str = f"{s:<10.2f}"
        for c in scales:
            vals = grid_med[(s, c)]
            row_str += f"{statistics.median(vals):>{col_w}.4f}" if vals else f"{'—':>{col_w}}"
        print(row_str)

    best = max(grid_med, key=lambda k: statistics.median(grid_med[k]))
    print(f"\nBest cell: strength={best[0]:.2f}, scale={best[1]:.2f}  "
          f"(median={statistics.median(grid_med[best]):.4f}  vs baseline {styl_median:.4f})")


if __name__ == "__main__":
    main()
