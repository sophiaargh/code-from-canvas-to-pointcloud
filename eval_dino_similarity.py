"""
Cosine similarity in DINO space between original photos and each image variant:
  - stylized                      (TeleStyle output)
  - destylized_reverse_telestyle  (TeleStyle reversal)
  - destylized_controlnet         (ControlNet img2img + depth)

Only processes frames where both the original and the variant exist, so it works
even if destylization has only been run on a subset of styles/scenes.

Outputs a CSV and prints a mean-similarity summary table.

Usage:
    python eval_dino_similarity.py \
        --data-root /scratch/izar/silly/BlendedMVS \
        --styles watercolor oil_painting impressionism engraving \
        --max-scenes 10 \
        --out evaluation_results/dino_similarity.csv
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


STYLES = ["watercolor", "oil_painting", "impressionism", "engraving"]

# method name → path template relative to data_root (use {style} placeholder)
METHODS = {
    "stylized":                     "telestyle_output/{style}",
    "destylized_reverse_telestyle": "telestyle_output/destylized_reverse_telestyle/{style}",
    "destylized_controlnet":        "telestyle_output/destylized_controlnet/{style}",
}


@torch.no_grad()
def embed(model, processor, images: list, device, batch_size: int = 32) -> torch.Tensor:
    """Return L2-normalised CLS-token embeddings, shape (N, D), on CPU."""
    out = []
    for i in range(0, len(images), batch_size):
        inputs = processor(images=images[i:i + batch_size], return_tensors="pt").to(device)
        cls = model(**inputs).last_hidden_state[:, 0]
        out.append(F.normalize(cls, dim=-1).cpu())
    return torch.cat(out, dim=0)


def collect_frames(data_root: Path, max_scenes: int | None) -> list[tuple[str, str]]:
    """Return (scene_name, frame_stem) pairs matching the TeleStyle sampling strategy."""
    scenes = sorted((data_root / "renamed").glob("scene_*"))
    if max_scenes:
        scenes = scenes[:max_scenes]
    frames = []
    for scene_dir in scenes:
        non_masked = sorted(
            f for f in (scene_dir / "blended_images").glob("*.jpg")
            if not f.stem.endswith("masked")
        )
        if len(non_masked) > 150:
            continue
        for f in non_masked[::5]:
            frames.append((scene_dir.name, f.stem))
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/scratch/izar/silly/BlendedMVS")
    parser.add_argument("--styles", nargs="+", default=STYLES)
    parser.add_argument("--max-scenes", type=int, default=None,
                        help="Cap number of scenes (useful for quick tests)")
    parser.add_argument("--scene", default=None,
                        help="Evaluate only this scene (bypasses the >150 filter), e.g. scene_0")
    parser.add_argument("--out", default="evaluation_results/dino_similarity.csv")
    parser.add_argument("--model", default="facebook/dinov2-large")
    parser.add_argument("--tuning-dir", default=None,
                        help="Root of ControlNet tuning output (e.g. ~/destylize_tuning). "
                             "Used instead of destylized_controlnet when present.")
    parser.add_argument("--tuning-strength", type=float, default=0.20)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading {args.model} ...")
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float16).to(device)
    model.eval()

    if args.scene:
        # Bypass the >150 filter and evaluate a specific scene directly
        scene_dir = data_root / "renamed" / args.scene
        non_masked = sorted(
            f for f in (scene_dir / "blended_images").glob("*.jpg")
            if not f.stem.endswith("masked")
        )
        frames = [(args.scene, f.stem) for f in non_masked[::5]]
    else:
        frames = collect_frames(data_root, args.max_scenes)
    print(f"Found {len(frames)} candidate frames\n")

    rows = []
    for scene, stem in frames:
        original_path = data_root / "renamed" / scene / "blended_images" / f"{stem}.jpg"
        if not original_path.exists():
            continue

        orig_img = Image.open(original_path).convert("RGB")
        orig_emb = embed(model, processor, [orig_img], device)  # (1, D)

        for style in args.styles:
            for method, template in METHODS.items():
                # For ControlNet destylized, prefer tuning dir at the specified strength
                if method == "destylized_controlnet" and args.tuning_dir:
                    strength_tag = f"s{args.tuning_strength:.2f}"
                    variant_path = (
                        Path(args.tuning_dir).expanduser() / style
                        / f"{stem}_{strength_tag}.png"
                    )
                else:
                    variant_path = (
                        data_root / template.format(style=style)
                        / scene / "blended_images" / f"{stem}_result.png"
                    )

                if not variant_path.exists():
                    continue

                var_img = Image.open(variant_path).convert("RGB")
                var_emb = embed(model, processor, [var_img], device)  # (1, D)
                sim = F.cosine_similarity(orig_emb, var_emb).item()

                rows.append({
                    "scene":      scene,
                    "frame":      stem,
                    "style":      style,
                    "method":     method,
                    "cosine_sim": round(sim, 6),
                })

        print(f"  {scene}/{stem}: {len(rows)} rows so far")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scene", "frame", "style", "method", "cosine_sim"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows → {out_path}")

    # Summary table — matched frames only: stylized mean computed over the same
    # frames as each destylized method to ensure a fair comparison.
    by_key: dict[tuple, dict] = defaultdict(dict)
    for row in rows:
        key = (row["scene"], row["frame"], row["style"])
        by_key[key][row["method"]] = row["cosine_sim"]

    destyl_sims: dict[tuple, list] = defaultdict(list)
    matched_stylized: dict[tuple, list] = defaultdict(list)
    for (scene, frame, style), methods in by_key.items():
        if "stylized" not in methods:
            continue
        for method, sim in methods.items():
            if method == "stylized":
                continue
            destyl_sims[(style, method)].append(sim)
            matched_stylized[(style, method)].append(methods["stylized"])

    print(f"\n{'Style':<20} {'Method':<35} {'Stylized (matched)':>20} {'Destylized':>12} {'N':>5}")
    print("-" * 96)
    for (style, method) in sorted(destyl_sims.keys()):
        d = destyl_sims[(style, method)]
        s = matched_stylized[(style, method)]
        n = len(d)
        print(f"{style:<20} {method:<35} {sum(s)/len(s):>20.4f} {sum(d)/len(d):>12.4f} {n:>5}")


if __name__ == "__main__":
    main()
