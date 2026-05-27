"""
Quick hyperparameter sweep over ControlNet img2img `strength` values.
Outputs all variants into a single flat folder for easy side-by-side comparison.

Usage:
    python tune_controlnet_strength.py \
        --stylized-dir /scratch/izar/silly/BlendedMVS/telestyle_output/watercolor \
        --scene scene_0 \
        --n-images 4 \
        --strengths 0.3 0.4 0.5 0.6 0.7 \
        --save-dir ~/destylize_tuning/watercolor
"""

import argparse
import torch
from pathlib import Path
from PIL import Image

from controlnet_destylize_inference import ControlNetDestylizeInference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stylized-dir", required=True)
    parser.add_argument("--scene", required=True, help="Scene to sample from, e.g. scene_0")
    parser.add_argument("--n-images", type=int, default=4, help="Number of frames to process")
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7])
    parser.add_argument("--save-dir", default="~/destylize_tuning", help="Output folder (no scratch)")
    args = parser.parse_args()

    stylized_dir = Path(args.stylized_dir)
    save_dir = Path(args.save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted((stylized_dir / args.scene / "blended_images").glob("*_result.png"))
    frames = frames[:args.n_images]
    if not frames:
        raise FileNotFoundError(f"No *_result.png frames found in {stylized_dir / args.scene / 'blended_images'}")
    print(f"Tuning on {len(frames)} frame(s): {[f.name for f in frames]}")
    print(f"Strengths: {args.strengths}")
    print(f"Output: {save_dir}\n")

    engine = ControlNetDestylizeInference()

    for frame_path in frames:
        content_img = Image.open(frame_path).convert("RGB")
        stem = frame_path.stem.replace("_result", "")

        for strength in args.strengths:
            out_path = save_dir / f"{stem}_s{strength:.2f}.png"
            if out_path.exists():
                print(f"  [skip] {out_path.name}")
                continue

            with torch.no_grad():
                result = engine.inference(content_img, seed=123, strength=strength)

            result.save(out_path)
            print(f"  [done] {out_path.name}")

    print(f"\nDone. Results in {save_dir}")


if __name__ == "__main__":
    main()
