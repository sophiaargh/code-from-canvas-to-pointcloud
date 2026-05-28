"""
Generates the (img2img strength × ControlNet conditioning scale) grid for one
scene across all four styles.  Load the model once, sweep all (strength, scale)
combinations per frame to avoid redundant model reloads.

Output naming:
  {out_dir}/{style}/{orig_stem}_result_s{strength:.2f}_c{scale:.2f}.png

Evaluate afterwards with:
  python eval_controlnet_grid.py --tuning-dir {out_dir} --scene {scene}
"""
import argparse
import sys
from itertools import product
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from controlnet_destylize_inference import ControlNetDestylizeInference


STRENGTHS = [0.20, 0.35, 0.50, 0.65]
SCALES    = [0.6, 0.8, 1.0, 1.3]
STYLES    = ["watercolor", "oil_painting", "impressionism", "engraving"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stylized-root", required=True,
                        help="telestyle_output/ root (contains {style}/ subdirs)")
    parser.add_argument("--scene", required=True,
                        help="Scene name, e.g. scene_0000")
    parser.add_argument("--out-dir", required=True,
                        help="Root output directory for tuning images")
    parser.add_argument("--styles",     nargs="+", type=str,   default=STYLES)
    parser.add_argument("--strengths",  nargs="+", type=float, default=STRENGTHS)
    parser.add_argument("--scales",     nargs="+", type=float, default=SCALES)
    parser.add_argument("--max-frames", type=int,  default=8,
                        help="Max frames per style to use (evenly sampled). Default: 8")
    args = parser.parse_args()

    stylized_root = Path(args.stylized_root)
    out_dir = Path(args.out_dir).expanduser()

    # Collect frames from the first available style directory
    frames_per_style: dict[str, list[Path]] = {}
    for style in args.styles:
        scene_img_dir = stylized_root / style / args.scene / "blended_images"
        frames = sorted(scene_img_dir.glob("*_result.png"))
        if not frames:
            print(f"[WARN] no frames for {style}/{args.scene}, skipping style")
            continue
        # Subsample evenly so tuning stays fast regardless of scene size
        if len(frames) > args.max_frames:
            step = len(frames) / args.max_frames
            frames = [frames[int(i * step)] for i in range(args.max_frames)]
        frames_per_style[style] = frames

    if not frames_per_style:
        raise FileNotFoundError(f"No stylized frames found under {stylized_root} for scene {args.scene}")

    n_frames = max(len(v) for v in frames_per_style.values())
    total = sum(len(v) for v in frames_per_style.values()) * len(args.strengths) * len(args.scales)
    print(f"Scene: {args.scene} | ~{n_frames} frames/style | "
          f"{len(args.strengths)} strengths × {len(args.scales)} scales | "
          f"total images: {total}")

    engine = None
    done = skipped = 0

    for style, frames in frames_per_style.items():
        for frame_path in frames:
            orig_stem = frame_path.stem.replace("_result", "")  # "00000001"

            content_img = None  # load once per frame, reuse across grid cells
            for strength, scale in product(args.strengths, args.scales):
                tag = f"s{strength:.2f}_c{scale:.2f}"
                out_path = out_dir / style / f"{orig_stem}_result_{tag}.png"

                if out_path.exists():
                    skipped += 1
                    continue

                if engine is None:
                    engine = ControlNetDestylizeInference()
                if content_img is None:
                    content_img = Image.open(frame_path).convert("RGB")

                out_path.parent.mkdir(parents=True, exist_ok=True)
                with torch.no_grad():
                    result = engine.inference(
                        content_img,
                        seed=123,
                        strength=strength,
                        controlnet_conditioning_scale=scale,
                    )
                result.save(out_path)
                done += 1
                if done % 10 == 0 or done == 1:
                    print(f"[{done + skipped}/{total}] {style} {orig_stem} s={strength:.2f} c={scale:.2f}")

    print(f"\nDone. {done} generated, {skipped} skipped.")


if __name__ == "__main__":
    main()
