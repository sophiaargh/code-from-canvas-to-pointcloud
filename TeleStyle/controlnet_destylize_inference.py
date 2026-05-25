import torch
import numpy as np
from PIL import Image
from transformers import pipeline as hf_pipeline
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler


class ControlNetDestylizeInference:
    """
    Destylizes an image by combining img2img (content anchor) with ControlNet-depth
    (structural guidance).  The stylized image is used as the init image so the
    model stays close to the original scene; depth conditioning preserves geometry;
    the photorealistic prompt pushes the style away from artistic rendering.
    """

    PROMPT = (
        "a photorealistic photograph, high quality, realistic lighting, "
        "detailed texture, sharp focus, natural colors"
    )
    NEG_PROMPT = (
        "painting, artwork, watercolor, oil painting, sketch, drawing, "
        "artistic, stylized, anime, cartoon, low quality, blurry"
    )

    def __init__(self):
        self._load_models()

    def _load_models(self):
        # Depth estimator on CPU to avoid VRAM conflicts with the SD pipeline
        self.depth_estimator = hf_pipeline(
            "depth-estimation",
            model="Intel/dpt-large",
        )

        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-depth",
            torch_dtype=torch.float16,
        )
        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to("cuda")

    def _get_depth_map(self, image: Image.Image) -> Image.Image:
        depth = self.depth_estimator(image)["depth"]
        depth = np.array(depth, dtype=np.float32)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255
        depth_u8 = depth.astype(np.uint8)
        return Image.fromarray(np.stack([depth_u8, depth_u8, depth_u8], axis=2))

    def inference(self,
        content_img,
        seed=123,
        num_inference_steps=20,
        strength=0.65,
        minedge=512,
        ):
        w, h = content_img.size
        minedge = minedge - minedge % 8

        if w > h:
            r = w / h
            h = minedge
            w = int(h * r) - int(h * r) % 8
        else:
            r = h / w
            w = minedge
            h = int(w * r) - int(w * r) % 8

        content_img = content_img.resize((w, h))
        depth_map = self._get_depth_map(content_img)

        generator = torch.Generator(device="cuda").manual_seed(seed)
        image = self.pipe(
            prompt=self.PROMPT,
            negative_prompt=self.NEG_PROMPT,
            image=content_img,       # img2img init — anchors scene content
            control_image=depth_map, # ControlNet condition — preserves geometry
            strength=strength,
            num_inference_steps=num_inference_steps,
            generator=generator,
            controlnet_conditioning_scale=0.8,
        ).images[0]

        return image


if __name__ == "__main__":
    import argparse
    import shutil
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--stylized-dir", required=True,
                        help="Per-style stylized folder (e.g. telestyle_output/watercolor/)")
    parser.add_argument("--save-dir", required=True,
                        help="Root output directory (style subfolder appended automatically)")
    parser.add_argument("--home-backup-dir", default=None,
                        help="Home directory mirror for backup copies")
    parser.add_argument("--scene", default=None,
                        help="Process only this scene, e.g. scene_0000. Omit for all scenes.")
    parser.add_argument("--strength", type=float, default=0.65,
                        help="img2img denoising strength (0=no change, 1=full diffusion). Default: 0.65")
    args = parser.parse_args()

    stylized_dir = Path(args.stylized_dir)
    style_name = stylized_dir.name
    save_dir = Path(args.save_dir) / style_name
    home_backup_dir = Path(args.home_backup_dir) / style_name if args.home_backup_dir else None

    all_frames = sorted(stylized_dir.glob("*/blended_images/*_result.png"))
    if args.scene:
        all_frames = [f for f in all_frames if f.parent.parent.name == args.scene]
    total = len(all_frames)
    print(f"Found {total} stylized frames under {stylized_dir}")

    done = skipped = 0
    inference_engine = None

    for frame_path in all_frames:
        rel = frame_path.relative_to(stylized_dir)
        out_path = save_dir / rel

        if out_path.exists():
            skipped += 1
            continue

        if inference_engine is None:
            inference_engine = ControlNetDestylizeInference()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        content_img = Image.open(frame_path).convert("RGB")

        with torch.no_grad():
            generated_image = inference_engine.inference(content_img, seed=123, strength=args.strength)

        generated_image.save(out_path)
        if home_backup_dir is not None:
            home_out = home_backup_dir / rel
            home_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(out_path, home_out)
        done += 1
        print(f"[{done + skipped}/{total}] saved {out_path}")

    print(f"Done. {done} processed, {skipped} skipped (already existed).")
