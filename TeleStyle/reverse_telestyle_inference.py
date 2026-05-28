import torch
from PIL import Image
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from huggingface_hub import hf_hub_download


class ReverseTeleStyleInference:
    """
    Reverses stylization by running TeleStyle forward with the original photo as
    style reference.  Figure 1 = stylized image, Figure 2 = original photo → the
    model transfers the photorealistic style of the original back onto the content.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_models()

    def _load_models(self):
        vram_config = {
            "offload_dtype": torch.bfloat16,
            "offload_device": "cpu",
            "onload_dtype": torch.bfloat16,
            "onload_device": "cpu",
            "preparing_dtype": torch.bfloat16,
            "preparing_device": "cuda",
            "computation_dtype": torch.bfloat16,
            "computation_device": "cuda",
        }

        self.pipe = QwenImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=[
                ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509",
                download_source='huggingface',
                origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
                **vram_config),
                ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509",
                download_source='huggingface', origin_file_pattern="text_encoder/model*.safetensors",
                **vram_config),
                ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509",
                download_source='huggingface', origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
                **vram_config),
            ],
            tokenizer_config=None,
            processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509",
            download_source='huggingface', origin_file_pattern="processor/"),
            vram_limit=28,
        )

        telestyle_lora = hf_hub_download(repo_id="Tele-AI/TeleStyle", filename="weights/diffsynth_Qwen-Image-Edit-2509-telestyle.safetensors")
        speedup = hf_hub_download(repo_id="Tele-AI/TeleStyle", filename="weights/diffsynth_Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors")
        self.pipe.load_lora(self.pipe.dit, telestyle_lora)
        self.pipe.load_lora(self.pipe.dit, speedup)

    def inference(self,
        stylized_img,
        photo_ref_img,
        seed=123,
        num_inference_steps=4,
        minedge=768,
        ):
        w, h = stylized_img.size
        minedge = minedge - minedge % 16

        if w > h:
            r = w / h
            h = minedge
            w = int(h * r) - int(h * r) % 16
        else:
            r = h / w
            w = minedge
            h = int(w * r) - int(w * r) % 16

        if stylized_img.size != (w, h):
            stylized_img = stylized_img.resize((w, h))
        if photo_ref_img.size != (minedge, minedge):
            photo_ref_img = photo_ref_img.resize((minedge, minedge))

        image = self.pipe(
            'Style Transfer the style of Figure 2 to Figure 1, and keep the content and characteristics of Figure 1.',
            edit_image=[stylized_img, photo_ref_img],
            seed=seed,
            num_inference_steps=num_inference_steps,
            height=h,
            width=w,
            edit_image_auto_resize=False,
            cfg_scale=1.0
        )

        return image


if __name__ == "__main__":
    import argparse
    import shutil
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--stylized-dir", required=True,
                        help="Per-style stylized folder (e.g. telestyle_output/watercolor/)")
    parser.add_argument("--original-dir", required=True,
                        help="BlendedMVS renamed root (contains scene_*/blended_images/*.jpg)")
    parser.add_argument("--save-dir", required=True,
                        help="Root output directory (style subfolder appended automatically)")
    parser.add_argument("--home-backup-dir", default=None,
                        help="Home directory mirror for backup copies")
    parser.add_argument("--scene", default=None,
                        help="Process only this scene, e.g. scene_0000. Omit for all scenes.")
    args = parser.parse_args()

    stylized_dir = Path(args.stylized_dir)
    style_name = stylized_dir.name
    original_dir = Path(args.original_dir)
    save_dir = Path(args.save_dir) / style_name
    home_backup_dir = Path(args.home_backup_dir) / style_name if args.home_backup_dir else None

    all_frames = sorted(stylized_dir.glob("*/blended_images/*_result.png"))
    if args.scene:
        all_frames = [f for f in all_frames if f.parent.parent.name == args.scene]
    total = len(all_frames)
    print(f"Found {total} stylized frames under {stylized_dir}")

    done = skipped = missing = 0
    inference_engine = None

    for frame_path in all_frames:
        rel = frame_path.relative_to(stylized_dir)
        out_path = save_dir / rel

        if out_path.exists():
            skipped += 1
            continue

        # "00000001_result.png" → "00000001.jpg"
        original_stem = frame_path.stem.replace("_result", "")
        scene_name = frame_path.parent.parent.name
        original_path = original_dir / scene_name / "blended_images" / (original_stem + ".jpg")

        if not original_path.exists():
            print(f"[WARN] original not found: {original_path}")
            missing += 1
            continue

        if inference_engine is None:
            inference_engine = ReverseTeleStyleInference()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        stylized_img = Image.open(frame_path).convert("RGB")
        photo_ref_img = Image.open(original_path).convert("RGB")

        with torch.no_grad():
            generated_image = inference_engine.inference(stylized_img, photo_ref_img, seed=123)

        generated_image.save(out_path)
        if home_backup_dir is not None:
            home_out = home_backup_dir / rel
            home_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(out_path, home_out)
        done += 1
        print(f"[{done + skipped}/{total}] saved {out_path}")

    print(f"Done. {done} processed, {skipped} skipped, {missing} missing originals.")
