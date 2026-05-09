import torch
from PIL import Image
from pathlib import Path
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from huggingface_hub import hf_hub_download


class ImageStyleInference:

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
        speedup_lora = hf_hub_download(repo_id="Tele-AI/TeleStyle", filename="weights/diffsynth_Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors")

        self.pipe.load_lora(self.pipe.dit, telestyle_lora)
        self.pipe.load_lora(self.pipe.dit, speedup_lora)

    def inference(self, prompt, content_img, style_img, seed=123, num_inference_steps=4, minedge=768):
        w, h = content_img.size
        minedge = minedge - minedge % 16

        if w > h:
            r = w / h
            h = minedge
            w = int(h * r) - int(h * r) % 16
        else:
            r = h / w
            w = minedge
            h = int(w * r) - int(w * r) % 16

        if content_img.size != (w, h):
            content_img = content_img.resize((w, h))
        if style_img.size != (minedge, minedge):
            style_img = style_img.resize((minedge, minedge))

        image = self.pipe(
            prompt,
            edit_image=[content_img, style_img],
            seed=seed,
            num_inference_steps=num_inference_steps,
            height=h,
            width=w,
            edit_image_auto_resize=False,
            cfg_scale=1.0,
        )

        return image


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--styles-dir", required=True, help="Folder containing style reference images")
    parser.add_argument("--dataset-root", required=True, help="Path to BlendedMVS root folder")
    parser.add_argument("--save-dir", required=True, help="Path to output directory")
    args = parser.parse_args()

    styles_dir = Path(args.styles_dir)
    dataset_root = Path(args.dataset_root)
    save_dir = Path(args.save_dir)

    style_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    style_paths = sorted(p for p in styles_dir.iterdir() if p.suffix.lower() in style_extensions)
    if not style_paths:
        raise RuntimeError(f"No style images found in {styles_dir}")

    first_scene = sorted(dataset_root.iterdir())[0]
    content_candidates = sorted((first_scene / "blended_images").glob("*.jpg"))
    # skip masked variants
    content_candidates = [p for p in content_candidates if "_masked" not in p.name]
    if not content_candidates:
        raise RuntimeError(f"No content images found in {first_scene / 'blended_images'}")
    content_path = content_candidates[0]

    print(f"Content image: {content_path}")
    print(f"Styles to apply: {[p.name for p in style_paths]}")

    prompt = "Style Transfer the style of Figure 2 to Figure 1, and keep the content and characteristics of Figure 1."

    engine = ImageStyleInference()

    content_img = Image.open(content_path).convert("RGB")
    save_dir.mkdir(parents=True, exist_ok=True)

    for style_path in style_paths:
        style_name = style_path.stem
        out_path = save_dir / f"{content_path.stem}_{style_name}.png"

        if out_path.exists():
            print(f"[skip] {out_path.name} already exists")
            continue

        style_img = Image.open(style_path).convert("RGB")

        with torch.no_grad():
            result = engine.inference(prompt, content_img, style_img, seed=123, num_inference_steps=4, minedge=768)

        result.save(out_path)
        print(f"[done] saved {out_path}")

    print("All styles processed.")
