import torch
from PIL import Image
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from huggingface_hub import hf_hub_download


class ImageStyleInference:
   
    def __init__(self,):

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
                download_source='huggingface',origin_file_pattern="text_encoder/model*.safetensors",
                **vram_config),
                ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509", 
                download_source='huggingface',origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
                **vram_config),
            ],
            tokenizer_config=None,
            processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509", 
            download_source='huggingface',origin_file_pattern="processor/"),
            vram_limit=28,
        )

        telestyle_image= hf_hub_download(repo_id="Tele-AI/TeleStyle", filename="weights/diffsynth_Qwen-Image-Edit-2509-telestyle.safetensors")

        speedup = hf_hub_download(repo_id="Tele-AI/TeleStyle", filename="weights/diffsynth_Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors")
        #https://huggingface.co/lightx2v/Qwen-Image-Lightning converted to diffsynth format

        self.pipe.load_lora(self.pipe.dit, telestyle_image)
        self.pipe.load_lora(self.pipe.dit, speedup)

    def inference(self,
        prompt,
        content_img,
        style_img,
        seed=123,
        num_inference_steps=4,
        minedge=768,
        ):
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
            cfg_scale=1.0
        )  # lightning

        return image




if __name__ == "__main__":
    import argparse
    from itertools import groupby
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--style", required=True, help="Path to style reference image")
    parser.add_argument("--dataset-root", required=True, help="Path to BlendedMVS root folder")
    parser.add_argument("--save-dir", required=True, help="Path to output directory")
    parser.add_argument("--home-backup-dir", default=None, help="Home directory mirror for backup copies")
    args = parser.parse_args()

    import shutil

    style_ref = args.style
    dataset_root = Path(args.dataset_root)
    style_name = Path(style_ref).stem
    save_dir = Path(args.save_dir) / style_name
    home_backup_dir = Path(args.home_backup_dir) / style_name if args.home_backup_dir else None
    style_img = Image.open(style_ref).convert("RGB")

    prompt = 'Style Transfer the style of Figure 2 to Figure 1, and keep the content and characteristics of Figure 1.'

    # Collect all frames across all scenes, sorted for deterministic order.
    # Skip masked images (stem ends with "masked") and keep every 5th non-masked frame per scene.
    all_raw = sorted(dataset_root.glob("*/blended_images/*.jpg"))
    all_frames = []
    for _, scene_iter in groupby(all_raw, key=lambda f: f.parent.parent):
        non_masked = [f for f in scene_iter if not f.stem.endswith("masked")]
        all_frames.extend(non_masked[::5])
    total = len(all_frames)
    print(f"Found {total} frames under {dataset_root} (every 5th per scene, from {len(all_raw)} total)")

    done = skipped = 0
    inference_engine = None  # load only if there is work to do

    for frame_path in all_frames:
        rel = frame_path.relative_to(dataset_root)
        out_path = save_dir / rel.parent / (rel.stem + "_result.png")

        if out_path.exists():
            skipped += 1
            continue

        if inference_engine is None:
            inference_engine = ImageStyleInference()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        content_img = Image.open(frame_path).convert("RGB")

        with torch.no_grad():
            generated_image = inference_engine.inference(
                prompt, content_img, style_img, seed=123, num_inference_steps=4, minedge=768
            )

        generated_image.save(out_path)
        if home_backup_dir is not None:
            home_out = home_backup_dir / rel.parent / (rel.stem + "_result.png")
            home_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(out_path, home_out)
        done += 1
        print(f"[{done + skipped}/{total}] saved {out_path}")


    print(f"Done. {done} processed, {skipped} skipped (already existed).")
            
