from dataclasses import dataclass
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms as tvf
import numpy as np

# Assuming your models.py can be imported relatively or absolutely
from .models import get_model, infer

import os
import torch
from torch.utils.data import Dataset
from mapanything.utils.image import load_images

from datetime import datetime

def timed_print(msg, **kwargs):
    # Formats as YYYY-MM-DD HH:MM:SS
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True, **kwargs)

class PairedStyleDataset(Dataset):
    def __init__(self, style: str):
        self.real_dir = "/scratch/izar/silly/BlendedMVS/renamed/"
        self.style_dir = f"/scratch/izar/silly/BlendedMVS/telestyle_output/{style}/"
        self.pairs = []

        # --- Build Dataset Paths ---
        scenes = sorted(os.listdir(self.real_dir))
        for scene in scenes:
            if not scene.startswith("scene_"):
                continue
            
            # Filter rule: Skip any scene folder index starting with '1'
            if scene.split("_")[1].startswith("1"):
                continue

            real_scene_path = os.path.join(self.real_dir, scene, "blended_images")
            style_scene_path = os.path.join(self.style_dir, scene, "blended_images")

            if not os.path.exists(real_scene_path) or not os.path.exists(style_scene_path):
                continue

            # Counter to limit photos to 50 per scene
            scene_photo_count = 0

            for img_name in sorted(os.listdir(real_scene_path)):
                # Break early if we already hit the limit for this scene
                if scene_photo_count >= 25:
                    break

                if not img_name.endswith(".jpg") or img_name.endswith("_masked.jpg"):
                    continue

                stem = os.path.splitext(img_name)[0]
                style_img_path = os.path.join(style_scene_path, f"{stem}_result.png")
                real_img_path = os.path.join(real_scene_path, img_name)

                if os.path.exists(style_img_path):
                    self.pairs.append((real_img_path, style_img_path))
                    scene_photo_count += 1  # Only count it if the pair actually exists

        print(f"Loaded {len(self.pairs)} paired image paths for style: '{style}'")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        real_path, style_path = self.pairs[idx]
        
        # Load directly using MapAnything's native loader logic
        real_views = load_images([real_path], resolution_set=518, norm_type="dinov2", patch_size=14)
        style_views = load_images([style_path], resolution_set=518, norm_type="dinov2", patch_size=14)

        return real_views, style_views

def mapanything_collate_fn(batch):
    """
    Combines B separate single-image view lists into a single batched view list
    matching MapAnything's expected layout: [ {'img': Tensor([B, C, H, W]), ...} ]
    """
    real_dicts = [item[0][0] for item in batch]
    style_dicts = [item[1][0] for item in batch]

    def merge(dict_list):
        batched_img = torch.cat([d["img"] for d in dict_list], dim=0)
        batched_shape = np.concatenate([d["true_shape"] for d in dict_list], axis=0)
        
        # Extract the string "dinov2"
        norm_type_string = dict_list[0]["data_norm_type"][0]

        return {
            "img": batched_img,
            "true_shape": batched_shape,
            "idx": [d["idx"] for d in dict_list],
            "instance": [d["instance"] for d in dict_list],
            # Wrap it in a single list so that data_norm_type[0] extracts "dinov2"
            "data_norm_type": [norm_type_string],  
        }

    return [merge(real_dicts)], [merge(style_dicts)]

def train_adapter():
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", type=str, required=True, choices=['watercolor', 'oil_painting', 'engraving', 'impressionism'])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4) # Lowered default for finetuning stability
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--accumulation_steps", type=int, default=8) # New: Simulates batch size of 8
    parser.add_argument("--encoder_block_prefix", type=str, default="encoder.model.blocks")
    parser.add_argument("--norm_num_blocks", type=int, default=3)
    parser.add_argument("--norm_from_end", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timed_print("Loading Teacher model (no hooks)...")
    teacher_model, _ = get_model(
        checkpoint="facebook/map-anything",
    )
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    timed_print("Loading Student model (with hooks)...")
    student_model, _ = get_model(
        checkpoint="facebook/map-anything",
        encoder_block_prefix=args.encoder_block_prefix,
        norm_num_blocks=args.norm_num_blocks,
        norm_from_end=args.norm_from_end,
        norm_affine=True, 
    )
    student_model.eval()
    for param in student_model.parameters():
        param.requires_grad = False

    dataset = PairedStyleDataset(style=args.style)
    
    # Divide total samples by 8 to define dynamic sub-epochs
    num_samples_per_epoch = max(1, len(dataset) // 8)
    timed_print(f"Total dataset size: {len(dataset)}. Sampling {num_samples_per_epoch} random unique images per epoch split.")
    
    sampler = RandomSampler(dataset, replacement=True, num_samples=num_samples_per_epoch)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        sampler=sampler, 
        collate_fn=mapanything_collate_fn
    )

    timed_print("Running a dummy forward pass to allocate affine weights...")

    # 1. Grab a real batch from the dataloader
    batch_real, batch_style = next(iter(dataloader))

    with torch.no_grad():
        # Run student on style images to trigger lazy initialization
        _ = infer(student_model, batch_style)

    if isinstance(batch_style, list) and isinstance(batch_style[0], dict):
        batch_style_cuda = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                            for k, v in view.items()} for view in batch_style]
    elif isinstance(batch_style, dict):
        batch_style_cuda = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch_style.items()}
    else:
        batch_style_cuda = batch_style.to(device)

    # 2. Call the student model DIRECTLY (no infer(), no torch.no_grad())
    # This forces standard initialization of your InstanceNorm layers
    _ = student_model(batch_style_cuda)

    # 3. Extract the parameters and manually recreate them
    trainable_params = []
    if hasattr(student_model, "_norm_hook_triples"):
        for name, hook, handle in student_model._norm_hook_triples:
            if hook._norm1d is not None and hook._norm1d.weight is not None:
                # 1. Break the tensor connection to the dummy forward pass
                w_clean = hook._norm1d.weight.data.clone()
                b_clean = hook._norm1d.bias.data.clone()
                
                # 2. Assign them back as fresh parameters 
                hook._norm1d.weight = torch.nn.Parameter(w_clean, requires_grad=True)
                hook._norm1d.bias = torch.nn.Parameter(b_clean, requires_grad=True)
                
                trainable_params.extend([hook._norm1d.weight, hook._norm1d.bias])
                
            if hook._norm2d is not None and hook._norm2d.weight is not None:
                # Do the same for 2D norm layers
                w_clean_2d = hook._norm2d.weight.data.clone()
                b_clean_2d = hook._norm2d.bias.data.clone()
                
                hook._norm2d.weight = torch.nn.Parameter(w_clean_2d, requires_grad=True)
                hook._norm2d.bias = torch.nn.Parameter(b_clean_2d, requires_grad=True)
                
                trainable_params.extend([hook._norm2d.weight, hook._norm2d.bias])

    # Double-check that we actually captured parameters
    if not trainable_params:
        raise ValueError("No trainable adapter parameters found! Check if affine=True.")

    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    criterion = nn.L1Loss()

    timed_print(f"Beginning adapter training for {args.style} with {args.norm_num_blocks} from {'end' if args.norm_from_end else 'front'} ...")

    folder_path = './eval_pipeline/weights'
    os.makedirs(folder_path, exist_ok=True)

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        optimizer.zero_grad() # Initialize clear gradients before accumulation step loop

        for i, (batch_real, batch_style) in enumerate(dataloader):
            
            # 1. GENERATE PERFECT TARGETS (Using clean Teacher)
            with torch.no_grad():
                out_real = infer(teacher_model, batch_real)
                if isinstance(out_real, (list, tuple)):
                    out_real = out_real[0]
                
            # 4. Predict using stylized inputs WITH gradients tracked
            if isinstance(batch_style, list) and isinstance(batch_style[0], dict):
                batch_style_cuda = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                     for k, v in view.items()} for view in batch_style]
            elif isinstance(batch_style, dict):
                batch_style_cuda = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                     for k, v in batch_style.items()}
            else:
                batch_style_cuda = batch_style.to(device)

            # Call the model directly to keep the gradient tape alive!
            out_style = student_model(batch_style_cuda)
            if isinstance(out_style, (list, tuple)):
                out_style = out_style[0]

            # 5. Compute relative depth alignment loss
            loss_depth = criterion(out_style["depth_along_ray"], out_real["depth_along_ray"])
            loss_pts   = criterion(out_style["pts3d"], out_real["pts3d"])
            
            # Normalize structural loss down by accumulation value
            loss = (loss_depth + loss_pts) / args.accumulation_steps
            loss.backward()

            # Apply optimizer step every accumulated step milestone or at end of loader
            if (i + 1) % args.accumulation_steps == 0 or (i + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * args.accumulation_steps

        # Log average loss across the custom sub-epoch
        timed_print(f"Epoch [{epoch+1}/{args.epochs}] - Average Loss: {epoch_loss / len(dataloader):.4f}")

        # Save learned parameters at the end of EACH epoch for independent validation
        save_path = f"{folder_path}/adapter_{args.style}_{args.norm_num_blocks}_{'end' if args.norm_from_end else 'front'}_{epoch}.pth"
        
        torch.save(
            {name: (h._norm1d or h._norm2d).state_dict() for name, h, _ in student_model._norm_hook_triples},
            save_path
        )
        timed_print(f"Saved checkpoint to {save_path}")

    timed_print("Training complete.")

if __name__ == "__main__":
    train_adapter()