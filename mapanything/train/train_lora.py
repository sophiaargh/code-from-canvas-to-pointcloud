# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Standalone LoRA fine-tuning script for MapAnything.

Loads a pretrained MapAnything checkpoint, injects LoRA adapters, and trains
on TeleStyle-rendered BlendedMVS images while keeping the original depth/camera GT.
Only the LoRA parameters are updated; all base weights remain frozen.

Example usage:
    python -m mapanything.train.train_lora \
        --base_checkpoint facebook/map-anything \
        --lora_out_dir /scratch/izar/silly/lora_checkpoints/impressionism \
        --style_name impressionism \
        --lora_rank 8 \
        --lora_alpha 16 \
        --max_steps 5000
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from mapanything.datasets.wai.blendedmvs_styled import BlendedMVSStyled
from mapanything.models.mapanything.lora_adapter import apply_lora, probe_target_modules, save_lora_weights
from mapanything.train.losses import *  # noqa — exposes ConfLoss, Regr3D, L2Loss, etc.
from mapanything.utils.inference import loss_of_one_batch_multi_view

# Typical resolution and normalisation used by the pretrained MapAnything model.
_DEFAULT_RESOLUTION = (518, 392)
_DEFAULT_DATA_NORM = "dinov2"

# Default criterion string — same family as DUSt3R / MASt3R training defaults.
_DEFAULT_CRITERION = "ConfLoss(Regr3D(L2Loss(), norm_mode='?avg_dis', gt_scale=False, loss_in_log=True), alpha=0.2)"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="LoRA fine-tuning for MapAnything")

    # Model
    p.add_argument("--base_checkpoint", default="facebook/map-anything",
                   help="HuggingFace model ID or local path for the pretrained MapAnything checkpoint")
    p.add_argument("--lora_out_dir", required=True,
                   help="Directory to save LoRA adapter weights")

    # LoRA
    p.add_argument("--lora_rank", type=int, default=8, help="LoRA rank r")
    p.add_argument("--lora_alpha", type=float, default=16.0, help="LoRA scaling alpha")
    p.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    p.add_argument("--lora_include_mlp", action="store_true",
                   help="Also inject LoRA into MLP / FFN layers (default: attention only)")
    p.add_argument("--lora_target_modules", nargs="+", default=None,
                   help="Explicit list of linear layer name suffixes to target. "
                        "If omitted, auto-detected from model.")

    # Dataset
    p.add_argument("--style_name", default="impressionism",
                   help="Name of the artistic style (must match TeleStyle output subdirectory)")
    p.add_argument("--styled_root",
                   default="/scratch/izar/silly/BlendedMVS/telestyle_output",
                   help="Root directory containing per-style TeleStyle output")
    p.add_argument("--dataset_root",
                   default="/scratch/izar/silly/BlendedMVS/renamed",
                   help="Root directory of the original BlendedMVS dataset")
    p.add_argument("--num_views", type=int, default=2,
                   help="Number of views per training sample")
    p.add_argument("--num_workers", type=int, default=4)

    # Training
    p.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate")
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--batch_size", type=int, default=4,
                   help="Number of scene samples per batch (each yields num_views images)")
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--save_every", type=int, default=500,
                   help="Save LoRA checkpoint every N steps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_amp", action="store_true",
                   help="Use automatic mixed precision (bfloat16)")
    p.add_argument("--criterion", default=_DEFAULT_CRITERION,
                   help="Loss criterion expression (eval'd in the losses module namespace)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def build_styled_dataset(args):
    dataset = BlendedMVSStyled(
        ROOT=args.dataset_root,
        split="train",
        num_views=args.num_views,
        resolution=_DEFAULT_RESOLUTION,
        transform="colorjitter",
        data_norm_type=_DEFAULT_DATA_NORM,
        aug_crop=16,
        style_name=args.style_name,
        styled_root=args.styled_root,
    )
    return dataset


# ---------------------------------------------------------------------------
# Collation — the dataset returns a list-of-views per sample; we stack them.
# ---------------------------------------------------------------------------

def collate_views(batch_of_view_lists):
    """Collate a list of view-lists into a single batched view-list.

    Args:
        batch_of_view_lists: List of length B, each element is a list of V view dicts.

    Returns:
        List of V collated view dicts with tensors of shape (B, ...).
    """
    num_views = len(batch_of_view_lists[0])
    collated = []
    for v in range(num_views):
        keys = batch_of_view_lists[0][v].keys()
        view_batch = {}
        for k in keys:
            samples = [b[v][k] for b in batch_of_view_lists]
            if isinstance(samples[0], torch.Tensor):
                view_batch[k] = torch.stack(samples, dim=0)
            elif isinstance(samples[0], np.ndarray):
                view_batch[k] = torch.from_numpy(np.stack(samples, axis=0))
            elif isinstance(samples[0], (bool, np.bool_)):
                view_batch[k] = torch.tensor(samples)
            else:
                view_batch[k] = samples  # strings, tuples, etc.
        collated.append(view_batch)
    return collated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load pretrained model ----
    print(f"Loading pretrained model: {args.base_checkpoint}")
    from mapanything.models import MapAnything
    model = MapAnything.from_pretrained(args.base_checkpoint).to(device)

    # ---- Apply LoRA ----
    print("Probing target modules …")
    if args.lora_target_modules:
        target_modules = args.lora_target_modules
        print(f"  Using explicitly specified modules: {target_modules}")
    else:
        target_modules = probe_target_modules(model, include_mlp=args.lora_include_mlp)
        print(f"  Auto-detected modules: {target_modules}")

    model = apply_lora(
        model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        target_modules=target_modules,
        dropout=args.lora_dropout,
    )
    model.to(device)

    # ---- Loss criterion ----
    criterion = eval(args.criterion).to(device)
    print(f"Criterion: {criterion}")

    # ---- Dataset & DataLoader ----
    print("Building dataset …")
    dataset = build_styled_dataset(args)
    print(f"  Dataset size: {len(dataset)} scenes")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=collate_views,
        pin_memory=(device.type == "cuda"),
    )

    # ---- Optimizer (LoRA params only) ----
    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=args.weight_decay)

    # ---- Output dir ----
    Path(args.lora_out_dir).mkdir(parents=True, exist_ok=True)

    # ---- Training loop ----
    model.train()
    step = 0
    running_loss = 0.0
    t0 = time.time()

    print(f"Starting LoRA training for {args.max_steps} steps …")
    while step < args.max_steps:
        for batch in loader:
            if step >= args.max_steps:
                break

            optimizer.zero_grad()

            result = loss_of_one_batch_multi_view(
                batch=batch,
                model=model,
                criterion=criterion,
                device=device,
                use_amp=args.use_amp,
                amp_dtype="bf16",
            )
            loss = result["loss"]
            if loss is None:
                print("Warning: criterion returned None loss, skipping batch.")
                continue
            if isinstance(loss, tuple):
                loss, _ = loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            step += 1

            if step % 50 == 0:
                avg_loss = running_loss / 50
                elapsed = time.time() - t0
                print(f"  step {step:5d}/{args.max_steps}  loss={avg_loss:.4f}  "
                      f"elapsed={elapsed:.0f}s")
                running_loss = 0.0

            if step % args.save_every == 0:
                ckpt_dir = os.path.join(args.lora_out_dir, f"step_{step:06d}")
                save_lora_weights(model, ckpt_dir)
                print(f"  Saved LoRA checkpoint → {ckpt_dir}")

    # ---- Final checkpoint ----
    final_dir = os.path.join(args.lora_out_dir, "final")
    save_lora_weights(model, final_dir)
    print(f"Training complete. Final LoRA weights saved to {final_dir}")


if __name__ == "__main__":
    main()
