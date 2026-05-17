import torch
import torch.nn as nn
from typing import Literal


class InstanceNormHook:
    """
    Applies Instance Normalization to the output of a DINOv2 encoder block.
    Works on tensors shaped (B, N, C) — the standard ViT sequence format —
    or (B, C, H, W) spatial feature maps.
    """
    def __init__(self):
        self._norm = None  # lazily initialized on first call

    def __call__(self, module, input, output):
        # ViT outputs are (B, seq_len, C); treat each channel independently
        if output.dim() == 3:
            B, N, C = output.shape
            x = output.permute(0, 2, 1)          # (B, C, N)
            if self._norm is None or self._norm.num_features != C:
                self._norm = nn.InstanceNorm1d(C, affine=False).to(output.device)
            return self._norm(x).permute(0, 2, 1)
        # Spatial feature maps (B, C, H, W)
        elif output.dim() == 4:
            B, C, H, W = output.shape
            if self._norm is None or self._norm.num_features != C:
                self._norm = nn.InstanceNorm2d(C, affine=False).to(output.device)
            return self._norm(output)
        return output  # passthrough for anything unexpected


def apply_instance_norm_to_encoder(
    model: nn.Module,
    encoder_block_prefix: str,
    num_blocks: int | None = None,
) -> list:
    """
    Register InstanceNorm hooks on DINOv2 encoder blocks.

    Args:
        model:                 The loaded MapAnything model.
        encoder_block_prefix:  The module path prefix for ViT blocks,
                               e.g. "encoder.blocks" or "backbone.dino.blocks".
        num_blocks:            How many blocks to hook (None = all found blocks).
                               Start with just the first few (e.g. 4) to limit impact.

    Returns:
        List of hook handles — call handle.remove() to undo.
    """
    handles = []
    blocks_hooked = 0
    for name, module in model.named_modules():
        if name.startswith(encoder_block_prefix) and name.count(".") == encoder_block_prefix.count(".") + 1:
            # Direct children of the prefix (i.e., individual blocks, not sub-layers)
            hook = InstanceNormHook()
            handle = module.register_forward_hook(hook)
            handles.append(handle)
            blocks_hooked += 1
            print(f"  [hook] InstanceNorm attached to: {name}")
            if num_blocks is not None and blocks_hooked >= num_blocks:
                break
    if blocks_hooked == 0:
        raise ValueError(
            f"No modules found under prefix '{encoder_block_prefix}'. "
            "Run model.named_modules() to inspect the correct path."
        )
    return handles