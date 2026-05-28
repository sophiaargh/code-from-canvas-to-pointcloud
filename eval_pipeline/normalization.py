import torch
import torch.nn as nn


class InstanceNormHook(nn.Module):
    """
    Applies Instance Normalization to patch tokens only, preserving the CLS token.
    Inherits from nn.Module to cleanly expose affine parameters to the optimizer.
    """

    def __init__(self, affine: bool = False, eps: float = 1e-5):
        super().__init__()
        self.affine = affine
        self.eps = eps
        self._norm1d = None
        self._norm2d = None

    def forward(self, module, input, output):
        if output.dim() == 3:
            B, N, C = output.shape

            # --- Protect CLS token (index 0) ---
            cls_token = output[:, :1, :]   # (B, 1, C)
            patches   = output[:, 1:, :]   # (B, N-1, C)

            x = patches.permute(0, 2, 1)   # (B, C, N-1)

            if self._norm1d is None or self._norm1d.num_features != C:
                self._norm1d = nn.InstanceNorm1d(
                    C, affine=self.affine, eps=self.eps
                ).to(output.device)

            normed = self._norm1d(x).permute(0, 2, 1)   # (B, N-1, C)

            return torch.cat([cls_token, normed], dim=1)  # (B, N, C)

        elif output.dim() == 4:
            B, C, H, W = output.shape
            if self._norm2d is None or self._norm2d.num_features != C:
                self._norm2d = nn.InstanceNorm2d(
                    C, affine=self.affine, eps=self.eps
                ).to(output.device)
            return self._norm2d(output)

        return output


def apply_instance_norm_to_encoder(
    model:                 nn.Module,
    encoder_block_prefix: str,
    num_blocks:           int | None = None,
    from_end:              bool       = False,
    affine:                bool       = False,
) -> list:
    """Register InstanceNorm hooks on encoder blocks."""
    all_blocks: list[tuple[str, nn.Module]] = []
    target_depth = encoder_block_prefix.count(".") + 1

    for name, module in model.named_modules():
        if (
            name.startswith(encoder_block_prefix)
            and name.count(".") == target_depth
        ):
            all_blocks.append((name, module))

    if not all_blocks:
        raise ValueError(f"No modules found under prefix '{encoder_block_prefix}'.")

    if num_blocks is None:
        selected = all_blocks
    elif from_end:
        selected = all_blocks[-num_blocks:]
    else:
        selected = all_blocks[:num_blocks]

    handles = []
    for name, module in selected:
        hook   = InstanceNormHook(affine=affine)
        handle = module.register_forward_hook(hook)
        handles.append((name, hook, handle))
        print(f"  [hook] InstanceNorm ({'affine' if affine else 'no affine'}) → {name}")

    return handles