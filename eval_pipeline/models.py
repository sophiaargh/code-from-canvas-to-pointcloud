import torch
from mapanything.models import MapAnything
from .normalization import apply_instance_norm_to_encoder


def get_model(
    checkpoint: str = "facebook/map-anything",
    device: str | None = None,
    encoder_block_prefix: str | None = None,   # e.g. "encoder.blocks"
    norm_num_blocks: int | None = None,         # None = all blocks
):
    """Load MapAnything model and return (model, device).

    Minimal wrapper to centralize model creation for experiments.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = MapAnything.from_pretrained(checkpoint).to(device)
    model.eval()

    if encoder_block_prefix is not None:
        print(f"Applying InstanceNorm to DinoV2 blocks under '{encoder_block_prefix}'")
        handles = apply_instance_norm_to_encoder(model, encoder_block_prefix, norm_num_blocks)
        # Store handles on the model so they stay alive for the process lifetime
        model._norm_hook_handles = handles

    return model, device


def infer(model, views):
    """Call the model's inference helper when available, else fallback to forward.

    Prefers `model.infer(...)` because it handles device/dtype and postprocessing.
    """
    if hasattr(model, "infer"):
        return model.infer(views)
    # fallback: call model directly (must ensure views are on correct device/dtype)
    with torch.no_grad():
        return model(views)
