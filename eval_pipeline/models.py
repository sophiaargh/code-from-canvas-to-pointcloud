import os
import torch
import torch.nn as nn
from mapanything.models import MapAnything
from .normalization import apply_instance_norm_to_encoder


def get_model(
    checkpoint:           str       = "facebook/map-anything",
    device:               str | None = None,
    encoder_block_prefix: str | None = None,
    norm_num_blocks:      int | None = None,
    norm_from_end:        bool       = False,   
    norm_affine:          bool       = False,   
    adapter_weights:      str | None = None,   
) -> tuple:
    """
    Load MapAnything, attach InstanceNorm hooks, and optionally load AdaIN weights.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model  = MapAnything.from_pretrained(checkpoint).to(device)
    model.eval()

    if encoder_block_prefix is not None:
        position = "last" if norm_from_end else "first"
        n_label  = norm_num_blocks if norm_num_blocks is not None else "all"
        print(f"Attaching InstanceNorm to {n_label} blocks "
              f"({position}) under '{encoder_block_prefix}'", flush=True)

        hook_triples = apply_instance_norm_to_encoder(
            model,
            encoder_block_prefix,
            num_blocks = norm_num_blocks,
            from_end   = norm_from_end,
            affine     = norm_affine,
        )
        model._norm_hook_triples = hook_triples

        print(f"Adaper: {adapter_weights}")
        # --- Load Adapter Weights if Provided ---
        if adapter_weights:
            if not os.path.exists(adapter_weights):
                raise FileNotFoundError(f"Could not find adapter weights at: {adapter_weights}")
                
            print(f"Loading adapter weights from: {adapter_weights}...", flush=True)
            state_dicts = torch.load(adapter_weights, map_location=device, weights_only=True)
            
            for name, hook, _ in model._norm_hook_triples:
                if name in state_dicts:
                    sd = state_dicts[name]
                    # Dynamically find feature dimensions from the checkpoint size
                    C = sd['weight'].shape[0]
                    
                    # Manually build and mount the sub-norm modules to bypass the lazy loader logic
                    hook._norm1d = nn.InstanceNorm1d(C, affine=hook.affine, eps=hook.eps).to(device)
                    hook._norm2d = nn.InstanceNorm2d(C, affine=hook.affine, eps=hook.eps).to(device)
                    
                    hook._norm1d.load_state_dict(sd)
                    hook._norm2d.load_state_dict(sd)
                    
            print("Adapter weights successfully injected into the hooks!", flush=True)

    return model, device


def remove_hooks(model: torch.nn.Module) -> None:
    """Cleanly remove all registered norm hooks (useful between experiments)."""
    if hasattr(model, "_norm_hook_triples"):
        for name, hook, handle in model._norm_hook_triples:
            handle.remove()
        del model._norm_hook_triples
        print("All norm hooks removed.")
def load_with_lora(
    lora_path: str,
    base_checkpoint: str = "facebook/map-anything",
    device: str | None = None,
):
    """Load a pretrained MapAnything model with a LoRA adapter applied.

    Args:
        lora_path: Path to a directory previously saved by
            ``mapanything.models.mapanything.lora_adapter.save_lora_weights``.
        base_checkpoint: HuggingFace model ID or local path for the base MapAnything weights.
        device: Target device string.  Defaults to CUDA if available.

    Returns:
        Tuple of (peft_model, device).  The model is in eval mode with LoRA weights merged.
    """
    from mapanything.models.mapanything.lora_adapter import load_lora_weights

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    base_model = MapAnything.from_pretrained(base_checkpoint)
    model = load_lora_weights(base_model, lora_path).to(device)
    model.eval()
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