import torch
from mapanything.models import MapAnything


def get_model(checkpoint: str = "facebook/map-anything", device: str | None = None):
    """Load MapAnything model and return (model, device).

    Minimal wrapper to centralize model creation for experiments.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = MapAnything.from_pretrained(checkpoint).to(device)
    model.eval()
    return model, device


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
