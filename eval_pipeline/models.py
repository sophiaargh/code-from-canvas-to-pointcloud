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


def infer(model, views):
    """Call the model's inference helper when available, else fallback to forward.

    Prefers `model.infer(...)` because it handles device/dtype and postprocessing.
    """
    if hasattr(model, "infer"):
        return model.infer(views)
    # fallback: call model directly (must ensure views are on correct device/dtype)
    with torch.no_grad():
        return model(views)
