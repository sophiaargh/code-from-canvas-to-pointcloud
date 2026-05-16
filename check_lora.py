from mapanything.models import MapAnything
from mapanything.models.mapanything.lora_adapter import apply_lora, probe_target_modules
model = MapAnything.from_pretrained('facebook/map-anything')
print('Detected modules:', probe_target_modules(model))
model = apply_lora(model, rank=8, alpha=16)