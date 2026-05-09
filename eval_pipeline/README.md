Eval pipeline for MapAnything models

Files:
- models.py   : model factory and infer helper
- evaluator.py: Evaluator class implementing depth and point-cloud metrics
- runner.py   : CLI entrypoint

Quick start:

Activate your environment then run:

```bash
python -m eval_pipeline.runner \
  --data_dir /path/to/BlendedMVS \
  --checkpoint facebook/map-anything \
  --baseline_name photographs \
  --view_ids 0 10 20 30 40
```

Notes:
- `models.infer` will call `model.infer(...)` when available (preferred).
- The pipeline writes CSV summaries to `evaluation_results/` by default.
- You can adjust the number of scenes evaluated using the argument `--max_scenes`
