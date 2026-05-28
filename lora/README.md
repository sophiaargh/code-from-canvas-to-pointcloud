# LoRA fine-tuning for MapAnything

Fine-tunes MapAnything with LoRA adapters so it handles stylized (TeleStyle) inputs. Only the LoRA parameters are trained, all base weights stay frozen.

## Structure

```
lora/
├── lora_adapter.py          Core LoRA utilities: inject adapters, save/load weights
├── train_lora.py            Training entry point (run via scripts/run_lora_training.sh)
├── check_lora.py            Quick sanity check: loads the model and prints detected LoRA modules
│
├── datasets/                Dataset classes for training
│   ├── blendedmvs_raw.py    Raw BlendedMVS reader (no WAI wrapper, used as base class)
│   ├── blendedmvs_styled.py BlendedMVS with all views replaced by a single TeleStyle style
│   └── blendedmvs_mixed_styles.py  BlendedMVS with n views replaced by random styles (used for training)
│
├── eval/                    Evaluation pipeline (mixed-mode: styled + original views)
│   ├── evaluator.py         Scene-level evaluator with mixed/styled/grayscale support
│   ├── models.py            Model loaders: get_model() for baseline, load_with_lora() for LoRA
│   └── runner.py            CLI entry point — python -m lora.eval.runner
│
├── configs/
│   └── lora.yaml            Hydra config for training hyperparameters
│
├── scripts/                 SLURM job scripts (submit with sbatch from the project root)
│   ├── run_lora_training.sh Launch a training run
│   ├── submit_evaluate.sh   Run evaluation (baseline and/or LoRA)
│   ├── submit_export.sh     Export point clouds to PLY files
│   └── submit_visualize_depth.sh  Generate depth map visualizations
│
├── export_pointclouds.py    Export per-scene PLY files for visual quality comparison
├── visualize_ply.py         Open3D GUI viewer for exported PLY point clouds (run locally)
├── visualize_depth.py       Generate side-by-side depth map PNGs (baseline vs LoRA)
│
├── notre_dame_example/      Qualitative demo on Notre-Dame images (outside training set)
│
└── results/                 All generated outputs (gitignored for large runs, committed for key results)
    ├── evaluation_results/  CSV files with per-scene F-score metrics
    ├── ply_exports/         Exported point clouds grouped by condition (baseline / lora / lora_const)
    └── depth_visualizations/ Depth map PNG comparisons per scene
```