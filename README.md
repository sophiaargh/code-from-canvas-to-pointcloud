# From Canvas to Point Cloud: 3D reconstruction from artistic imagery

This repository collects tools, models and evaluation pipelines for visual-style transfer, depth/LiDAR processing, and related evaluations. It contains multiple subprojects used for training, inference and benchmarking. 

This project was developed for the EPFL course **CS503 — Visual Intelligence: Machines and Minds**.

**Report and visualizations**
Checkout the website! [https://sophiaargh.github.io/from-canvas-to-pointcloud/](https://sophiaargh.github.io/from-canvas-to-pointcloud/)

---

**Quick links**
- **TeleStyle**: [TeleStyle](TeleStyle/README.md) — style transfer inference & pipelines
- **LoRA adapters**: [lora/README.md](lora/README.md) — LoRA training and examples
- **Evaluation pipeline**: [eval_pipeline/README.md](eval_pipeline/README.md) — evaluators and metrics
- **Model code**: `mapanything/` — models, training utilities, which is a copy of the mapanything folder from the [MapAnything repository](https://github.com/facebookresearch/map-anything)

**Contents**
- **TeleStyle/** — Style transfer inference scripts, controlnet tuning, video pipelines and styles.
- **lora/** — LoRA adapter training, examples, exports and visualization tools.
- **eval_pipeline/** — Evaluation runners, additional metrics, and example scripts for depth/normal evaluation.
- **mapanything/** — Package containing models, training and utilities used across the project.
- Misc scripts: `train_adain.sh`, `train_adain_epoch.sh`, `run_dino_similarity.sh`, `transfer_folders.py` and lightweight benchmarks.

**Prerequisites**
- Linux or macOS with Python 3.8+ (Conda recommended).
- GPU (CUDA) for training and heavy inference workloads.
- Each subproject contains its own `requirements.txt` where applicable (for example, [TeleStyle/requirements.txt](TeleStyle/requirements.txt) and [lora/requirements.txt](lora/requirements.txt)).

Installation (example using conda)

```bash
conda create -n visual-intel python=3.11 -y
conda activate visual-intel
pip install -r TeleStyle/requirements.txt
pip install -r lora/requirements.txt
pip install -r eval_pipeline/requirements.txt

```

How the repository is organized

- **Training**
	- `train_adain.sh`, `train_adain_epoch.sh` — scripts to train AdaIN-style models (see `eval_pipeline/train_adaIN.py`).
	- `lora/train_lora.py` — training script for LoRA adapters (configs in `lora/configs/`).

- **Inference / Demo**
	- `TeleStyle/telestyleimage_inference.py`, `telestylevideo_inference.py` — image/video inference pipelines.
	- `TeleStyle/run_telestyle.sh`, `TeleStyle/run_telestyle_style_preview.sh` — convenience run scripts.
	- `lora/export_pointclouds.py`, `lora/visualize_ply.py` — export and visualize pointcloud results.

- **Evaluation**
	- `eval_pipeline/runner.py` and `evaluator.py` — run benchmarks and compute metrics.
	- `eval_dino_similarity.py` and `run_dino_similarity.sh` — DINO-based similarity evaluations.

- **Utilities**
	- `transfer_folders.py` — helper for moving or reorganizing dataset folders.
	- `plot_adain.ipynb`, `plot_gray.ipynb`, `plot_norm.ipynb` — notebooks for visualization.

**Repository tree (high-level)**

- `/` — repository root
	- `TeleStyle/` — style-transfer inference, ControlNet tuning, video pipelines and convenience run scripts. Key files: `telestyleimage_inference.py`, `telestylevideo_inference.py`, `run_telestyle.sh`, `tune_controlnet_strength.py`, `styles/`.
	- `TeleStyle/*.sh` and top-level scripts — convenience wrappers for common tasks: `run_telestyle.sh`, `run_destylize.sh`, `run_tune_controlnet_strength.sh`, `submit_evaluate.sh`, `train_adain.sh`, `train_adain_epoch.sh`.
	- `lora/` — LoRA adapter training, adapters, export & visualization tools. Key files: `train_lora.py`, `lora_adapter.py`, `export_pointclouds.py`, `configs/`, `requirements.txt`.
	- `eval_pipeline/` — evaluation runners, metrics, and training helpers for depth/normal reconstruction. Key files: `runner.py`, `evaluator.py`, `models.py`, `normalization.py`, `train_adaIN.py`, `additional_evals/`.
	- `eval_*.py` and `evaluation_results/` — quick evaluation entrypoints and collected CSV results (e.g., `eval_dino_similarity.py`, `eval_controlnet_grid.py`, `evaluation_results/*.csv`).
	- `mapanything/` — model implementations, datasets and training utilities (copied from MapAnything project; see folder README).
	- `mini-benchmark/` — small example datasets and quick benchmarks used for development and demos.
	- `transfer_folders.py` — dataset/folder utilities used by evaluation and training scripts.
	- Notebooks: `plot_adain.ipynb`, `plot_gray.ipynb`, `plot_norm.ipynb` — visualization and analysis notebooks.

This tree highlights the primary components and entrypoints; see subproject READMEs (for example, [TeleStyle/README.md](TeleStyle/README.md) and [lora/README.md](lora/README.md)) for detailed usage and examples.

Usage examples on the Scitas cluster.

- Run a TeleStyle image inference (example):

```bash
# from repo root
sbatch TeleStyle/run_telestyle.sh --input path/to/image.jpg --style TeleStyle/styles/your_style.npy
```

- Train a LoRA adapter (example):

```bash
# submit a training job using the provided script
sbtach lora/scripts/run_lora_training.sh
```

- Run evaluation pipeline (example):

```bash
# submit the evaluation job using the helper script
sbatch submit_evaluate.sh
```

**Important note:** The dataset which we use throughout this project weights approximately 17GB and will be downloaded in your HuggingFace cache. 
