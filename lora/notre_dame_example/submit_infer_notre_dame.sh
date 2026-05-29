#!/bin/bash
#SBATCH --job-name=infer_notre_dame
#SBATCH --time=01:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=notre_dame_example/logs/infer_%j.out
#SBATCH --error=notre_dame_example/logs/infer_%j.err

export HF_HOME="/scratch/izar/$USER/huggingface"
export HF_HUB_CACHE="/scratch/izar/$USER/huggingface/hub"
export TRANSFORMERS_CACHE="/scratch/izar/$USER/huggingface/transformers"
export HF_DATASETS_CACHE="/scratch/izar/$USER/huggingface/datasets"
export SHARED_SCRATCH_DIR="/scratch/izar/silly"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate visual-intel

export PYTHONPATH="/home/$USER/visual-intelligence:${PYTHONPATH}"

mkdir -p notre_dame_example/logs

LORA=$SHARED_SCRATCH_DIR/lora_checkpoints/mixed_styles_gray/final
LORA_CONSISTENCY=$SHARED_SCRATCH_DIR/lora_checkpoints/mixed_styles_gray_consistency/step_002500

# Base model (no LoRA)
python -m lora.notre_dame_example.infer_notre_dame.py \
  --out_path lora/notre_dame_example/notre_dame_base.ply \
  --grayscale --n_points 0

python -m lora.notre_dame_example.infer_notre_dame.py \
  --out_path lora/notre_dame_example/notre_dame_base_viewcolors.ply \
  --grayscale --n_points 0 --color_by_view

# LoRA model
python -m lora.notre_dame_example.infer_notre_dame.py \
  --lora_path $LORA \
  --out_path lora/notre_dame_example/notre_dame_lora.ply \
  --grayscale --n_points 0

python -m lora.notre_dame_example.infer_notre_dame.py \
  --lora_path $LORA \
  --out_path lora/notre_dame_example/notre_dame_lora_viewcolors.ply \
  --grayscale --n_points 0 --color_by_view

# LoRA consistency model
python -m lora.notre_dame_example.infer_notre_dame.py \
  --lora_path $LORA_CONSISTENCY \
  --out_path lora/notre_dame_example/notre_dame_lora_consistency.ply \
  --grayscale --n_points 0

python -m lora.notre_dame_example.infer_notre_dame.py \
  --lora_path $LORA_CONSISTENCY \
  --out_path lora/notre_dame_example/notre_dame_lora_consistency_viewcolors.ply \
  --grayscale --n_points 0 --color_by_view
