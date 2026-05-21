#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --time=04:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/evaluate_%j.out
#SBATCH --error=logs/evaluate_%j.err

export HF_HOME="/scratch/izar/$USER/huggingface"
export HF_HUB_CACHE="/scratch/izar/$USER/huggingface/hub"
export TRANSFORMERS_CACHE="/scratch/izar/$USER/huggingface/transformers"
export HF_DATASETS_CACHE="/scratch/izar/$USER/huggingface/datasets"
export SHARED_SCRATCH_DIR="/scratch/izar/silly"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate mapanything

export PYTHONPATH="/home/qsandoz/visual-intelligence:${PYTHONPATH}"

mkdir -p logs

# ---------------------------------------------------------------------------
# Evaluation runs — uncomment ONE block at a time, then: sbatch submit_evaluate.sh
#
# Correct scenario (mixed): 1 styled view + rest original photos per scene.
# This matches the actual use case of the LoRA adapter.
#
# Already done (all-styled, now known to be the wrong scenario):
#   photographs.csv, baseline_impressionism.csv, lora_impressionism.csv
# ---------------------------------------------------------------------------

STYLED_ROOT=/scratch/izar/silly/BlendedMVS/telestyle_output
LORA_FINAL=/scratch/izar/silly/lora_checkpoints/mixed_styles_gray_4views/final
DATA_DIR=/scratch/izar/silly/BlendedMVS/renamed

# --- Run 1: baseline on original photographs (already done, skip)
# python -m eval_pipeline.runner \
#   --data_dir $DATA_DIR \
#   --checkpoint facebook/map-anything \
#   --baseline_name photographs

# --- Run 4: baseline model, mixed input (4 styled + 4 original, mixed styles), grayscale ---
python -m eval_pipeline.runner \
  --data_dir $DATA_DIR \
  --checkpoint facebook/map-anything \
  --styled_root $STYLED_ROOT \
  --style_names engraving impressionism oil_painting watercolor \
  --n_styled 4 \
  --mixed \
  --grayscale \
  --baseline_name mixed_baseline_gray_4_training_views

# --- Run 5: LoRA model, mixed input (4 styled + 4 original, mixed styles), grayscale ---
python -m eval_pipeline.runner \
  --data_dir $DATA_DIR \
  --checkpoint facebook/map-anything \
  --lora_path $LORA_FINAL \
  --styled_root $STYLED_ROOT \
  --style_names engraving impressionism oil_painting watercolor \
  --n_styled 4 \
  --mixed \
  --grayscale \
  --baseline_name mixed_lora_gray_4_training_views