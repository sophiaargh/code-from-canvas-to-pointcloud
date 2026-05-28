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

STYLED_ROOT=/scratch/izar/silly/BlendedMVS/telestyle_output
DATA_DIR=/scratch/izar/silly/BlendedMVS/renamed

# ---------------------------------------------------------------------------
# Point this at whichever checkpoint to evaluate.
# For mid-training checks use step_XXXXXX; for the final run use "final".
# ---------------------------------------------------------------------------
LORA_CKPT=/scratch/izar/silly/lora_checkpoints/mixed_styles_gray_consistency/step_002500
CKPT_TAG=consistency_step2500   # appended to baseline_name for easy comparison

# ---------------------------------------------------------------------------
# Evaluation runs — uncomment ONE block at a time, then: sbatch submit_evaluate.sh
# ---------------------------------------------------------------------------

# --- [REFERENCE] Baseline on original photographs (already done → photographs_100_0blocks.csv)
# python -m lora.eval.runner \
#   --data_dir $DATA_DIR \
#   --checkpoint facebook/map-anything \
#   --baseline_name photographs_100_0blocks \
#   --max_scenes 100

# --- [REFERENCE] Baseline on pure impressionism (already done → impressionism_100_0blocks.csv)
# python -m lora.eval.runner \
#   --data_dir $STYLED_ROOT/impressionism \
#   --checkpoint facebook/map-anything \
#   --baseline_name impressionism_100_0blocks

# --- [REFERENCE] Baseline, mixed input grayscale (already done → mixed_baseline_gray_4_training_views.csv)
# python -m lora.eval.runner \
#   --data_dir $DATA_DIR \
#   --checkpoint facebook/map-anything \
#   --styled_root $STYLED_ROOT \
#   --style_names engraving impressionism oil_painting watercolor \
#   --n_styled 4 --mixed --grayscale \
#   --baseline_name mixed_baseline_gray_4_training_views

# ---------------------------------------------------------------------------
# TEST 1 — LoRA on pure impressionism
# Key question: does consistency training improve robustness to styled images?
# Compare against: impressionism_lora.csv (old LoRA) and impressionism_100_0blocks.csv (baseline)
# ---------------------------------------------------------------------------
python -m lora.eval.runner \
  --data_dir $STYLED_ROOT/impressionism \
  --checkpoint facebook/map-anything \
  --lora_path $LORA_CKPT \
  --baseline_name impressionism_lora_${CKPT_TAG}

# ---------------------------------------------------------------------------
# TEST 2 — LoRA on clean photographs
# Key question: does consistency training preserve performance on clean inputs?
# Compare against: photographs_100_lora.csv (old LoRA) and photographs_100_0blocks.csv (baseline)
# ---------------------------------------------------------------------------
python -m lora.eval.runner \
  --data_dir $DATA_DIR \
  --checkpoint facebook/map-anything \
  --lora_path $LORA_CKPT \
  --baseline_name photographs_100_lora_${CKPT_TAG} \
  --max_scenes 100

# ---------------------------------------------------------------------------
# TEST 3 — LoRA on mixed input (4 styled + 4 original, grayscale) — training conditions
# Key question: does the model handle the exact training distribution well?
# Compare against: mixed_lora_gray_4_training_views.csv (old LoRA)
# ---------------------------------------------------------------------------
python -m lora.eval.runner \
  --data_dir $DATA_DIR \
  --checkpoint facebook/map-anything \
  --lora_path $LORA_CKPT \
  --styled_root $STYLED_ROOT \
  --style_names engraving impressionism oil_painting watercolor \
  --n_styled 4 --mixed --grayscale \
  --baseline_name mixed_lora_gray_${CKPT_TAG}