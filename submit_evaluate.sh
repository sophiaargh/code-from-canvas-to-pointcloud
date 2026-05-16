#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --time=02:00:00
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
# Three-way evaluation to assess LoRA style-agnosticism.
# Run 1 (already done): baseline on original photos → upper bound
# Run 2: baseline on styled images → shows degradation from style
# Run 3: LoRA model on styled images → shows recovery

# Uncomment ONE block at a time and submit with: sbatch submit_evaluate.sh
# ---------------------------------------------------------------------------

STYLED_ROOT=/scratch/izar/silly/BlendedMVS/telestyle_output
LORA_FINAL=/scratch/izar/silly/lora_checkpoints/impressionism/final
DATA_DIR=/scratch/izar/silly/BlendedMVS/renamed

# --- Run 1: baseline on original photographs (already done) ---
# python -m eval_pipeline.runner \
#   --data_dir $DATA_DIR \
#   --checkpoint facebook/map-anything \
#   --baseline_name photographs

# --- Run 2: baseline on impressionism styled images ---
#python -m eval_pipeline.runner \
#  --data_dir $DATA_DIR \
#  --checkpoint facebook/map-anything \
#  --styled_root $STYLED_ROOT \
#  --style_name impressionism \
#  --baseline_name baseline_impressionism

# --- Run 3: LoRA model on impressionism styled images ---
python -m eval_pipeline.runner \
  --data_dir $DATA_DIR \
  --checkpoint facebook/map-anything \
  --lora_path $LORA_FINAL \
  --styled_root $STYLED_ROOT \
  --style_name impressionism \
  --baseline_name lora_impressionism