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
LORA_FINAL=/scratch/izar/silly/lora_checkpoints/mixed_styles_gray_4views/final
DATA_DIR=/scratch/izar/silly/BlendedMVS/renamed

# ---------------------------------------------------------------------------
# Evaluation runs — uncomment ONE block at a time, then: sbatch submit_evaluate.sh
# ---------------------------------------------------------------------------

# --- Baseline on original photographs (photographs_100_0blocks.csv, already done)
# baseline="photographs"
# max_scenes=100
# python -m eval_pipeline.runner \
#   --data_dir "/scratch/izar/silly/BlendedMVS/renamed/" \
#   --checkpoint facebook/map-anything \
#   --baseline_name "${baseline}_${max_scenes}_0blocks" \
#   --max_scenes "$max_scenes"

# --- Baseline on a single style (e.g. engraving_100_0blocks.csv, already done)
# baseline="engraving"  # or impressionism / oil_painting / watercolor
# max_scenes=100
# python "./transfer_folders.py" "$baseline"
# python -m eval_pipeline.runner \
#   --data_dir "/scratch/izar/silly/BlendedMVS/telestyle_output/${baseline}/" \
#   --checkpoint facebook/map-anything \
#   --baseline_name "${baseline}_${max_scenes}_0blocks" \
#   --max_scenes "$max_scenes"

# --- Normalization experiments (example — adjust nbr and baseline as needed)
# for nbr in 1 2 3 4 8 12 16 20 24; do
#   python -m eval_pipeline.runner \
#     --data_dir "/scratch/izar/silly/BlendedMVS/renamed/" \
#     --checkpoint facebook/map-anything \
#     --baseline_name "photographs_100_${nbr}blocks" \
#     --max_scenes 100 \
#     --encoder_block_prefix encoder.model.blocks \
#     --norm_num_blocks "$nbr"
# done

# --- Baseline model, mixed input (4 styled + 4 original, grayscale) ---
python -m eval_pipeline.runner \
  --data_dir $DATA_DIR \
  --checkpoint facebook/map-anything \
  --styled_root $STYLED_ROOT \
  --style_names engraving impressionism oil_painting watercolor \
  --n_styled 4 \
  --mixed \
  --grayscale \
  --baseline_name mixed_baseline_gray_4_training_views

# --- LoRA model, mixed input (4 styled + 4 original, grayscale) ---
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
