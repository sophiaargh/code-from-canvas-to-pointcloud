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
conda activate visual-intel

export PYTHONPATH="/home/$USER/code-from-canvas-to-pointcloud:${PYTHONPATH}"

mkdir -p logs

STYLED_ROOT=/scratch/izar/silly/BlendedMVS/telestyle_output
DATA_DIR=/scratch/izar/silly/BlendedMVS/renamed

LORA_CONSISTENCY_CKPT=/scratch/izar/silly/lora_checkpoints/mixed_styles_gray_consistency/step_002500
LORA_CKPT=/scratch/izar/silly/lora_checkpoints/mixed_styles_gray/final

#  Baseline on original photographs 
python -m lora.eval.runner \
  --data_dir $DATA_DIR \
  --checkpoint facebook/map-anything \
  --baseline_name photographs_baseline \
  --max_scenes 100

#  Baseline on pure impressionism 
python -m lora.eval.runner \
  --data_dir $STYLED_ROOT/impressionism \
  --checkpoint facebook/map-anything \
  --baseline_name impressionism_baseline \
  --max_scenes 100

# Baseline, mixed input grayscale 
python -m lora.eval.runner \
  --data_dir $DATA_DIR \
  --checkpoint facebook/map-anything \
  --styled_root $STYLED_ROOT \
  --style_names engraving impressionism oil_painting watercolor \
  --n_styled 4 --mixed --grayscale \
  --baseline_name mixed_baseline_gray

python -m lora.eval.runner \
  --data_dir $STYLED_ROOT/impressionism \
  --checkpoint facebook/map-anything \
  --lora_path $LORA_CKPT \
  --baseline_name impressionism_lora

python -m lora.eval.runner \
  --data_dir $STYLED_ROOT/impressionism \
  --checkpoint facebook/map-anything \
  --lora_path $LORA_CONSISTENCY_CKPT \
  --baseline_name impressionism_lora_consistency


python -m lora.eval.runner \
  --data_dir $DATA_DIR \
  --checkpoint facebook/map-anything \
  --lora_path $LORA_CKPT \
  --baseline_name photographs_lora\

python -m lora.eval.runner \
  --data_dir $DATA_DIR \
  --checkpoint facebook/map-anything \
  --lora_path $LORA_CONSISTENCY_CKPT \
  --baseline_name photographs_lora_consistency

python -m lora.eval.runner \
  --data_dir $DATA_DIR \
  --checkpoint facebook/map-anything \
  --lora_path $LORA_CKPT \
  --styled_root $STYLED_ROOT \
  --style_names engraving impressionism oil_painting watercolor \
  --n_styled 4 --mixed --grayscale \
  --baseline_name mixed_lora_gray

python -m lora.eval.runner \
  --data_dir $DATA_DIR \
  --checkpoint facebook/map-anything \
  --lora_path $LORA_CONSISTENCY_CKPT \
  --styled_root $STYLED_ROOT \
  --style_names engraving impressionism oil_painting watercolor \
  --n_styled 4 --mixed --grayscale \
  --baseline_name mixed_lora_gray_consistency