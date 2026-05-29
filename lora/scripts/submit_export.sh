#!/bin/bash
#SBATCH --job-name=export_ply
#SBATCH --time=03:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/export_ply_%j.out
#SBATCH --error=logs/export_ply_%j.err

export HF_HOME="/scratch/izar/$USER/huggingface"
export HF_HUB_CACHE="/scratch/izar/$USER/huggingface/hub"
export TRANSFORMERS_CACHE="/scratch/izar/$USER/huggingface/transformers"
export HF_DATASETS_CACHE="/scratch/izar/$USER/huggingface/datasets"
export SHARED_SCRATCH_DIR="/scratch/izar/silly"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate visual-intel

export PYTHONPATH="/home/$USER/visual-intelligence:${PYTHONPATH}"

mkdir -p logs lora/results/ply_exports

DATA_DIR=$SHARED_SCRATCH_DIR/BlendedMVS/renamed
STYLED_ROOT=$SHARED_SCRATCH_DIR/BlendedMVS/telestyle_output
LORA=$SHARED_SCRATCH_DIR/lora_checkpoints/mixed_styles_gray/final
LORA_CONST=$SHARED_SCRATCH_DIR/lora_checkpoints/mixed_styles_gray_consistency/step_002500

SCENES="scene_15 scene_33 scene_51 scene_63 scene_100 scene_22 scene_1 scene_27 scene_16 scene_26 scene_40 scene_23 scene_36 scene_38 scene_0 scene_13"

python lora.export_pointclouds.py \
  --condition mixed_baseline \
  --data_dir $DATA_DIR \
  --styled_root $STYLED_ROOT \
  --style_names engraving impressionism oil_painting watercolor \
  --n_styled 4 \
  --grayscale \
  --scenes $SCENES

python lora.export_pointclouds.py \
  --condition mixed_lora \
  --data_dir $DATA_DIR \
  --styled_root $STYLED_ROOT \
  --lora_path $LORA \
  --style_names engraving impressionism oil_painting watercolor \
  --n_styled 4 \
  --grayscale \
  --scenes $SCENES

python lora.export_pointclouds.py \
  --condition mixed_lora_const \
  --data_dir $DATA_DIR \
  --styled_root $STYLED_ROOT \
  --lora_path $LORA_CONST \
  --style_names engraving impressionism oil_painting watercolor \
  --n_styled 4 \
  --grayscale \
  --scenes $SCENES


