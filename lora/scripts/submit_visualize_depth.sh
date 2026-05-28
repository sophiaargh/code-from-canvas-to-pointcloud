#!/bin/bash
#SBATCH --job-name=visualize_depth
#SBATCH --time=06:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/visualize_depth_%j.out
#SBATCH --error=logs/visualize_depth_%j.err

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

LORA_1=/scratch/izar/silly/lora_checkpoints/mixed_styles_gray/final
LORA_2=/scratch/izar/silly/lora_checkpoints/mixed_styles_gray_consistency/step_002500

python -m eval_pipeline.visualize_depth \
  --data_dir        $DATA_DIR \
  --max_scenes      20 \
  --styled_root     $STYLED_ROOT \
  --style_names     engraving impressionism oil_painting watercolor \
  --checkpoint      facebook/map-anything \
  --lora_path_1     $LORA_1 \
  --lora_path_2     $LORA_2 \
  --label_baseline  baseline \
  --label_1         lora_mixed_gray \
  --label_2         lora_consistency \
  --grayscale \
  --out_dir         lora/results/depth_visualizations
