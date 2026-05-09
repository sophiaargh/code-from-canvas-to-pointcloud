#!/bin/bash
#SBATCH --job-name=telestyle_preview
#SBATCH --output=logs/telestyle_%j.log
#SBATCH --error=logs/telestyle_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

export HF_HOME="/scratch/izar/$USER/huggingface"
export HF_HUB_CACHE="/scratch/izar/$USER/huggingface/hub"
export TRANSFORMERS_CACHE="/scratch/izar/$USER/huggingface/transformers"
export HF_DATASETS_CACHE="/scratch/izar/$USER/huggingface/datasets"
export SHARED_SCRATCH_DIR="/scratch/izar/silly"

export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8

SCRIPT_DIR=$SLURM_SUBMIT_DIR

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate telestyle

STYLES_DIR="$SCRIPT_DIR/styles"
DATASET_ROOT="$SHARED_SCRATCH_DIR/BlendedMVS/renamed"
SAVE_DIR="$SCRIPT_DIR/style_preview_output"

cd "/scratch/izar/$USER"

python "$SCRIPT_DIR/telestyle_style_preview.py" \
    --styles-dir "$STYLES_DIR" \
    --dataset-root "$DATASET_ROOT" \
    --save-dir "$SAVE_DIR"
