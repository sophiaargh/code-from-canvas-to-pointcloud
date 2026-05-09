#!/bin/bash
#SBATCH --job-name=telestyle_inference
#SBATCH --output=logs/telestyle_%j.log
#SBATCH --error=logs/telestyle_%j.err
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

export HF_HOME="/scratch/izar/$USER/huggingface"
export HF_HUB_CACHE="/scratch/izar/$USER/huggingface/hub"
export TRANSFORMERS_CACHE="/scratch/izar/$USER/huggingface/transformers"
export HF_DATASETS_CACHE="/scratch/izar/$USER/huggingface/datasets"
export SHARED_SCRATCH_DIR="/scratch/izar/silly"

export PYTORCH_ALLOC_CONF=expandable_segments:True
#export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate telestyle

STYLE="monet"  # TODO: choose style — must match a filename in styles/ (without extension)
DATASET_ROOT="$SHARED_SCRATCH_DIR/BlendedMVS/renamed"
SAVE_DIR="$SHARED_SCRATCH_DIR/BlendedMVS/telestyle_output"

STYLE_PATH=$(find "$SCRIPT_DIR/styles" -iname "${STYLE}.*" | head -1)
if [[ -z "$STYLE_PATH" ]]; then
    echo "Error: no file matching '${STYLE}.*' found in ${SCRIPT_DIR}/styles" >&2
    exit 1
fi

python "$SCRIPT_DIR/telestyleimage_inference.py" \
    --style "$STYLE_PATH" \
    --dataset-root "$DATASET_ROOT" \
    --save-dir "$SAVE_DIR"