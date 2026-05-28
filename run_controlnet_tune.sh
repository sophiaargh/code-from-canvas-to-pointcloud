#!/bin/bash
#SBATCH --job-name=cn_tune
#SBATCH --output=logs/cn_tune_%j.log
#SBATCH --error=logs/cn_tune_%j.err
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --account=cs-503
#SBATCH --qos=cs-503

export HF_HOME="/scratch/izar/$USER/huggingface"
export HF_HUB_CACHE="/scratch/izar/$USER/huggingface/hub"
export TRANSFORMERS_CACHE="/scratch/izar/$USER/huggingface/transformers"
export HF_DATASETS_CACHE="/scratch/izar/$USER/huggingface/datasets"
export SHARED_SCRATCH_DIR="/scratch/izar/silly"

export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

SCRIPT_DIR=$SLURM_SUBMIT_DIR

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate telestyle

# ── Configure ────────────────────────────────────────────────────────────────
# One scene with ~8 frames is enough to rank the grid.
SCENE=${SCENE:-"scene_0"}

STYLIZED_ROOT="$SHARED_SCRATCH_DIR/BlendedMVS/telestyle_output"
TUNING_DIR="$HOME/destylize_tuning"
# ─────────────────────────────────────────────────────────────────────────────

cd "/scratch/izar/$USER"

echo "=== Step 1/2: generating grid images for scene $SCENE ==="
python "$SCRIPT_DIR/TeleStyle/controlnet_tune_grid.py" \
    --stylized-root "$STYLIZED_ROOT" \
    --scene         "$SCENE" \
    --out-dir       "$TUNING_DIR"

echo "=== Step 2/2: evaluating grid with DINOv2 ==="
python "$SCRIPT_DIR/eval_controlnet_grid.py" \
    --data-root  "$SHARED_SCRATCH_DIR/BlendedMVS" \
    --tuning-dir "$TUNING_DIR" \
    --scene      "$SCENE" \
    --out        "$SCRIPT_DIR/evaluation_results/controlnet_grid_${SCENE}.csv"
