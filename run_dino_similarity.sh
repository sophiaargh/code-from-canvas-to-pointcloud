#!/bin/bash
#SBATCH --job-name=dino_similarity
#SBATCH --output=logs/dino_similarity_%j.log
#SBATCH --error=logs/dino_similarity_%j.err
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

export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

SCRIPT_DIR=$SLURM_SUBMIT_DIR

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate telestyle

# ── Configure ────────────────────────────────────────────────────────────────
MAX_SCENES=${MAX_SCENES:-""}
SCENE=${SCENE:-""}
# ─────────────────────────────────────────────────────────────────────────────

EXTRA_ARGS=""
[ -n "$MAX_SCENES" ] && EXTRA_ARGS="$EXTRA_ARGS --max-scenes $MAX_SCENES"
[ -n "$SCENE"      ] && EXTRA_ARGS="$EXTRA_ARGS --scene $SCENE"

python "$SCRIPT_DIR/eval_dino_similarity.py" \
    --data-root "/scratch/izar/silly/BlendedMVS" \
    --styles watercolor oil_painting impressionism engraving \
    --out "$SCRIPT_DIR/evaluation_results/dino_similarity.csv" \
    $EXTRA_ARGS
