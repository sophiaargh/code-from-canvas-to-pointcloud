#!/bin/bash
#SBATCH --job-name=tune_controlnet
#SBATCH --output=logs/tune_controlnet_%j.log
#SBATCH --error=logs/tune_controlnet_%j.err
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
STYLE=${STYLE:-"watercolor"}
SCENE=${SCENE:-"scene_0"}
N_IMAGES=${N_IMAGES:-4}
STRENGTHS=${STRENGTHS:-"0.3 0.4 0.5 0.6 0.7"}
SAVE_DIR="$HOME/destylize_tuning/${STYLE}"
# ─────────────────────────────────────────────────────────────────────────────

cd "/scratch/izar/$USER"

python "$SCRIPT_DIR/tune_controlnet_strength.py" \
    --stylized-dir "$SHARED_SCRATCH_DIR/BlendedMVS/telestyle_output/${STYLE}" \
    --scene "$SCENE" \
    --n-images "$N_IMAGES" \
    --strengths $STRENGTHS \
    --save-dir "$SAVE_DIR"
