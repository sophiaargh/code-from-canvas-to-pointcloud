#!/bin/bash
#SBATCH --job-name=destylize
#SBATCH --output=logs/destylize_%j.log
#SBATCH --error=logs/destylize_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --account=cs-503
#SBATCH --qos=cs-503

export HF_HOME="/scratch/izar/$USER/huggingface"
export HF_HUB_CACHE="/scratch/izar/$USER/huggingface/hub"
export TRANSFORMERS_CACHE="/scratch/izar/$USER/huggingface/transformers"
export HF_DATASETS_CACHE="/scratch/izar/$USER/huggingface/datasets"
export SHARED_SCRATCH_DIR="/scratch/izar/silly"

export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

SCRIPT_DIR=$SLURM_SUBMIT_DIR

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate telestyle

# ── Configure ────────────────────────────────────────────────────────────────
METHOD=${METHOD:-"reverse_telestyle"}   # reverse_telestyle | controlnet
STRENGTH=${STRENGTH:-"0.20"}            # ControlNet denoising strength (tuned)

STYLES=(watercolor oil_painting impressionism engraving)
ORIGINAL_DIR="$SHARED_SCRATCH_DIR/BlendedMVS/renamed"
SAVE_DIR="$SHARED_SCRATCH_DIR/BlendedMVS/telestyle_output/destylized_${METHOD}"
HOME_BACKUP_DIR="$HOME/destylize_output/${METHOD}"
# ─────────────────────────────────────────────────────────────────────────────

cd "/scratch/izar/$USER"

# Outer loop: scenes — so if the job is cancelled, every finished scene has all 4 styles done
for SCENE_DIR in "$SHARED_SCRATCH_DIR/BlendedMVS/renamed"/scene_*; do
    SCENE=$(basename "$SCENE_DIR")
    echo "════ $METHOD / $SCENE ════"

    for STYLE in "${STYLES[@]}"; do
        echo "  ── $STYLE"
        STYLIZED_DIR="$SHARED_SCRATCH_DIR/BlendedMVS/telestyle_output/${STYLE}"

        if [ "$METHOD" = "reverse_telestyle" ]; then
            python "$SCRIPT_DIR/reverse_telestyle_inference.py" \
                --stylized-dir "$STYLIZED_DIR" \
                --original-dir "$ORIGINAL_DIR" \
                --save-dir "$SAVE_DIR" \
                --home-backup-dir "$HOME_BACKUP_DIR" \
                --scene "$SCENE"

        elif [ "$METHOD" = "controlnet" ]; then
            python "$SCRIPT_DIR/controlnet_destylize_inference.py" \
                --stylized-dir "$STYLIZED_DIR" \
                --save-dir "$SAVE_DIR" \
                --home-backup-dir "$HOME_BACKUP_DIR" \
                --strength "$STRENGTH" \
                --scene "$SCENE"

        else
            echo "Unknown METHOD='$METHOD'. Choose 'reverse_telestyle' or 'controlnet'." >&2
            exit 1
        fi
    done
done
