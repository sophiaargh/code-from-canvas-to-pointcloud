#!/bin/bash
#SBATCH --job-name=destylize_cmp
#SBATCH --output=logs/destylize_cmp_%j.log
#SBATCH --error=logs/destylize_cmp_%j.err
#SBATCH --time=4:00:00
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

SCRIPT_DIR=$SLURM_SUBMIT_DIR

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate telestyle

cd "/scratch/izar/$USER"

python "$SCRIPT_DIR/eval_destylize_comparison.py" \
    --data-root "/scratch/izar/silly/BlendedMVS" \
    --scenes scene_0 scene_1 scene_3 scene_4 scene_8 scene_9 scene_10 scene_11 scene_12 scene_13 \
    --max-frames 8 \
    --out "$SCRIPT_DIR/evaluation_results/destylize_comparison_grid.csv"
