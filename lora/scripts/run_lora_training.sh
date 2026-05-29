#!/bin/bash
#SBATCH --job-name=mapanything_lora
#SBATCH --output=logs/lora_%j.log
#SBATCH --error=logs/lora_%j.err
#SBATCH --time=12:00:00
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

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

SCRIPT_DIR=$SLURM_SUBMIT_DIR

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate visual-intel

export PYTHONPATH="/home/$USER/code-from-canvas-to-pointcloud:${PYTHONPATH}"

DATASET_ROOT="$SHARED_SCRATCH_DIR/BlendedMVS/renamed"
STYLED_ROOT="$SHARED_SCRATCH_DIR/BlendedMVS/telestyle_output"
LORA_CONSISTENCY_OUT_DIR="$SHARED_SCRATCH_DIR/lora_checkpoints/mixed_styles_gray_consistency"
LORA_OUT_DIR="$SHARED_SCRATCH_DIR/lora_checkpoints/mixed_styles_gray"

mkdir -p "$LORA_OUT_DIR"
mkdir -p "$LORA_CONSISTENCY_OUT_DIR"
mkdir -p "$SCRIPT_DIR/logs"

python -m lora.train_lora \
    --base_checkpoint facebook/map-anything \
    --lora_out_dir "$LORA_OUT_DIR" \
    --style_names engraving impressionism oil_painting watercolor \
    --n_styled 2 \
    --styled_root "$STYLED_ROOT" \
    --dataset_root "$DATASET_ROOT" \
    --lora_rank 8 \
    --lora_alpha 8 \
    --lora_dropout 0.05 \
    --lr 5e-5 \
    --batch_size 1 \
    --num_views 4 \
    --num_workers 8 \
    --max_steps 5000 \
    --save_every 500 \
    --use_amp \
    --gradient_checkpointing \
    --grayscale \
    --consistency_weight 0.0 \
    --resolution 392 280

python -m lora.train_lora \
    --base_checkpoint facebook/map-anything \
    --lora_out_dir "$LORA_CONSISTENCY_OUT_DIR" \
    --style_names engraving impressionism oil_painting watercolor \
    --n_styled 2 \
    --styled_root "$STYLED_ROOT" \
    --dataset_root "$DATASET_ROOT" \
    --lora_rank 8 \
    --lora_alpha 8 \
    --lora_dropout 0.05 \
    --lr 5e-5 \
    --batch_size 1 \
    --num_views 4 \
    --num_workers 8 \
    --max_steps 5000 \
    --save_every 500 \
    --use_amp \
    --gradient_checkpointing \
    --grayscale \
    --consistency_weight 0.1 \
    --resolution 392 280
