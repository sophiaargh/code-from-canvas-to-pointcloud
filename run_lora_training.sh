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
conda activate mapanything

export PYTHONPATH="/home/qsandoz/visual-intelligence:${PYTHONPATH}"

# TODO: set style name to your assigned style
# Camille: watercolor | Sophia: oil_painting | Quentin: impressionism | Emilien: engraving
STYLE="impressionism"

DATASET_ROOT="$SHARED_SCRATCH_DIR/BlendedMVS/renamed"
STYLED_ROOT="$SHARED_SCRATCH_DIR/BlendedMVS/telestyle_output"
LORA_OUT_DIR="$SHARED_SCRATCH_DIR/lora_checkpoints/${STYLE}"

mkdir -p "$LORA_OUT_DIR"
mkdir -p "$SCRIPT_DIR/logs"

cd "/home/qsandoz/visual-intelligence"

#echo "=== LoRA sanity check ==="
#python check_lora.py || { echo "check_lora.py failed — aborting"; exit 1; }
#echo "=== Sanity check passed, starting training ==="

python -m mapanything.train.train_lora \
    --base_checkpoint facebook/map-anything \
    --lora_out_dir "$LORA_OUT_DIR" \
    --style_name "$STYLE" \
    --styled_root "$STYLED_ROOT" \
    --dataset_root "$DATASET_ROOT" \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lr 1e-4 \
    --batch_size 1 \
    --num_views 2 \
    --num_workers 8 \
    --max_steps 5000 \
    --save_every 500 \
    --use_amp
