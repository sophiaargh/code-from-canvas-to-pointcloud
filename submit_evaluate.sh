#!/bin/bash
#SBATCH --job-name=evaluate    
#SBATCH --time=02:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:1                   
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4            
#SBATCH --output=eval_pipeline/logs/evaluate_%j.log
#SBATCH --error=eval_pipeline/logs/evaluate_%j.err  

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate mapanything

export HF_DATASETS_CACHE="/scratch/izar/$USER/huggingface/datasets"

SCRIPT_DIR=$SLURM_SUBMIT_DIR
cd $SCRIPT_DIR

baseline="photographs" # or e.g. "engraving", "watercolor", "oil_painting", "impressionism"
max_scenes=100

python -m eval_pipeline.runner \
  --checkpoint facebook/map-anything \
  --baseline_name "${baseline}" \
  --style "${baseline}" \
  --max_scenes "$max_scenes"

# Evaluate adaIn training results with multiple epochs
#for nbr in 0 1 2 3 4 5 6 7 8 9
#do
#
#  python -m eval_pipeline.runner \
#    --checkpoint facebook/map-anything \
#    --baseline_name "${baseline}_${max_scenes}_1blocks_end_epoch${nbr}" \
#    --max_scenes "$max_scenes" \
#    --encoder_block_prefix encoder.model.blocks \
#    --norm_num_blocks "$nbr" \
#    --norm_from_end \
#    --norm_affine \
#    --style "${baseline}" \
#    --adapter_weights "./eval_pipeline/weights/adapter_engraving_1_end_${nbr}.pth"
#
#done

# Evaluate grayscale and film results
#baseline=oil_painting
#
#python -m eval_pipeline.runner \
#  --checkpoint facebook/map-anything \
#  --baseline_name "${baseline}_${max_scenes}_grayscale" \
#  --max_scenes "$max_scenes" \
#    --style "${baseline}" \
#  --modification "grayscale"

#for baseline in photographs engraving watercolor oil_painting impressionism
#do
#
#  python -m eval_pipeline.runner \
#    --checkpoint facebook/map-anything \
#    --baseline_name "${baseline}_${max_scenes}_film" \
#    --max_scenes "$max_scenes" \
#    --style "${baseline}" \
#    --modification "film"
#
#done

# Example of how to run with blocks statistical mods
#for nbr in 1 2 3 4 8 12 16 20 24
#do
#  python -m eval_pipeline.runner \
#    --checkpoint facebook/map-anything \
#    --baseline_name "${baseline}_${max_scenes}_${nbr}blocks_fixed" \
#    --max_scenes "$max_scenes" \
#    --encoder_block_prefix encoder.model.blocks \
#    --style "${baseline}" \
#    --norm_num_blocks "$nbr"
#
#  python -m eval_pipeline.runner \
#    --checkpoint facebook/map-anything \
#    --baseline_name "${baseline}_${max_scenes}_${nbr}blocks_end" \
#    --max_scenes "$max_scenes" \
#    --encoder_block_prefix encoder.model.blocks \
#    --norm_num_blocks "$nbr" \
#    --style "${baseline}" \
#    --norm_from_end
#done