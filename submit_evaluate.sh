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

# change baseline_name to the baseline you evaluate (oil_painting ect)

#!/bin/bash

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate mapanything

SCRIPT_DIR=$SLURM_SUBMIT_DIR
cd $SCRIPT_DIR

# use "photographs" for the original path, else use the wanted style:
baseline="photographs" # or ex. "engraving"
max_scenes=100

if [ "$baseline" = "photographs" ]; then
    # original images
    dir="/scratch/izar/silly/BlendedMVS/renamed/"
else
    # stylized images
    dir="/scratch/izar/silly/BlendedMVS/telestyle_output/${baseline}/"

    # Run the python script ONLY for stylized baselines, passing the style name as an argument
    python "./transfer_folders.py" "$baseline"
fi



python -m eval_pipeline.runner \
  --data_dir "$dir" \
  --checkpoint facebook/map-anything \
  --baseline_name "${baseline}_${max_scenes}_0blocks" \
  --max_scenes "$max_scenes" 




# Evaluate adaIn training results with multiple epochs
#for nbr in 0 1 2 3 4 5 6 7 8 9
#do
#
#  # Hook from the end (blocks 24-nbr..23)
#  python -m eval_pipeline.runner \
#    --data_dir "$dir" \
#    --checkpoint facebook/map-anything \
#    --baseline_name "${baseline}_${max_scenes}_1blocks_end_epoch${nbr}" \
#    --max_scenes "$max_scenes" \
#    --encoder_block_prefix encoder.model.blocks \
#    --norm_num_blocks "$nbr" \
#    --norm_from_end\
#    --norm_affine \
#    --adapter_weights "./eval_pipeline/weights/adapter_engraving_1_end_${nbr}.pth"
#
#done


# Evaluate grayscale and film results

#baseline=oil_painting
#
#if [ "$baseline" = "photographs" ]; then
#    # original images
#    dir="/scratch/izar/silly/BlendedMVS/renamed/"
#else
#    # stylized images
#    dir="/scratch/izar/silly/BlendedMVS/telestyle_output/${baseline}/"
#
#    # Run the python script ONLY for stylized baselines
#    python ./transfer_folders.py "$baseline"
#fi
#
#python -m eval_pipeline.runner \
#  --data_dir "$dir" \
#  --checkpoint facebook/map-anything \
#  --baseline_name "${baseline}_${max_scenes}_grayscale" \
#  --max_scenes "$max_scenes" \
#  --modification "grayscale"
#
#done

#for baseline in photographs engraving watercolor oil_painting impressionism
#do
#
#if [ "$baseline" = "photographs" ]; then
#    # original images
#    dir="/scratch/izar/silly/BlendedMVS/renamed/"
#else
#    # stylized images
#    dir="/scratch/izar/silly/BlendedMVS/telestyle_output/${baseline}/"
#
#    # Run the python script ONLY for stylized baselines
#    python ./transfer_folders.py "$baseline"
#fi
#
#python -m eval_pipeline.runner \
#  --data_dir "$dir" \
#  --checkpoint facebook/map-anything \
#  --baseline_name "${baseline}_${max_scenes}_film" \
#  --max_scenes "$max_scenes" \
#  --modification "film"
#
#done



## Example of how to run with blocks statistical mods
#for nbr in 1 2 3 4 8 12 16 20 24
#do
#  python -m eval_pipeline.runner \
#    --data_dir "$dir" \
#    --checkpoint facebook/map-anything \
#    --baseline_name "${baseline}_${max_scenes}_${nbr}blocks_fixed" \
#    --max_scenes "$max_scenes" \
#    --encoder_block_prefix encoder.model.blocks \
#    --norm_num_blocks "$nbr"
#
#      # New: hook from the end (blocks 24-nbr..23)
#  python -m eval_pipeline.runner \
#    --data_dir "$dir" \
#    --checkpoint facebook/map-anything \
#    --baseline_name "${baseline}_${max_scenes}_${nbr}blocks_end" \
#    --max_scenes "$max_scenes" \
#    --encoder_block_prefix encoder.model.blocks \
#    --norm_num_blocks "$nbr" \
#    --norm_from_end
#done
#

