#!/bin/bash
#SBATCH --job-name=train_adain    
#SBATCH --time=08:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:1                   
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4            
#SBATCH --output=eval_pipeline/logs/train_adain_%j.log
#SBATCH --error=eval_pipeline/logs/train_adain_%j.err  

# change baseline_name to the baseline you evaluate (oil_painting ect)

#!/bin/bash

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate mapanything

SCRIPT_DIR=$SLURM_SUBMIT_DIR
cd $SCRIPT_DIR

# use "photographs" for the original path, else use the wanted style:
baseline="engraving" # or ex. "engraving"
epochs=10
nbr=1

python -m eval_pipeline.train_adaIN_epoch \
  --style "$baseline" \
  --norm_num_blocks "$nbr" \
  --epochs "$epochs"\
  --norm_from_end
