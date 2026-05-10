#!/bin/bash
#SBATCH --job-name=evaluate    
#SBATCH --time=02:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:1                   
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4            
#SBATCH --output=evaluate.out    
#SBATCH --error=evaluate.err    

# change baseline_name to the baseline you evaluate (oil_painting ect)

python -m eval_pipeline.runner \
  --data_dir /scratch/izar/silly/BlendedMVS/renamed \
  --checkpoint facebook/map-anything \
  --baseline_name photographs