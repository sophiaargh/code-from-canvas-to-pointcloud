#!/bin/bash
#SBATCH --job-name=evaluate_vggt       # Change as needed
#SBATCH --time=02:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:1                   # Request 2 GPUs
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4               # Adjust CPU allocation if needed
#SBATCH --output=evaluate.out    # Output log file
#SBATCH --error=evaluate.err     # Error log file



python evaluate_vggt.py --baseline_name photographs