#!/bin/bash -l
#SBATCH --job-name=v4_vs_baseline
#SBATCH --output=./logs/%x.%j.out # %x.%j expands to slurm JobName.JobID
#SBATCH --error=./logs/%x.%j.err # prints the error message
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --qos=standard
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=32GB     # Maximum allowable memory per CPU
#SBATCH --account=senjutib
#SBATCH --time=12:00:00      # D-HH:MM:SS

# Purge any module loaded by default
# module purge > /dev/null 2>&1
# module load wulver
# module load Miniforge3
# source conda.sh
conda activate DL
srun bash ./run_experiments.sh
# srun python main.py