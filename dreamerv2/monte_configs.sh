#!/bin/bash

#SBATCH -p gpu --gres=gpu:1
#SBATCH --time=24:00:00

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=120G

#SBATCH -J dv2_monte

#SBATCH -o monte_configs.out
#SBATCH -e monte_configs.err


#source /users/ssunda11/.conda/envs
#source activate dreamer_env
eval "$(conda shell.bash hook)"
conda activate dreamer_env
python3 train.py --configs monte --num_steps 25000000
