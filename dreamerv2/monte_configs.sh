#!/bin/bash

#SBATCH -p gpu --gres=gpu:1
#SBATCH --time=24:00:00

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=120G

#SBATCH -J dv2_monte

#SBATCH -o monte_configs.out
#SBATCH -e monte_configs.err

source venv/bin/activate
python3.9 train.py --configs monte --num_steps 40000
