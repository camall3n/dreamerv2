#!/bin/bash

#SBATCH -p gpu --gres=gpu:1
#SBATCH --time=24:00:00

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=32G

#SBATCH -J dv2_monte

#SBATCH -o taxi_original.out
#SBATCH -e taxi_original.err

source venv/bin/activate
python3.9 train.py --configs monte --num_steps 1000000 --seed 0 --logdir ../logs/taxi_50mil_binaryhead_original_metrics --save_step 0 --action_repeat 4
