#!/bin/bash

#SBATCH -p 3090-gcondo --gres=gpu:1
#SBATCH --time=24:00:00

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=33G

#SBATCH -J dv2_monte

#SBATCH -o taxi_step.out
#SBATCH -e taxi_step.err

source venv/bin/activate
python3.9 train.py --configs monte --num_steps 1000000 --seed 0 --logdir ../logs/taxi_50mil_binaryhead_step_metrics --save_step 1 --action_repeat 4