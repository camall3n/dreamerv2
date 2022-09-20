#!/bin/bash

#SBATCH -p gpu --gres=gpu:1
#SBATCH --time=00:20:00

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=32G

#SBATCH -J dv2_taxi_test

#SBATCH -o taxi_test.out
#SBATCH -e taxi_test.err

source venv/bin/activate
python3.9 train.py --configs monte --steps 5000 --prefill 1000 --eval_every 1000 --log_every 1000 --seed 0 --logdir ../logs/taxi_test --save_step False --action_repeat 4
