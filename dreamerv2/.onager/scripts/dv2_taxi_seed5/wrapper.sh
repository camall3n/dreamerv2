#!/bin/bash

python -m onager.worker .onager/scripts/dv2_taxi_seed5/jobs.json $SLURM_ARRAY_TASK_ID 

