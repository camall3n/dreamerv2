#!/bin/bash

python -m onager.worker .onager/scripts/exp01/jobs.json $SLURM_ARRAY_TASK_ID 

