#!/bin/bash
#SBATCH --account=cs_ga_3033_102-2025sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=9:00:00
#SBATCH --requeue
#SBATCH --job-name=diffuser-a100
#SBATCH --output=job.out
#SBATCH --error=job.err

cd ~/DiffuserV2
conda activate diffuser
source extras/setup_env.sh

python ~/gpu_utilize.py > /dev/null 2>&1 &
GPUPID=$!

python -m scripts.train --config config.maze2d --dataset maze2d-umaze-v1

kill $GPUPID