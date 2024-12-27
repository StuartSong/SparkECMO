#!/bin/bash
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:RTXA5000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yy408@duke.edu
#SBATCH --mem=32G

source /hpc/group/xulab/yy408/miniconda3/etc/profile.d/conda.sh

python script.py