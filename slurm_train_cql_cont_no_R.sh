#!/bin/bash
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:RTXA5000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yy408@duke.edu
#SBATCH --array=0-383
#SBATCH --mem=32G

source /hpc/group/xulab/yy408/miniconda3/etc/profile.d/conda.sh
conda activate d3rlpy

# Run the Python script with the corresponding parameters

CONFIG_DIR="./configs/cql_cont_generated"
# Generate the correct config file name
CONFIG_FILE="${CONFIG_DIR}/cql_config_${SLURM_ARRAY_TASK_ID}.yaml"

python train.py --config "$CONFIG_FILE" --is_continuous