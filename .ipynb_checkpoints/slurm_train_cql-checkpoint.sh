#!/bin/bash
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:RTXA5000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yy408@duke.edu
#SBATCH --array=0-3
#SBATCH --mem=32G

source /hpc/group/xulab/yy408/miniconda3/etc/profile.d/conda.sh
conda activate d3rlpy

# Define arrays for parameters
data_types=('discrete_no_R' 'discrete_Pos' 'discrete_no_R' 'discrete_Pos')
data_paths=(
    './Discrete Data/train_data_discrete_no_R_for_Survival.csv'
    './Discrete Data/train_data_discrete_Pos_for_Survival.csv'
    './Discrete Data/train_data_discrete_no_R_for_Survival.csv'
    './Discrete Data/train_data_discrete_Pos_for_Survival.csv'
)
algorithms=('cql' 'cql' 'bcq' 'bcq')

# Get parameters for the current task
index=$SLURM_ARRAY_TASK_ID
data_type=${data_types[$index]}
data_path=${data_paths[$index]}
continuous=${is_continuous[$index]}
algorithm=${algorithms[$index]}

# Run the Python script with the corresponding parameters
python train_discrete.py --data_type "$data_type" --data_path "$data_path" --algorithm "$algorithm"
