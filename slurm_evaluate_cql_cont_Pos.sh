#!/bin/bash
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:RTXA5000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yy408@duke.edu
#SBATCH --array=0-383
#SBATCH --mem=32G

source /hpc/group/xulab/yy408/miniconda3/etc/profile.d/conda.sh
#source /hpc/group/kamaleswaranlab/Conda_Environments/Stuart_Conda/etc/profile.d/conda.sh

conda activate d3rlpy


# Dynamically gather all model directories
model_base_dir="d3rlpy_logs/cql_training_continuous_Pos./configs/cql_cont_generated"
model_dirs=($(find "$model_base_dir" -mindepth 1 -maxdepth 1 -type d))

# Ensure that SLURM_ARRAY_TASK_ID doesn't exceed the number of directories
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#model_dirs[@]}" ]; then
  echo "Error: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID exceeds available model directories (${#model_dirs[@]})."
  exit 1
fi

# Select the task based on SLURM_ARRAY_TASK_ID
model_dir=${model_dirs[$SLURM_ARRAY_TASK_ID]}

# Run the corresponding task
python evaluate.py --is_continuous --model_dir "$model_dir" --data_path "./Continuous Data/val_data_continuous_Pos_for_Survival.csv" --algorithm "cql"
