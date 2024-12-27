#!/bin/bash
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:RTXA5000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yy408@duke.edu
#SBATCH --array=0
#SBATCH --mem=32G

source /hpc/group/xulab/yy408/miniconda3/etc/profile.d/conda.sh
conda activate d3rlpy



# Run the corresponding task
python test.py --model_dir "d3rlpy_logs/cql_training_discrete_no_R./configs/cql_disc_generated/cql_config_8.yaml_20241226014608" --data_path "./Discrete Data/test_data_discrete_no_R_for_Survival.csv" --algorithm "cql"
