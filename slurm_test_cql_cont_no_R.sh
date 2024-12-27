#!/bin/bash
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:RTXA5000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yy408@duke.edu
#SBATCH --mem=32G

source /hpc/group/xulab/yy408/miniconda3/etc/profile.d/conda.sh
conda activate d3rlpy



# Run the corresponding task
python test.py --is_continuous --model_dir "d3rlpy_logs/cql_training_continuous_no_R./configs/cql_cont_generated/cql_config_155.yaml_20241222035501" --data_path "./Continuous Data/test_data_continuous_no_R_for_Survival.csv" --algorithm "cql"
