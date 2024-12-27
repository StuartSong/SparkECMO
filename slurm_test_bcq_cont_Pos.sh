#!/bin/bash
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:RTXA5000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yy408@duke.edu
#SBATCH --mem=32G

source /hpc/group/xulab/yy408/miniconda3/etc/profile.d/conda.sh
conda activate d3rlpy



# Run the corresponding task
python test.py --is_continuous --model_dir "d3rlpy_logs/bcq_training_continuous_Pos./configs/bcq_cont_generated/bcq_config_5.yaml_20241222205509" --data_path "./Continuous Data/test_data_continuous_Pos_for_Survival.csv" --algorithm "bcq"
