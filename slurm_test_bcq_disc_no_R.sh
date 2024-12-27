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
python test.py --model_dir "/d3rlpy_logs/bcq_training_discrete_no_R./configs/bcq_disc_generated/bcq_config_3.yaml_20241226015850" --data_path "./Discrete Data/test_data_discrete_no_R_for_Survival.csv" --algorithm "bcq"
