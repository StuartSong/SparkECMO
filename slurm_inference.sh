#!/bin/bash
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:RTXA5000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yy408@duke.edu
#SBATCH --array=0
#SBATCH --mem=32G

# 加载环境
source /hpc/group/xulab/yy408/miniconda3/etc/profile.d/conda.sh
conda activate d3rlpy

# 定义任务列表
tasks=(
    # CQL Continuous
    # CQL Discrete
    "python inference.py --data_path './Continuous Data/test_data_continuous_no_R_for_Survival.csv'  --algorithm 'cql' --model_path '/hpc/group/xulab/yy408/SparkECMO/d3rlpy_logs/cql_training_continuous_no_R./configs/generated/cql_config_209.yaml_20241216181318/model_20000.d3' --data_type 'continuous_no_R'"
)

# 运行任务
eval "${tasks[$SLURM_ARRAY_TASK_ID]}"
