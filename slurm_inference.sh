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
    "python inference.py --data_path './Continuous Data/test_data_continuous_no_R_for_Survival.csv'  --algorithm 'cql' --model_path '/work/yy408/SparkECMO/d3rlpy_logs/cql_training_continuous_no_R./configs/cql_cont_generated/cql_config_155.yaml_20241222035501/model_20000.d3' --data_type 'continuous_no_R'"
    #"python inference.py --data_path './Continuous Data/test_data_continuous_no_R_for_Survival.csv'  --algorithm 'cql' --model_path '/work/yy408/SparkECMO/d3rlpy_logs/cql_training_continuous_no_R./configs/cql_cont_generated/cql_config_57.yaml_20241222163835/model_10000.d3' --data_type 'continuous_no_R'"
    #"python inference.py --data_path './Continuous Data/test_data_continuous_Pos_for_Survival.csv'  --algorithm 'cql' --model_path '/work/yy408/SparkECMO/d3rlpy_logs/cql_training_continuous_Pos./configs/cql_cont_generated/cql_config_57.yaml_20241222101112/model_10000.d3' --data_type 'continuous_Pos'"
    #"python inference.py --data_path './Continuous Data/test_data_continuous_Pos_for_Survival.csv'  --algorithm 'cql' --model_path '/work/yy408/SparkECMO/d3rlpy_logs/cql_training_continuous_Pos./configs/cql_cont_generated/cql_config_8.yaml_20241222090629/model_10000.d3' --data_type 'continuous_Pos'"
    #"python inference.py --data_path './Continuous Data/test_data_continuous_no_R_for_Survival.csv'  --algorithm 'bcq' --model_path '/work/yy408/SparkECMO/d3rlpy_logs/bcq_training_continuous_no_R./configs/bcq_cont_generated/bcq_config_38.yaml_20241222182136/model_40000.d3' --data_type 'continuous_no_R'"
    #"python inference.py --data_path './Continuous Data/test_data_continuous_Pos_for_Survival.csv'  --algorithm 'bcq' --model_path '/work/yy408/SparkECMO/d3rlpy_logs/bcq_training_continuous_Pos./configs/bcq_cont_generated/bcq_config_5.yaml_20241222205509/model_10000.d3' --data_type 'continuous_Pos'"
    #"python inference.py --data_path './Discrete Data/test_data_discrete_no_R_for_Survival.csv'  --algorithm 'cql' --model_path '/work/yy408/SparkECMO/d3rlpy_logs/cql_training_discrete_no_R./configs/cql_disc_generated/cql_config_8.yaml_20241226014608/model_10000.d3' --data_type 'discrete_no_R'"
    #"python inference.py --data_path './Discrete Data/test_data_discrete_Pos_for_Survival.csv'  --algorithm 'cql' --model_path '/work/yy408/SparkECMO/d3rlpy_logs/cql_training_discrete_Pos./configs/cql_disc_generated/cql_config_8.yaml_20241226015231/model_10000.d3' --data_type 'discrete_Pos'"
    #"python inference.py --data_path './Discrete Data/test_data_discrete_no_R_for_Survival.csv'  --algorithm 'bcq' --model_path '/work/yy408/SparkECMO/d3rlpy_logs/bcq_training_discrete_no_R./configs/bcq_disc_generated/bcq_config_3.yaml_20241226015850/model_10000.d3' --data_type 'discrete_no_R'"
    #"python inference.py --data_path './Discrete Data/test_data_discrete_Pos_for_Survival.csv'  --algorithm 'bcq' --model_path '/work/yy408/SparkECMO/d3rlpy_logs/bcq_training_discrete_Pos./configs/bcq_disc_generated/bcq_config_3.yaml_20241226022645/model_10000.d3' --data_type 'discrete_Pos'"
    )
# 运行任务
eval "${tasks[$SLURM_ARRAY_TASK_ID]}"
