import d3rlpy
import pandas as pd
import numpy as np
import argparse




def process_data(data_path):
    df = pd.read_csv(data_path)
    data_dict = {}
    
    observations_columns = [
        'temperature', 'map_line', 'map_cuff', 'pulse', 'unassisted_resp_rate', 'end_tidal_co2', 'o2_flow_rate',
        'base_excess', 'bicarb_(hco3)', 'blood_urea_nitrogen_(bun)', 'creatinine', 'phosphorus', 'hemoglobin',
        'met_hgb', 'platelets', 'white_blood_cell_count', 'carboxy_hgb', 'alanine_aminotransferase_(alt)', 'ammonia',
        'aspartate_aminotransferase_(ast)', 'bilirubin_total', 'fibrinogen', 'inr', 'lactate_dehydrogenase', 'lactic_acid',
        'partial_prothrombin_time_(ptt)', 'prealbumin', 'lipase', 'b-type_natriuretic_peptide_(bnp)', 'partial_pressure_of_carbon_dioxide_(paco2)',
        'ph', 'saturation_of_oxygen_(sao2)', 'procalcitonin', 'erythrocyte_sedimentation_rate_(esr)', 'gcs_total_score',
        'best_map', 'pf_sp', 'pf_pa', 'spo2', 'partial_pressure_of_oxygen_(pao2)', 'rass_score', 'CAM_ICU'
    ]
    
    actions_columns = ['vent_rate_set', 'peep', 'fio2', 'flow', 'sweep']
    rewards_column = 'reward'

    
    observations = df[observations_columns].values
    actions = df[actions_columns].values
    rewards = df[rewards_column].values


    low = np.min(actions, axis=0)
    high = np.max(actions, axis=0)

    # Normalize actions to range [-1, 1]
    normalized_actions = 2 * (actions - low) / (high - low) - 1

    # Identifying terminal states based on a condition (example: 'csn' column value changes)
    terminals = np.zeros(len(df), dtype=bool)
    terminals[np.where(df['csn'].ne(df['csn'].shift(-1)))[0]] = True
    terminals = terminals.astype(int)
    
    # Creating the dictionary
    data_dict = {
        'observations': observations,
        'actions': normalized_actions,
        'rewards': rewards,
        'terminals': terminals
    }
    
    return data_dict




def main(args):
    data_dict = process_data(args.data_path)

    dataset = d3rlpy.datasets.MDPDataset(
        observations = data_dict["observations"],
        actions = data_dict["actions"],
        rewards =  data_dict["rewards"],
        terminals = data_dict["terminals"],
    )


    cql = d3rlpy.algos.CQLConfig(compile_graph=True).create(device='cuda:0')
    # Training configuration
    cql.fit(
        dataset=dataset,
        n_steps=100000,
        n_steps_per_epoch=1000,
        save_metrics=True,
        experiment_name="cql_training"
    )






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/hpc/group/xulab/yy408/SparkECMO/Continuous Data/train_data_continuous_no_R_for_Survival.csv')
    args = parser.parse_args()
    main(args)