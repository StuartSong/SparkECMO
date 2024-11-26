import d3rlpy
import pandas as pd
import numpy as np
import argparse



def encode_discrete_action(actions, ranges):
    """
    Encode multidimensional discrete actions into a single index.
    
    Args:
        actions (ndarray): Array of shape (N, D), where D is the number of action dimensions.
        ranges (list): List of integers representing the range of each dimension.
        
    Returns:
        encoded_actions (ndarray): Encoded action indices of shape (N,).
    """
    # Ensure ranges is an ndarray for broadcasting
    ranges = np.array(ranges)
    multipliers = np.cumprod([1] + ranges[::-1].tolist()[:-1])[::-1]
    encoded_actions = np.sum(actions * multipliers, axis=1)
    return encoded_actions


def decode_discrete_action(indices, ranges):
    """
    Decode a single index back to multidimensional discrete actions.
    
    Args:
        indices (ndarray): Array of encoded action indices.
        ranges (list): List of integers representing the range of each dimension.
        
    Returns:
        decoded_actions (ndarray): Decoded multidimensional actions of shape (N, D).
    """
    ranges = np.array(ranges)
    multipliers = np.cumprod([1] + ranges[::-1].tolist()[:-1])[::-1]
    decoded_actions = []
    for index in indices:
        action = []
        for m in multipliers:
            action.append(index // m)
            index %= m
        decoded_actions.append(action)
    return np.array(decoded_actions)




def process_data(args):
    df = pd.read_csv(args.data_path)
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
    global low, high
    if args.is_continuous:
        low = np.min(actions, axis=0)
        high = np.max(actions, axis=0)

        # Normalize actions to range [-1, 1]
        normalized_actions = 2 * (actions - low) / (high - low) - 1
    else:
        # Encode discrete actions
        ranges = [4, 5, 4, 5, 5]
        normalized_actions = encode_discrete_action(actions, ranges)
        print(normalized_actions)

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

    data_dict = process_data(args)
    dataset = d3rlpy.datasets.MDPDataset(
        observations = data_dict["observations"],
        actions = data_dict["actions"],
        rewards =  data_dict["rewards"],
        terminals = data_dict["terminals"],
    )

    if args.algorithm == 'cql' and args.is_continuous:
        cql = d3rlpy.load_learnable(args.model_path)
        # Training configuration
        action = cql.predict(data_dict["observations"])
        action = (action + 1) / 2 * (high - low) + low
        pd.DataFrame(action, columns=['vent_rate_set', 'peep', 'fio2', 'flow', 'sweep']).to_csv(args.data_type+'cql_restored_actions.csv', index=False)
    elif args.algorithm == 'cql' and not args.is_continuous:
        cql = d3rlpy.load_learnable(args.model_path)
        # Training configuration
        action = cql.predict(data_dict["observations"])
        action = decode_discrete_action(action, [4, 5, 4, 5, 5])
        pd.DataFrame(action, columns=['vent_rate_set', 'peep', 'fio2', 'flow', 'sweep']).to_csv(args.data_type+'cql_restored_actions.csv', index=False)
    elif args.algorithm == 'bcq' and args.is_continuous:
        bcq = d3rlpy.load_learnable(args.model_path)
        # Training configuration
        action = bcq.predict(data_dict["observations"])
        action = (action + 1) / 2 * (high - low) + low
        pd.DataFrame(action, columns=['vent_rate_set', 'peep', 'fio2', 'flow', 'sweep']).to_csv(args.data_type+'bcq_restored_actions.csv', index=False)
    elif args.algorithm == 'bcq' and not args.is_continuous:
        bcq = d3rlpy.load_learnable(args.model_path)
        # Training configuration
        action = bcq.predict(data_dict["observations"])
        action = decode_discrete_action(action, [4, 5, 4, 5, 5])
        pd.DataFrame(action, columns=['vent_rate_set', 'peep', 'fio2', 'flow', 'sweep']).to_csv(args.data_type+'bcq_restored_actions.csv', index=False)
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='continuous_no_R')
    parser.add_argument('--data_path', type=str, default='./Continuous Data/train_data_continuous_no_R_for_Survival.csv')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--is_continuous', type=bool, default=True)
    parser.add_argument('--algorithm', type=str, default='cql')
    args = parser.parse_args()
    main(args)