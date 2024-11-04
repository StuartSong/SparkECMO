import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import joblib
import warnings
warnings.filterwarnings("ignore")


def epsilon_greedy_policy(state, Q_table, epsilon = 0.1):
    """ Selects action using epsilon-greedy policy. """
    if random.uniform(0, 1) < epsilon:
        # Explore: select a random action
        return random.choice(list(Q_table.columns))
    else:
        # Exploit: select the best action based on current Q-values
        return Q_table.loc[state].idxmax()

def q_learning(data_for_rl, unique_action_space, lr = 0.1, discount=0.9):
    # Extract unique states and actions
    unique_states = data_for_rl['state'].unique()
    unique_actions = unique_action_space['action_number'].unique()
    
    # Initialize the Q-table
    Q_table = pd.DataFrame(data=np.zeros((len(unique_states), len(unique_actions))),
                           index=unique_states, columns=unique_actions)
    
    # Number of episodes to run (using the number of unique CSNs in the dataset)
    num_episodes = data_for_rl['csn'].nunique()
    
    # Simulation loop
    for episode in range(num_episodes):
        # Filter the data for the current episode based on CSN
        episode_data = data_for_rl[data_for_rl['csn'] == data_for_rl['csn'].unique()[episode]]
        
        # Iterate through each step of the episode
        for i in range(1, len(episode_data) - 1):
            current_state = episode_data.iloc[i]['state']
            action = episode_data.iloc[i]['action']

            short_term_reward = -(episode_data.iloc[i]['short_term_reward'] - episode_data.iloc[i-1]['short_term_reward'])/16
            if episode_data.iloc[i]['long_term_reward'] == -1:
                long_term_reward = episode_data.iloc[i]['long_term_reward']*i/(len(episode_data) - 2)
            else:
                long_term_reward = 0
            reward = short_term_reward + long_term_reward
            
            next_state = episode_data.iloc[i + 1]['state']
            
            # Select an action for the next state using ε-greedy policy
            next_action = epsilon_greedy_policy(next_state, Q_table)
            
            # Update Q-table using the Q-learning formula
            best_next_q = Q_table.loc[next_state, next_action] if next_state in Q_table.index else 0
            Q_table.at[current_state, action] = (Q_table.at[current_state, action] +
                lr * (reward + discount * best_next_q - Q_table.at[current_state, action]))
    return Q_table
    
def get_best_action(current_state, Q_table):
    """ Retrieve the best action for the current state from the Q-table. """
    if current_state in Q_table.index:
        # Find the action with the maximum Q-value in the current state
        best_action = Q_table.loc[current_state].idxmax()
        return best_action
    else:
        # If the current state is not in the table, choose a default or random action
        return 'No state recognized' 

def process_cluster(i, data_clustered, merge_data, cluster_num, unique_action_space):
    # Get unique CSNs
    unique_csn = data_clustered['csn'].unique()
    train_csn, test_csn = train_test_split(unique_csn, test_size=0.2, random_state=np.random.randint(0, 10000))
    
    train_cluster = data_clustered[data_clustered['csn'].isin(train_csn)]
    test_cluster = data_clustered[data_clustered['csn'].isin(test_csn)]
    
    kmeans = KMeans(n_clusters=cluster_num, random_state=42)
    train_data = merge_data[merge_data['csn'].isin(train_csn)]
    train_state = kmeans.fit_predict(train_cluster.iloc[:, :-1]) + 1
    
    train_data["state"] = train_state
    test_data = merge_data[merge_data['csn'].isin(test_csn)]
    test_state = kmeans.predict(test_cluster.iloc[:, :-1]) + 1
    test_data["state"] = test_state
    
    Q_table_result = q_learning(train_data, unique_action_space)
    RL_action = [get_best_action(current_state, Q_table_result) for current_state in test_data.state]
    test_data["agent_action"] = RL_action

    # Save test data
    test_data.to_csv(f'cluster_{cluster_num}/test_data_{i}.csv', index=False)
    Q_table_result.to_csv(f'cluster_{cluster_num}_q_table/q_table_{i}.csv', index=False)

    # Save KMeans model
    joblib.dump(kmeans, f'cluster_{cluster_num}_models/kmeans_{i}.pkl')



def parallel_q_learning(data_clustered, merge_data, cluster_num, unique_action_space, num_models):
    os.makedirs(f'cluster_{cluster_num}', exist_ok=True)
    os.makedirs(f'cluster_{cluster_num}_q_table', exist_ok=True)
    os.makedirs(f'cluster_{cluster_num}_models', exist_ok=True)
    
    with ThreadPoolExecutor(max_workers=25) as executor:
        futures = [executor.submit(process_cluster, i, data_clustered, merge_data, cluster_num, unique_action_space) for i in range(num_models)]

        for future in tqdm(as_completed(futures), total=num_models):
            continue

def optimal_model(i, merged_test, clustered_test,cluster_num):
    kmeans = joblib.load(f'cluster_{cluster_num}_models/kmeans_{i}.pkl')
    new_states = kmeans.predict(clustered_test.iloc[:,:-1]) + 1
    merged_test['state'] = new_states
    
    q_table = pd.read_csv(f'cluster_{cluster_num}_q_table/q_table_{i}.csv')
    q_table.index = q_table.index+1
    best_action = [q_table.loc[current_state].idxmax() for current_state in merged_test['state']]
    merged_test["agent_action"] = best_action

    return merged_test