import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import multiprocessing
import joblib
from scipy.stats import norm
import sys
import matplotlib.pyplot as plt

def median_and_confidence_interval(array):
    array = np.array(array)
    median = np.median(array)
    max = np.max(array)
    array.sort()
    
    if len(array) < 3:
        lower_bound = array[0]
        upper_bound = array[-1]
    else:
        std_error = np.std(array, ddof=1) / np.sqrt(len(array))
        margin_of_error = std_error * norm.ppf(0.975)
        lower_bound = median - margin_of_error
        upper_bound = median + margin_of_error
    
    return median, max, (lower_bound, upper_bound)

def calculate_behavior_policy(dataset):
    # Get the unique states and actions
    unique_states = dataset['state'].unique()
    unique_actions = dataset['action'].unique()

    # Initialize the behavior policy as a DataFrame with zeros
    behavior_policy = pd.DataFrame(0, index=unique_states, columns=unique_actions)

    # Count the occurrences of each state-action pair
    state_action_counts = dataset.groupby(['state', 'action']).size().unstack(fill_value=0)

    # Calculate the behavior policy
    for state in unique_states:
        total_actions = state_action_counts.loc[state].sum()
        for action in unique_actions:
            if total_actions > 0:
                behavior_policy.loc[state, action] = state_action_counts.loc[state, action] / total_actions

    return behavior_policy

def reward_func(data_for_rl):
    rewards = []
    for i in range(1, len(data_for_rl)-1):

        short_term_reward = -(data_for_rl.iloc[i]['short_term_reward'] - data_for_rl.iloc[i-1]['short_term_reward'])/16

        if data_for_rl.iloc[i]['long_term_reward'] == -1:
            long_term_reward = data_for_rl.iloc[i]['long_term_reward']*i/(len(data_for_rl) - 2)
        else:
            long_term_reward = 0
        reward = short_term_reward + long_term_reward
        rewards.append(reward)
        
    return rewards

def weighted_importance_sampling(data_for_rl, Q_table, b_policy, epsilon=0.1):
    weights = []
    weighted_rewards = []

    for i in range(1, len(data_for_rl)-1):
        current_state = data_for_rl.iloc[i]['state']
        action = data_for_rl.iloc[i]['action']
        
        # Calculate the target policy probability for the current action
        best_action = Q_table.loc[current_state].idxmax()
        num_actions = Q_table.shape[1]
        target_prob = (1 - epsilon) + (epsilon / num_actions) if action == best_action else 0.1/(num_actions)
        
        # Calculate the behavior policy probability for the current action
        behavior_prob = b_policy.loc[current_state, action]
        
        # Calculate the importance weight
        weight = target_prob / behavior_prob if behavior_prob > 0 else 0
        # print(target_prob, behavior_prob)
        weights.append(weight)
    rewards = reward_func(data_for_rl)
    weighted_rewards = np.array(weights) * np.array(rewards)
    # sys.exit()
    # Adjust lists to match the original structure
    weights.insert(0, 0)
    np.insert(weighted_rewards, 0, 0)
    
    return weights, weighted_rewards

def load_and_predict(new_data, cluster_num, model_index):
    # Load the KMeans model
    kmeans = joblib.load(f'cluster_{cluster_num}_models/kmeans_{model_index}.pkl')

    # Predict the cluster states
    new_states = kmeans.predict(new_data) + 1

def wis_all_models(clustered_test, merged_test, num_models):
    rewards_sum_total = []
    rewards_avg_total = []
    medians = []
    maxs = []
    
    medians_avg = []
    lower_upper_bounds_avg = []
    maxs_avg = []
    
    max_overall = -100
    
    for i in tqdm(range(num_models)):
        rewards_sum_model = []
        rewards_avg_model = []
        kmeans = joblib.load(f'cluster_50_models/kmeans_{i}.pkl')
        new_states = kmeans.predict(clustered_test.iloc[:,:-1]) + 1
        merged_test['state'] = new_states
        b_policy = calculate_behavior_policy(merged_test)
        
        q_table = pd.read_csv(f'cluster_50_q_table/q_table_{i}.csv')
        q_table.index = q_table.index+1
        q_table = q_table.loc[:, (q_table != 0).any(axis=0)]
    
        for csn in merged_test.csn.unique():
            merged_test_selected = merged_test[merged_test.csn == csn]
            weights, weighted_rewards = weighted_importance_sampling(merged_test_selected, q_table, b_policy)
            
            rewards_sum_model.append(np.sum(weighted_rewards))
            rewards_avg_model.append(np.mean(weighted_rewards))
            
        # print(np.sum(rewards_sum_model))
        rewards_sum_total.append(np.sum(rewards_sum_model))
        
        rewards_avg_total.append(np.sum(rewards_avg_model))
        median, max, (lower_bound, upper_bound) = median_and_confidence_interval(rewards_avg_total)
        medians_avg.append(median)
        maxs_avg.append(max)
        lower_upper_bounds_avg.append((lower_bound, upper_bound))
    
        if max> max_overall:
            max_overall = max
            max_overall_index = i
    print(f"The model with maximum reward is Model {max_overall_index}")
    return max_overall_index, lower_upper_bounds_avg, medians_avg, maxs_avg

def calculate_clinician_reward(merged_test):
    clinician_reward = []
    for csn in merged_test.csn.unique():
        data_for_rl = merged_test[merged_test.csn == csn]
        avg_reward_csn = reward_func(data_for_rl)
        clinician_reward.append(np.mean(avg_reward_csn))
    return clinician_reward