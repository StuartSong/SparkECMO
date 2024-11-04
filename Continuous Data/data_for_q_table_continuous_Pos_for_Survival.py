#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


action_space = pd.read_csv("../Discrete Data/non_discritised_data.csv")
action_space


# In[3]:


merge_data = pd.read_csv("../Discrete Data/ECMO_data_clustered.csv")
merge_data


# In[4]:


con_data = pd.concat([merge_data,action_space],axis=1)
con_data.rename(columns={"ECMO Pump Flow": "flow","ECMO Sweep Gas Flow": "sweep"},inplace=True)
con_data
# con_data.to_csv("before_reward.csv",index=False)


# In[5]:


sofa_scores = pd.read_csv("../Discrete Data/data_for_RL.csv")
sofa_scores.rename(columns={"short_term_reward": "sofa_sigma"},inplace=True)
sofa_scores['long_term_reward'] = sofa_scores['long_term_reward'].replace(10, 1)
sofa_scores['long_term_reward'] = sofa_scores['long_term_reward'].replace(-10, -1)
sofa_scores['reward'] = sofa_scores.groupby('csn')['sofa_sigma'].diff().fillna(0)
sofa_scores['reward'] = -sofa_scores['reward']/16
sofa_scores['long_term_fraction'] = sofa_scores['long_term_reward'] / sofa_scores.groupby('csn')['csn'].transform('count')
sofa_scores['cum_long_term'] = sofa_scores.groupby('csn')['long_term_fraction'].cumsum()
sofa_scores['reward'] = sofa_scores['reward']+sofa_scores['cum_long_term']

sofa_scores


# In[6]:


con_data = con_data.loc[:,~con_data.columns.duplicated()]


# In[7]:


# Function to determine the bin index for a value based on given ranges
def get_bin_index(value, ranges):
    for i, (low, high) in enumerate(ranges):
        if low < value <= high:
            return i
    return len(ranges) - 1  # If the value doesn't fall into any bin, return the last bin

# Add a new column for the result
con_data['bin_change_penalty'] = 0.0

# Define ranges for each setting
fio2_ranges = [(0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, float('inf'))]
peep_ranges = [(0, 7), (7, 9), (9, 11), (11, 13), (13, float('inf'))]
vent_rate_set_ranges = [(0, 15), (15, 18), (18, 21), (21, float('inf'))]
sweep_ranges = [(0, 2.5), (2.5, 5), (5, 7.5), (7.5, 10), (10, float('inf'))]
flow_ranges = [(0, 3), (3, 4), (4, 5), (5, 6), (6, float('inf'))]

# Iterate over each csn group
for csn, group in con_data.groupby('csn'):
    # group = group.sort_values('timestamp')
    previous_row = None
    
    for index, row in group.iterrows():
        bin_change = 0  # Initialize the bin change for this row
        
        if previous_row is not None:
            # Check the difference in bins for each action
            fio2_diff = abs(get_bin_index(row['fio2'], fio2_ranges) - get_bin_index(previous_row['fio2'], fio2_ranges))
            peep_diff = abs(get_bin_index(row['peep'], peep_ranges) - get_bin_index(previous_row['peep'], peep_ranges))
            vent_rate_set_diff = abs(get_bin_index(row['vent_rate_set'], vent_rate_set_ranges) - get_bin_index(previous_row['vent_rate_set'], vent_rate_set_ranges))
            sweep_diff = abs(get_bin_index(row['sweep'], sweep_ranges) - get_bin_index(previous_row['sweep'], sweep_ranges))
            flow_diff = abs(get_bin_index(row['flow'], flow_ranges) - get_bin_index(previous_row['flow'], flow_ranges))
            
            # Accumulate -0.5 for each setting that exceeds one bin
            if fio2_diff > 1:
                bin_change += -0.5
            if peep_diff > 1:
                bin_change += -0.5
            if vent_rate_set_diff > 1:
                bin_change += -0.5
            if sweep_diff > 1:
                bin_change += -0.5
            if flow_diff > 1:
                bin_change += -0.5

            # Apply additional penalties based on specific conditions
            if row['fio2'] < 0.21 or row['fio2'] > 0.8:
                bin_change += -0.5
            if row['peep'] < 5 or row['peep'] > 20:
                bin_change += -0.5
            if row['vent_rate_set'] > 20:
                bin_change += -0.5
            if row['sweep'] >= 10:
                bin_change += -1
            if row['flow'] < 0.5 or row['flow'] >= 6:
                bin_change += -1
                
            con_data.at[index, 'bin_change_penalty'] = bin_change
        
        previous_row = row


# In[8]:


con_data["reward"]=sofa_scores['reward']+con_data["bin_change_penalty"]
con_data.drop(columns="bin_change_penalty",inplace=True)


# In[9]:


train_csn = pd.read_csv("../Discrete Data/ECMO_data_clustered_train.csv").csn.unique()
test_csn = pd.read_csv("../Discrete Data/ECMO_data_clustered_test.csv").csn.unique()


# In[10]:


train_data = con_data[con_data['csn'].isin(train_csn)]
test_data = con_data[con_data['csn'].isin(test_csn)]

train_data.to_csv("train_data_continuous_Pos_for_Survival.csv",index=False)
test_data.to_csv("test_data_continuous_Pos_for_Survival.csv",index=False)


# In[ ]:




