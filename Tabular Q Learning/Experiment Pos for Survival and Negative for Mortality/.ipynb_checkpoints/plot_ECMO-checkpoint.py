import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.backends.backend_pdf import PdfPages
import glob

ranges_dict = {
        'fio2': [(0.21, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 1)], # min 21%, max 100%
        'peep': [(5, 7), (7, 9), (9, 11), (11, 13), (13, 20)],
        'vent_rate_set': [(0, 15), (15, 18), (18, 21), (21, 40)],
        'sweep': [(0.25, 2.5), (2.5, 5), (5, 7.5), (7.5, 10), (10, 20)],
        'flow': [(0.5, 3), (3, 4), (4, 5), (5, 6), (6, 10)]
        }
# Define a function to create grouped bar plots for each action type

def plot_action_ranges(agent_action, clinician_action, action, ranges, pdf):
    # Calculate the total occurrences for each range
    range_columns = [f"{action}_range_{i}" for i in range(len(ranges))]
    occurrences_agent = agent_action[range_columns].sum().values
    occurrences_clinician = clinician_action[range_columns].sum().values
    
    # Create a grouped bar plot
    bar_width = 0.35
    index = range(len(ranges))
    
    plt.figure(figsize=(12, 6))
    plt.bar(index, occurrences_agent, bar_width, label='Agent')
    plt.bar([i + bar_width for i in index], occurrences_clinician, bar_width, label='Clinician')
    
    plt.xlabel(f'{action} Ranges')
    plt.ylabel('Occurrences')
    plt.title(f'Occurrences in Each {action} Range')
    plt.xticks([i + bar_width / 2 for i in index], [f"{r[0]} to {r[1]}" for r in ranges])
    plt.legend()
    pdf.savefig()  # Save the current figure into the PDF
    plt.close()  # Close the current figure to avoid display

def plot_comparison(model, cluster_num, all_action_combination):
    # DataFrame to store the sums of each model
    model_sums = pd.DataFrame()
    model['agent_action'] = model['agent_action'].astype(int)
    all_action_combination['action_number'] = all_action_combination['action_number'].astype(int)
    # Assuming all_action_combination is previously defined and loaded
    # Merge the model DataFrame with an all_action_combination DataFrame
    merged_df = model[['agent_action','csn']].merge(
        all_action_combination, 
        how='left', 
        left_on='agent_action', 
        right_on='action_number'
    ).drop(columns='agent_action')
    
    # Assuming merged_agent_action and merged_clinician_action are the merged DataFrames with action data
    # Create a PdfPages object to save multiple plots in one PDF file
    pdf_path = f'Optimal Model action Comparison.pdf'
    merged_agent_action = model.merge(all_action_combination, left_on="agent_action", right_on="action_number")
    merged_clinician_action = model.merge(all_action_combination, left_on="action", right_on="action_number")
    
    with PdfPages(pdf_path) as pdf:
        for action, ranges in ranges_dict.items():
            plot_action_ranges(merged_agent_action, merged_clinician_action, action, ranges, pdf)


def map_values(df, value_array):
    # Verify that the length of the value_array matches the number of columns in the dataframe
    if len(value_array) != df.shape[1]:
        raise ValueError("The length of value_array must match the number of columns in the DataFrame.")
    
    # Get column names as a list to map string index to integer position
    columns = list(df.columns)
    
    # Find the column index of the '1' in each row as a string and convert to integer
    idx = df.idxmax(axis=1).map(lambda x: columns.index(x))
    
    # Map these indices to the corresponding values in value_array
    return [value_array[i] for i in idx]


def plot_traj(model, cluster_num, ori_AS, all_action_combination):
    # DataFrame to store the sums of each model
    model_sums = pd.DataFrame()
    
    # Assuming all_action_combination is previously defined and loaded
    # Merge the model DataFrame with an all_action_combination DataFrame
    merged_df = model[['agent_action','csn']].merge(
        all_action_combination, 
        how='left', 
        left_on='agent_action', 
        right_on='action_number'
    ).drop(columns='agent_action')

    ranges_dict = {
        'fio2': [0.3, 0.45, 0.55, 0.7],
        'peep': [6, 8, 10, 12, 14],
        'vent': [13, 16.5, 19.5, 23],
        'sweep': [1.5, 4, 6.5, 9, 11],
        'flow': [2, 3.5, 4.5, 5.5, 7]}
    
    # Convert columns to int64
    merged_df.csn = merged_df.csn.astype('int64')
    ori_AS.csn = ori_AS.csn.astype('int64')
    
    merge_cmn = ["vent_rate_set", 'peep', 'fio2', 'ECMO Pump Flow', 'ECMO Sweep Gas Flow']
    ori_cmn = ["vent", 'peep', 'fio2', 'flow', 'sweep']
    
    # Create a PDF file to save the figures
    with PdfPages(f'{cluster_num} Clusters Trajectories Comparison.pdf') as pdf:
        for csn in ori_AS.csn.unique():
            csn_data = ori_AS[ori_AS.csn == csn]
            csn_merge = merged_df[merged_df.csn == int(csn)]
            if len(csn_merge) == 0:
                continue
            if len(csn_data) < 2:
                continue
            
            fig, axs = plt.subplots(5, figsize=(8, 12))
            fig.suptitle(f"CSN: {csn}", fontsize=16)  # Add a super title with the current csn
    
            for i, key in enumerate(merge_cmn):
                columns = [name for name in csn_merge.columns if ori_cmn[i] in name]
                RL_data = map_values(csn_merge[columns], ranges_dict[ori_cmn[i]])
                axs[i].plot(np.array(csn_data[merge_cmn[i]]), marker='o', markersize=3, label="clinician practices")
                axs[i].plot(RL_data, marker='o', markersize=3, label="RL suggested")
                axs[i].legend(frameon=False,fontsize='small')
                axs[i].set_title(key)
    
            # Adjust the space between subplots and provide space for the super title
            plt.subplots_adjust(hspace=0.5, top=0.92)  # Adjust 'top' to provide space for the super title
    
            # Save the current figure to the PDF
            pdf.savefig(fig)
            plt.close(fig)  # Close the figure after saving it to the PDF


def plot_average_occurrences(all_action_combination, cluster_num):
    models = []
    csv_files = glob.glob(os.path.join(f'cluster_{cluster_num}', '*.csv'))
    num_models = len(csv_files)
    
    for i in range(num_models):
        model_path = f'cluster_{cluster_num}/test_data_{i}.csv'
        model = pd.read_csv(model_path)
        models.append(model)
    
    # DataFrame to store the occurrences for each model
    all_occurrences = []

    for model in models:
        merged_agent_action = model[['agent_action', 'csn']].merge(
            all_action_combination, 
            how='left', 
            left_on='agent_action', 
            right_on='action_number'
        ).drop(columns='agent_action')

        merged_clinician_action = model[['action', 'csn']].merge(
            all_action_combination, 
            how='left', 
            left_on='action', 
            right_on='action_number'
        ).drop(columns='action')
        
        model_occurrences = {}
        for action, ranges in ranges_dict.items():
            range_columns = [f"{action}_range_{i}" for i in range(len(ranges))]
            occurrences_agent = merged_agent_action[range_columns].sum().values
            occurrences_clinician = merged_clinician_action[range_columns].sum().values
            
            model_occurrences[action] = {
                'agent': occurrences_agent,
                'clinician': occurrences_clinician
            }
        all_occurrences.append(model_occurrences)
    
    # Calculate the mean and standard deviation
    mean_occurrences = {}
    std_occurrences = {}
    
    for action in ranges_dict.keys():
        agent_data = np.array([model[action]['agent'] for model in all_occurrences])
        clinician_data = np.array([model[action]['clinician'] for model in all_occurrences])
        
        mean_occurrences[action] = {
            'agent': agent_data.mean(axis=0),
            'clinician': clinician_data.mean(axis=0)
        }
        std_occurrences[action] = {
            'agent': agent_data.std(axis=0),
            'clinician': clinician_data.std(axis=0)
        }

    # Create a PdfPages object to save multiple plots in one PDF file
    pdf_path = f'{cluster_num} Clusters ECMO action Comparison.pdf'
    
    with PdfPages(pdf_path) as pdf:
        for action, ranges in ranges_dict.items():
            range_labels = [f"{r[0]} to {r[1]}" for r in ranges]

            if action == 'fio2':
                range_labels = [f"{int(r[0]*100)} to {int(r[1]*100)}" for r in ranges]
                xlabel = f'FiO$_2$ (%)'
            elif action == 'peep':
                xlabel = f'PEEP (cm H$_2$O)'
            elif action == 'vent_rate_set':
                xlabel = f'Vent Rate (bpm)'
            elif action == 'sweep':
                range_labels = [f"{int(r[0]*1000)} to {int(r[1]*1000)}" for r in ranges]
                xlabel = f'Sweep Gas Flow (mL/min)'
            elif action == 'sweep':
                range_labels = [f"{int(r[0]*1000)} to {int(r[1]*1000)}" for r in ranges]
                xlabel = f'Blood Flow (mL/min)'
            
            mean_agent = mean_occurrences[action]['agent']
            mean_clinician = mean_occurrences[action]['clinician']
            
            std_agent = std_occurrences[action]['agent']
            std_clinician = std_occurrences[action]['clinician']
            
            bar_width = 0.35
            index = np.arange(len(ranges))
            
            plt.figure(figsize=(12, 6))
            plt.bar(index, mean_agent, bar_width, yerr=std_agent, label='Agent', capsize=5)
            plt.bar(index + bar_width, mean_clinician, bar_width, yerr=std_clinician, label='Clinician', capsize=5)
            
            plt.xlabel(xlabel)
            plt.ylabel('Average Occurrences')
            plt.title(f'Average Occurrences in Each {action} Range')
            plt.xticks(index + bar_width / 2, range_labels)
            plt.legend()
            pdf.savefig()  # Save the current figure into the PDF
            plt.close()  # Close the current figure to avoid display

def plot_performance_comparison(lower_upper_bounds_avg, medians_avg, maxs_avg, avg_reward, type="normalized"):
    lower_bounds_avg = [lb for lb, ub in lower_upper_bounds_avg]
    upper_bounds_avg = [ub for lb, ub in lower_upper_bounds_avg]

    min_value = np.min(lower_bounds_avg)
    max_value = (np.max(maxs_avg) - min_value) / 100

    length = len(medians_avg) + 1
    
    plt.figure(figsize=(10, 6))
    if type == "normalized":
        
        plt.plot(range(1, length), (np.array(medians_avg) - min_value) / max_value, label='Agent Median', color='blue')
        plt.plot(range(1, length), (np.array(maxs_avg) - min_value) / max_value, label='Agent Max', color='red')
        plt.fill_between(range(1, length), (np.array(lower_bounds_avg) - min_value) / max_value, (np.array(upper_bounds_avg) - min_value) / max_value, color='lightblue', alpha=0.5, label='95% Confidence Interval')
        plt.plot(range(1, length), [(np.mean(avg_reward) - min_value) / max_value] * (length - 1), label='Clinician Reference', color='orange', linestyle='dashed')
        plt.title('Agent and Clinician Rewards Comparison (Normalized)')
    else:
        plt.plot(range(1, length), medians_avg, label='Agent Median', color='blue')
        plt.plot(range(1, length), maxs_avg, label='Agent Max', color='red')
        plt.fill_between(range(1, length), lower_bounds_avg, upper_bounds_avg, color='lightblue', alpha=0.5, label='95% Confidence Interval')
        plt.plot(range(1, length), [np.mean(avg_reward)]*(length-1), label='Clinician Reference', color='orange', linestyle='dashed')
        plt.title('Agent and Clinician Rewards Comparison')

    plt.xlabel('Number of Models')
    plt.ylabel('Estimated Performance Return')
    
    plt.legend(loc="lower right")
    plt.savefig(f'Reward Comparison on testing data {type}.png', bbox_inches='tight')
    plt.show()