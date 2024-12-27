import os
import pandas as pd

# Define the base directory
base_dir = "./d3rlpy_logs/fqe_./d3rlpy_logs/bcq_training_discrete_Pos./configs/bcq_disc_generated"

# Initialize a list to store results
results = []

# Traverse subdirectories
for subdir in os.listdir(base_dir):
    subdir_path = os.path.join(base_dir, subdir)
    for subsubdir in os.listdir(subdir_path):
        subsubdir_path = os.path.join(subdir_path, subsubdir)
        if os.path.isdir(subsubdir_path):
            # Construct the paths for init_value.csv and loss.csv
            init_value_path = os.path.join(subsubdir_path, "init_value.csv")
            loss_path = os.path.join(subsubdir_path, "loss.csv")
            
            # Check if the files exist
            if os.path.exists(init_value_path) and os.path.exists(loss_path):
                try:
                    # Read init_value.csv and loss.csv
                    init_value_df = pd.read_csv(init_value_path, header=None, names=["Index", "Init Value"])
                    loss_df = pd.read_csv(loss_path, header=None, names=["Index", "Loss"])
                    
                    # Merge the two dataframes
                    merged_df = pd.merge(init_value_df, loss_df, on="Index")
                    
                    # Filter rows where Loss < 0.20
                    filtered_df = merged_df[merged_df["Loss"] < 0.04].copy()
                    # Filter rows where Init Value < 0
                    # filtered_df = filtered_df[filtered_df["Init Value"] < -0.6].copy()
                    
                    # Check if any rows match the criteria
                    if not filtered_df.empty and (merged_df["Init Value"] < 1).all():
                        # Add the subdirectory path information
                        filtered_df.loc[:, "Path"] = subsubdir_path
                        
                        # Append the filtered rows to the results list
                        results.append(filtered_df)
                    else:
                        print(f"No rows in {subsubdir_path} meet the Loss < 0.20 criteria.")
                except Exception as e:
                    print(f"Error reading files in {subsubdir_path}: {e}")

# Combine results from all subdirectories
if results:
    combined_df = pd.concat(results, ignore_index=True)
    
    # Sort by Init Value in descending order and take the top 10
    top_10 = combined_df.sort_values(by="Init Value", ascending=False).head(10)
    
    # Display the results
    print("Top 10 Init Values with Loss < 0.20 and Corresponding Paths:")
    print(top_10)
    
    # Save the results to a CSV file
    top_10.to_csv("bcq_training_discrete_Pos_top_10_results.csv", index=False)
    print("Results saved to 'top_10_results.csv'.")
else:
    print("No results found with Loss < 0.20.")
