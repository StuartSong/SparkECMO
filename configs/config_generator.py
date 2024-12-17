import yaml
import itertools
import os
from pathlib import Path

def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def generate_combinations(config):
    # Separate the keys that have multiple options (lists) and single options
    multi_value_keys = {k: v for k, v in config.items() if isinstance(v, list)}
    single_value_keys = {k: v for k, v in config.items() if k not in multi_value_keys}

    # Generate all combinations of the parameters that have multiple options
    combinations = list(itertools.product(*multi_value_keys.values()))

    # Initialize counter for "cql"
    cql_count = 0

    # Generate a list of new configs by combining single value keys with each combination of multi-value keys
    all_configs = []
    for combo in combinations:
        new_config = single_value_keys.copy()
        # Ensure the types are preserved in the new config
        combo_with_correct_types = {}
        for k, v in zip(multi_value_keys.keys(), combo):
            if isinstance(v, (int, float)):
                # Directly use the number if it's an int or float
                combo_with_correct_types[k] = v
            elif isinstance(v, str):
                # Keep the string as is
                combo_with_correct_types[k] = v
            elif isinstance(v, list):
                # Keep the list as is
                combo_with_correct_types[k] = v
            else:
                try:
                    # Try converting to float if possible
                    combo_with_correct_types[k] = float(v)
                except ValueError:
                    # If it fails, keep the original value
                    combo_with_correct_types[k] = v
        new_config.update(combo_with_correct_types)
        
        # Update algo_name with the current count
        if new_config.get("algo_name") == "cql":
            new_config["algo_name"] = f"cql-{cql_count}"
            cql_count += 1
        
        all_configs.append(new_config)
    
    return all_configs

def save_configs(configs, output_dir):
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for i, config in enumerate(configs):
        file_name = f"cql_config_{i}.yaml"
        file_path = os.path.join(output_dir, file_name)
        
        with open(file_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=None)
        print(f"Saved config: {file_path}")

def main():
    # Path to the initial config file
    input_config_path = "./cql_config.yaml"
    # Directory to save generated configs
    output_config_dir = "./generated"
    
    # Load initial config
    config = load_yaml_config(input_config_path)
    
    # Generate all combinations of the config
    all_configs = generate_combinations(config)
    
    # Save all generated configs to the output directory
    save_configs(all_configs, output_config_dir)

if __name__ == "__main__":
    main()
