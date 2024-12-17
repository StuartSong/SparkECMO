import re
import os
import glob

def find_best_model(directory="./"):
    # Regex to extract init_value from the line
    pattern = r"Best model is: .* with init_value: (-?\d+\.\d+)"
    max_init_value = float("-inf")
    best_file = None
    best_line = ""

    # Define the file pattern to match files like slurm-22801554_199.out
    search_pattern = os.path.join(directory, "slurm-22801554_*.out")
    
    # Debug: Print search pattern
    print(f"Searching for files matching: {search_pattern}")
    
    # Find all matching files
    files = glob.glob(search_pattern)
    print(f"Found {len(files)} files.")

    # Iterate through all matching files
    for file in files:
        try:
            with open(file, 'r') as f:
                for line in f:
                    match = re.search(pattern, line)
                    if match:
                        init_value = float(match.group(1))
                        if init_value > max_init_value:
                            max_init_value = init_value
                            best_file = file
                            best_line = line.strip()
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if best_file:
        print(f"File with max init_value: {best_file}")
        print(f"Line: {best_line}")
        print(f"Max init_value: {max_init_value}")
    else:
        print("No matching lines found in the given files.")

# Run the function
if __name__ == "__main__":
    find_best_model()
