import os
import re, sys


DIR = '/home/flucchetti/Connectome/Dev/Analytics/results/simmatrix/MindfullTeen'  # Replace 'DIR' with the path to your directory

# Compile a regex pattern for matching folder names ending with '_LipRem08'
pattern = re.compile(r'(.*)_LipRem08$')

# Walk through the directory
for root, dirs, files in os.walk(DIR, topdown=False):
    for name in dirs:
        # Check if the directory name matches the pattern
        match = pattern.match(name)
        if match:
            # Construct the new directory name by replacing '_LipRem08' with '_LipRem8'
            new_name = match.group(1) + '_LipRem8'
            # Full path for the current and new directory name
            old_dir_path = os.path.join(root, name)
            new_dir_path = os.path.join(root, new_name)
            
            # Rename the directory
            os.rename(old_dir_path, new_dir_path)
            print(f"Renamed '{old_dir_path}' to '{new_dir_path}'")
            # sys.exit()
print("Renaming process completed.")