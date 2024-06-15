import os
import shutil, re

# Define the source and destination directories
SOURCEDIR = '/media/veracrypt1/Mindfulness-Project'
DESDIR = '/media/flucchetti/MRSIBackup/MF_T1'
STRING = 'your_specific_string'

def copy_matching_files(source_dir, dest_dir):
    # Ensure the destination directory exists
    os.makedirs(dest_dir, exist_ok=True)
    
    # Define the substring to match files
    substring = "_acq-memprage_T1w.nii.gz"
    
    # Loop through the files in the source directory and its subdirectories
    for dirpath, _, filenames in os.walk(source_dir):
        for filename in filenames:
            if substring in filename:
                # Construct full file paths
                src_path = os.path.join(dirpath, filename)
                dest_path = os.path.join(dest_dir, filename)
                
                # Copy file
                shutil.copy2(src_path, dest_path)
                print(f"Copied: {src_path} to '{dest_path}'")

# Run the function
copy_matching_files(SOURCEDIR, DESDIR)

