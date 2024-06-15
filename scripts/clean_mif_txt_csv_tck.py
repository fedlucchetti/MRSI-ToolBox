import os
import shutil

def delete_files_in_dir_except_subdir(dest_dir):
    # Define the file extensions to be deleted
    extensions_to_delete = ['.mif', '.txt', '.csv', '.tck','5tt_nocoreg.nii.gz','mean_b0.nii.gz']
    # Define the subdirectory to exclude
    exclude_subdir = os.path.join(dest_dir, 'derivatives')
    
    # Walk through the directory tree
    for root, dirs, files in os.walk(dest_dir):
        # Skip the excluded subdirectory
        if root.startswith(exclude_subdir):
            continue
        
        # Iterate over the files in the current directory
        for file in files:
            # Check if the file has one of the target extensions
            if any(file.endswith(ext) for ext in extensions_to_delete):
                # Construct the full file path
                file_path = os.path.join(root, file)
                try:
                    # Delete the file
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")

# Example usage:
DESTDIR = "/media/veracrypt2/Connectome/Data/Mindfulness-Project/"
delete_files_in_dir_except_subdir(DESTDIR)
