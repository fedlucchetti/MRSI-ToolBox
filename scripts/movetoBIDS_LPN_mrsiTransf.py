import os
import shutil
import sys
from tools.debug import Debug
import numpy as np
from tools.datautils import DataUtils

from os.path import join,split

debug  = Debug()
dutils = DataUtils()


# Configuration
source_dir = '/media/flucchetti/NSA1TB1/Connectome/Data/ARMS'
bids_dir   = '/media/veracrypt2/Connectome/Data/LPN-Project/derivatives/transforms/ants'
# retain_lipid_file = [
#     ['S001_V1', '8'],
#     # Add other mappings here...
# ]

# Mapping for session identifiers
session_map = {
    'V1': 'ses-V1',
    'V2': 'ses-V2',
    'V3': 'ses-V3',
    'V4': 'ses-V4', 
    'V5': 'ses-V5',
}

# Metabolite file descriptors
metabolites = [ 
    'CrPCr', 'GluGln', 'GPCPCh', 'Ins', 'NAA+NAAG'
]




subjectid_exceptions = ["A016","A028"]

# Helper function to create target directory if it doesn't exist
def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

# Function to move files according to BIDS standard
def process_subject(subject_dir_path):
    # debug.info("subject_dir_path",subject_dir_path)
    subj_id, session_id = split(subject_dir_path)[1].split('_V')
    session_id=f"V{session_id}"
    # debug.info("subj_id",subj_id,"session_id",session_id)
    
    if subj_id in subjectid_exceptions:
        subj_id = f"NA{subj_id}"
    bids_subj_id          = f"sub-CHUV{subj_id[0:].zfill(2)}"
    bids_session_id       = session_map[session_id]
    subject_bids_dir_path = join(bids_dir, bids_subj_id, bids_session_id)

    dest_bids_subdir = join(subject_bids_dir_path, 'spectroscopy')
    
    debug.info("Processing",subject_dir_path)
    for space in os.listdir(subject_dir_path):
        if "MRSI2T1" != space:
            continue 
        source_space_dir = join(subject_dir_path, space)
        
        for file in os.listdir(source_space_dir):
            if "GenericAffine.mat" in file:
                new_file_name = "mrsi_Cr_to_t1.affine.mat"
            elif "GenericAffineInv.mat" in file:
                new_file_name = "mrsi_Cr_to_t1.affine_inv.mat"
            elif "Warp.nii" in file:
                new_file_name = "mrsi_Cr_to_t1.syn.nii.gz"
            elif "WarpInv.nii" in file:
                new_file_name = "mrsi_Cr_to_t1.syn_inv.nii.gz"
            else:
                continue


            # Construct the new BIDS filename and path
            os.makedirs(dest_bids_subdir,exist_ok=True)
            old_file_name = f"{bids_subj_id}_{bids_session_id}_{new_file_name}"
            old_file_path = os.path.join(dest_bids_subdir, old_file_name)
            
            # Ensure the target directory exists
            ensure_dir(os.path.dirname(old_file_path))
            
            # Move the file
            debug.info(f"Removed {old_file_name} \n")
            if os.path.exists(old_file_path):
                os.remove(old_file_path)
            new_file_name = f"{bids_subj_id}_{bids_session_id}_desc-{new_file_name}"
            new_file_path = os.path.join(dest_bids_subdir, new_file_name)
            shutil.copy(os.path.join(source_space_dir, file), new_file_path)
            debug.info(f"Moved {space}/{file} to {new_file_name} \n")
    # sys.exit()

# Main processing loop
for subject_dir in os.listdir(source_dir):
    subject_dir_path = join(source_dir,subject_dir)
    process_subject(subject_dir_path)
    # sys.exit()
