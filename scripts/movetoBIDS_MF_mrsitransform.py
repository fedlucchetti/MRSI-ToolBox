import os
import shutil
import sys
from tools.debug import Debug
import numpy as np
from tools.datautils import DataUtils

debug  = Debug()
dutils = DataUtils()

retain_list_arr = np.load(os.path.join(dutils.ANARESULTSPATH,"Qcheck","retain_list.npz"))
subject_id_arr  = retain_list_arr["subject_id"]
lipid_arr           = retain_list_arr["lipid"]

retain_lipid_file = list()
for ids,subjec_id in enumerate(subject_id_arr):
    retain_lipid_file.append([subjec_id,str(int(lipid_arr[ids]))])

# Configuration
source_dir = '/media/flucchetti/NSA1TB1/Connectome/Data/MindfullTeen/Reg'
bids_dir   = '/media/veracrypt2/Connectome/Data/Mindfulness-Project/derivatives/transforms/ants'

# retain_lipid_file = [
#     ['S001_V1', '8'],
#     # Add other mappings here...
# ]

# Mapping for session identifiers
session_map = {
    'V1': 'ses-V1',
    'V2': 'ses-V2',
    'V2_BIS': 'ses-V3'
}

# Metabolite file descriptors
metabolites = [
    'CrPCr', 'GluGln', 'GPCPCh', 'Ins', 'NAA+NAAG'
]


# Helper function to create target directory if it doesn't exist
def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

# Function to move files according to BIDS standard
def process_subject(session_folder, lipid_preference):
    # debug.info("session_folder",session_folder)
    subj_id, session_id = session_folder.split('_V')
    session_id=f"V{session_id}"
    # debug.info("subj_id",subj_id,"session_id",session_id)
    bids_subj_id = f"sub-{subj_id[0:].zfill(2)}"
    bids_session_id = session_map[session_id]

    # Directory where the LipRem_X folder is located
    liprem_dir = os.path.join(source_dir, session_folder, f'LipRem_{lipid_preference}')
    debug.info(liprem_dir,os.path.exists(liprem_dir))
    # debug.info(session_folder)

    dest_subdir = os.path.join(bids_dir, bids_subj_id, bids_session_id, 'spectroscopy')
    for space in os.listdir(liprem_dir):
        if "Transform_MRSI2T1" != space:
            continue 
        
        source_space_dir = os.path.join(liprem_dir, space)
        
        for file in os.listdir(source_space_dir):
            if "GenericAffine.mat" in file:
                new_file_name = f"desc-mrsi_Cr_liprem_{lipid_preference}_to_t1.affine.mat"
            elif "GenericAffineInv.mat" in file:
                new_file_name = f"desc-mrsi_Cr_liprem_{lipid_preference}_to_t1.affine_inv.mat"
            elif "Warp.nii" in file:
                new_file_name = f"desc-mrsi_Cr_liprem_{lipid_preference}_to_t1.syn.nii.gz"
            elif "WarpInv.nii" in file:
                new_file_name = f"desc-mrsi_Cr_liprem_{lipid_preference}_to_t1.syn_inv.nii.gz"
            else:
                continue

            # Construct the new BIDS filename and path
            os.makedirs(dest_subdir,exist_ok=True)
            new_file_name = f"{bids_subj_id}_{bids_session_id}_{new_file_name}"
            new_file_path = os.path.join(dest_subdir, new_file_name)
            
            # Ensure the target directory exists
            ensure_dir(os.path.dirname(dest_subdir))
            
            # Move the file
            shutil.copy(os.path.join(source_space_dir, file), new_file_path)
            debug.info(f"Moved {source_space_dir}/{file} to {new_file_name} \n")
    # sys.exit()

# Main processing loop
for session_folder, lipid_preference in retain_lipid_file:
    process_subject(session_folder, lipid_preference)
