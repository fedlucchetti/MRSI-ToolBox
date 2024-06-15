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
bids_dir   = '/media/veracrypt2/Connectome/Data/LPN-Project'
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


# Space mapping
space_mapping = {
    # 'Conc': 'space-t1w',
    'OrigRes': 'space-orig',
    'OrigResFilt': 'space-origfilt'
}

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
    
    bids_subj_id    = f"sub-CHUV{subj_id[0:].zfill(2)}"
    bids_session_id = session_map[session_id]
    subject_bids_dir_path = join(bids_dir, bids_subj_id, bids_session_id)

    if not os.path.exists(subject_bids_dir_path):
        bids_subj_id          = f"sub-CHUVNA{subj_id[0:].zfill(2)}"
        subject_bids_dir_path = join(bids_dir, bids_subj_id, bids_session_id)
        if not os.path.exists(subject_bids_dir_path):
            debug.error(bids_subj_id,"not found")
            return 
        
    

    dest_bids_subdir = join(subject_bids_dir_path, 'spectroscopy')
    
    for space in os.listdir(subject_dir_path):
        if space not in space_mapping:
            continue  # Skip unexpected directories
        
        bids_space       = space_mapping[space]
        source_space_dir = join(subject_dir_path, space)
        
        for file in os.listdir(source_space_dir):

            if ".pdf" in file or ".txt" in file or "._"==file[0:2] or "_thr" in file:
                continue
            new_file_name = file.replace("+", "")
            if "CRLB" in file:
                bids_acq = "acq-crlb"
                new_file_name = new_file_name.replace("CRLB", "")
            elif "conc" in file:
                bids_acq = "acq-conc"
                new_file_name = new_file_name.replace("conc", "")
            elif "Conc" in file:
                bids_acq = "acq-conc"
                new_file_name = new_file_name.replace("Conc", "")
            elif "Voxel_SNR" in file:
                bids_acq = "acq-snr"
                new_file_name = new_file_name.replace("Voxel_SNR", "Voxel")
            elif "Voxel_FreqShift" in file:
                bids_acq = "acq-freqshift"
                new_file_name = new_file_name.replace("Voxel_FreqShift", "Voxel")
            elif "Voxel_index" in file:
                bids_acq = "acq-index"
                new_file_name = new_file_name.replace("Voxel_index", "Voxel")
            elif "Voxel_Ph" in file:
                bids_acq = "acq-ph"
                new_file_name = new_file_name.replace("Voxel_Ph", "Voxel")
            elif "Voxel_PhOne" in file:
                bids_acq = "acq-phone"
                new_file_name = new_file_name.replace("Voxel_PhOne", "Voxel")
            elif "Voxel_FWHM" in file:
                bids_acq = "acq-fwhm"
                new_file_name = new_file_name.replace("Voxel_FWHM", "Voxel")
            elif "WaterSignal" in file:
                bids_acq = "acq-conc"
                new_file_name = new_file_name.replace("WaterSignal", "WaterSignal")
            elif "slice_mask" in file:
                bids_acq = "acq-conc"
                new_file_name = new_file_name.replace("slice_mask", "slice_mask")
            elif "brain_mask" in file:
                bids_acq = "acq-conc"
                new_file_name = new_file_name.replace("brain_mask", "brain_mask")
            elif "MetabNorm" in file:
                bids_acq = "acq-metabnorm"
                new_file_name = new_file_name.replace("MetabNorm", "")
            elif "LCMBaseline" in file:
                bids_acq = "acq-meta"
            elif "LCMFit" in file:
                bids_acq = "acq-meta"
            
            new_file_name = new_file_name.replace("OrigResFilt", "")
            new_file_name = new_file_name.replace("OrigRes", "")
            new_file_name = new_file_name.replace("_", "")

            # Construct the new BIDS filename and path
            os.makedirs(dest_bids_subdir,exist_ok=True)
            new_file_name = new_file_name.replace('.nii.gz', "_spectroscopy.nii.gz")
            new_file_name = f"{bids_subj_id}_{bids_session_id}_{bids_space}_{bids_acq}_desc-{new_file_name}"
            new_file_path = os.path.join(dest_bids_subdir, new_file_name)
            
            # Ensure the target directory exists
            ensure_dir(os.path.dirname(new_file_path))
            
            # Move the file
            shutil.copy(os.path.join(source_space_dir, file), new_file_path)
            debug.info(f"Copied {space}/{file} to {new_file_name} \n")
    # sys.exit()

# Main processing loop
for subject_dir in os.listdir(source_dir):
    subject_dir_path = join(source_dir,subject_dir)
    process_subject(subject_dir_path)
