import os
import shutil
import sys
from tools.debug import Debug
import numpy as np
from tools.datautils import DataUtils

debug  = Debug()
dutils = DataUtils()
LIPID_PREFERENCE = "08"
retain_list_arr = np.load(os.path.join(dutils.ANARESULTSPATH,"Qcheck","retain_list.npz"))
subject_id_arr  = retain_list_arr["subject_id"]
lipid_arr           = retain_list_arr["lipid"]

retain_lipid_file = list()
for ids,subject_id in enumerate(subject_id_arr):
    retain_lipid_file.append([subject_id,str(int(lipid_arr[ids]))])

# Configuration
# source_dir   = '/media/veracrypt2/Connectome/Data/Mindful_missing'
source_dir   = '/media/flucchetti/NSA1TB2/Connectome/Data/MindfullTeen/OrigRes'

bids_dir   = '/media/veracrypt2/Connectome/Data/Mindfulness-Project'

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

# Metabolite filename descriptors
metabolites = [
    'CrPCr', 'GluGln', 'GPCPCh', 'Ins', 'NAA+NAAG'
]


# Space mapping
space_mapping = {
    'Conc': 'space-t1w',
    'OrigRes': 'space-orig',
    'OrigResFilt': 'space-origfilt'
}

# Helper function to create target directory if it doesn't exist
def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

# Function to move files according to BIDS standard
def process_subject(subj_id, session_id, lipid_preference="08"):
    # debug.info("session_folder",session_folder)
    # debug.info("subj_id",subj_id,"session_id",session_id)
    bids_subj_id = f"sub-{subj_id.zfill(2)}"
    bids_session_id = session_map[session]

    # Directory where the LipRem_X folder is located
    # liprem_dir = os.path.join(source_dir, session_folder, f'LipRem_{lipid_preference}')
    liprem_dir = os.path.join(source_dir,f"{subj_id}_{session}",f"Lipid{lipid_preference}",f"Results_{subj_id}_{session_id}","MRSI_Nifti")
    # debug.info(liprem_dir,os.path.exists(liprem_dir))
    # debug.info(os.listdir(liprem_dir))
    # debug.info(session_folder)
    dest_subdir = os.path.join(bids_dir, bids_subj_id, bids_session_id, 'spectroscopy')
    if os.path.exists(dest_subdir):
        if len(os.listdir(dest_subdir))!=0:
            # debug.info( bids_subj_id, bids_session_id,"already exists")
            return
    debug.info("Processing", bids_subj_id, bids_session_id)
    for filename in os.listdir(liprem_dir):
        
        # source_space_dir = os.path.join(liprem_dir,filename)
        if ".pdf" in filename or ".txt" in filename:continue
        new_file_name = filename.replace("+", "")
        if "CRLB" in filename:
            bids_acq = "acq-crlb"
            new_file_name = new_file_name.replace("CRLB", "")
        elif "conc" in filename:
            bids_acq = "acq-conc"
            new_file_name = new_file_name.replace("conc", "")
        elif "Voxel_SNR" in filename:
            bids_acq = "acq-snr"
            new_file_name = new_file_name.replace("Voxel_SNR", "Voxel")
        elif "Voxel_FreqShift" in filename:
            bids_acq = "acq-freqshift"
            new_file_name = new_file_name.replace("Voxel_FreqShift", "Voxel")
        elif "Voxel_index" in filename:
            bids_acq = "acq-index"
            new_file_name = new_file_name.replace("Voxel_index", "Voxel")
        elif "Voxel_Ph" in filename:
            bids_acq = "acq-ph"
            new_file_name = new_file_name.replace("Voxel_Ph", "Voxel")
        elif "Voxel_PhOne" in filename:
            bids_acq = "acq-phone"
            new_file_name = new_file_name.replace("Voxel_PhOne", "Voxel")
        elif "Voxel_FWHM" in filename:
            bids_acq = "acq-fwhm"
            new_file_name = new_file_name.replace("Voxel_FWHM", "Voxel")
        elif "WaterSignal" in filename:
            bids_acq = "acq-conc"
            new_file_name = new_file_name.replace("WaterSignal", "WaterSignal")
        elif "slice_mask" in filename:
            bids_acq = "acq-conc"
            new_file_name = new_file_name.replace("slice_mask", "slice_mask")
        elif "brain_mask" in filename:
            bids_acq = "acq-conc"
            new_file_name = new_file_name.replace("brain_mask", "brain_mask")
        elif "MetabNorm" in filename:
            bids_acq = "acq-metabnorm"
            new_file_name = new_file_name.replace("MetabNorm", "")
        elif "LCMBaseline" in filename:
            bids_acq = "acq-meta"
        elif "LCMFit" in filename:
            bids_acq = "acq-meta"
        # elif "OrigRes" in filename:
        bids_space = "space-orig"
        
        new_file_name = new_file_name.replace("OrigResFilt", "")
        new_file_name = new_file_name.replace("OrigRes", "")
        new_file_name = new_file_name.replace("_", "")

        # Construct the new BIDS filename and path

        new_file_name = new_file_name.replace('.nii.gz', "_spectroscopy.nii.gz")
        new_file_name = f"{bids_subj_id}_{bids_session_id}_{bids_space}_{bids_acq}_desc-{new_file_name}"
        new_file_path = os.path.join(dest_subdir, new_file_name)
        
        # Ensure the target directory exists
        ensure_dir(os.path.dirname(new_file_path))
        
        # Move the filename
        os.makedirs(dest_subdir,exist_ok=True)
        shutil.copy(os.path.join(liprem_dir, filename), new_file_path)
        debug.info(f"Copied {filename} to {new_file_name} \n")
    # sys.exit()
    debug.separator()

# Main processing loop
folder_list = os.listdir(source_dir)
for folder_name in folder_list:
    folder_name = folder_name.replace("_8","_08")
    if f"_{LIPID_PREFERENCE}" in folder_name:
        subject_id = folder_name[0:4]
        session    = folder_name[5::].replace(f"_LipRem_{LIPID_PREFERENCE}","")
        # debug.info("Processing",subject_id,session)
        process_subject(subject_id, session, lipid_preference=LIPID_PREFERENCE)


    
