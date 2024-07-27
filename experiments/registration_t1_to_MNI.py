import os, sys
import numpy as np
from registration.registration import Registration
from registration.tools import RegTools
from rich.progress import Progress
from tools.progress_bar import ProgressBar
from tools.datautils import DataUtils
from os.path import split, join
from tools.filetools import FileTools
from tools.debug import Debug
from os.path import join, split, exists
from bids.mridata import MRIData
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '16'


# GROUP = "LPN-Project"
GROUP = "Mindfulness-Project"


dutils   = DataUtils()
ftools   = FileTools(GROUP)
debug    = Debug()
regtools = RegTools()
reg      = Registration()
pb       = ProgressBar()

BIDS_ROOT_PATH     = join(dutils.DATAPATH,GROUP)
PATH_2_MNI         = join(dutils.DATAPATH,"MNI","MNI152_T1_1mm_brain.nii.gz")
ANTS_TRANFORM_PATH = join(BIDS_ROOT_PATH,"derivatives","transforms","ants")



############ List all subjects ##################
recording_list = ftools.list_recordings()
to_register_list = list()
debug.warning("The following recordings require registration")
for recording_id in recording_list:
    subject_id,session = recording_id
    transform_dir_path        = join(ANTS_TRANFORM_PATH,f"sub-{subject_id}",f"ses-{session}","anat")
    filename = f"sub-{subject_id}_ses-{session}_desc-t1w_to_mni.syn.nii.gz"
    if not exists(join(transform_dir_path,filename)):
        debug.info("To register",subject_id,session)
        to_register_list.append([subject_id,session])

flag_continue = input("Confirm [Y,n]")
if flag_continue!="Y": sys.exit()
################################################

# sys.exit()
# recording_id   = to_register_list[0]
for ids, recording_id in enumerate(to_register_list):
    subject_id,session = recording_id
    mrsiData = MRIData(subject_id,session,group=GROUP)
    debug.separator()
    debug.title(f"Processing {subject_id}-{session} --- {ids}/{len(to_register_list)}")
    ############ Denoise PCr+Cr  ##################
    ############ T1w to MNI Registration ##################  
    transform_dir_path        = join(ANTS_TRANFORM_PATH,f"sub-{subject_id}",f"ses-{session}","anat")
    transform_prefix          = f"sub-{subject_id}_ses-{session}_desc-t1w_to_mni"
    transform_dir_prefix_path = join(transform_dir_path,f"{transform_prefix}")
    debug.warning(f"{transform_prefix} to T1w Registration not found or not up to date")
    syn_tx,_          = reg.register(fixed_input  = PATH_2_MNI,
                                    moving_input  = mrsiData.data["t1w"]["brain"]["orig"]["path"],
                                    fixed_mask    = None, 
                                    moving_mask   = None,
                                    transform     = "s",
                                    verbose       = 0)
    # Save Transform
    os.makedirs(transform_dir_path,exist_ok=True)
    regtools.save_all_transforms(syn_tx,transform_dir_prefix_path)

    #################################
debug.title("DONE")
debug.separator()











    






