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
from os.path import join, split
from bids.mridata import MRIData
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '32'


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
recording_id   = recording_list[0]
for ids, recording_id in enumerate(recording_list):
    subject_id,session = recording_id
    mrsiData = MRIData(subject_id,session,group=GROUP)
    debug.separator()
    debug.title(f"Processing {subject_id}-{session} --- {ids}/{len(recording_list)}")
    ############ Denoise PCr+Cr  ##################
    ############ T1w to MNI Registration ##################  
    transform_dir_path        = join(ANTS_TRANFORM_PATH,f"sub-{subject_id}",f"ses-{session}","anat")
    transform_prefix          = f"sub-{subject_id}_ses-{session}_desc-t1w_to_mni"
    transform_dir_prefix_path = join(transform_dir_path,f"{transform_prefix}")
    if not os.path.exists(f"{transform_dir_prefix_path}.syn.nii.gz") :
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
    else:
        debug.success("Already registered")
    #################################
debug.title("DONE")
debug.separator()











    






