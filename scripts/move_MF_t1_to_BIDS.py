import os, sys
import numpy as np
from tools.datautils import DataUtils
from os.path import join, split
from tools.debug import Debug
import shutil


dutils   = DataUtils()
debug    = Debug()


BIDS_ROOT_PATH     = join(dutils.DATAPATH,"Mindfulness-Project")
NEW_BIDS_DIR_PATH  = join(BIDS_ROOT_PATH)
OG_ROOT_DIR_PATH   = "/media/flucchetti/77FF-B8072/Mindfulness-Project"

def move_t1brain():
    subject_list = os.listdir(OG_ROOT_DIR_PATH)
    subject_list.sort()
    for subject_id in subject_list:
        if "sub-" not in subject_id:continue
        session_list = os.listdir(join(OG_ROOT_DIR_PATH,subject_id))
        for session in session_list:
            if "ses-" in session:
                acq_list = os.listdir(join(OG_ROOT_DIR_PATH,subject_id,session))
                if "anat" in acq_list:
                    anat_dir_path = join(OG_ROOT_DIR_PATH,subject_id,session,"anat")
                    t1_filenames = os.listdir(anat_dir_path)
                    if "T1Brain_noSkull.nii.gz" in t1_filenames:
                        for filename in t1_filenames:
                            if "T1Brain_noSkull.nii.gz" in filename:
                                new_filename  = f"{subject_id}_{session}_run-01_acq-memprage_T1w_brain.nii.gz"
                                dest_filepath = join(NEW_BIDS_DIR_PATH,subject_id,session,"anat",new_filename)
                                og_filepath   = join(anat_dir_path,filename)
                                debug.info("Copying",subject_id,session,"anat",filename)
                                debug.info("to     ",dest_filepath)
                                # os.remove(og_filepath)
                                shutil.copy(og_filepath, dest_filepath)
                    else:
                        debug.warning("No t1 brain found in",subject_id,session)
                else:
                    debug.warning("No anmat found in",subject_id,session)

def move_t1brainmask():
    subject_list = os.listdir(OG_ROOT_DIR_PATH)
    subject_list.sort()
    for subject_id in subject_list:
        if "sub-" not in subject_id:continue
        session_list = os.listdir(join(OG_ROOT_DIR_PATH,subject_id))
        for session in session_list:
            if "ses-" in session:
                acq_list = os.listdir(join(OG_ROOT_DIR_PATH,subject_id,session))
                if "anat" in acq_list:
                    anat_dir_path = join(OG_ROOT_DIR_PATH,subject_id,session,"anat")
                    t1_filenames = os.listdir(anat_dir_path)
                    if "T1Brain_noSkull_mask.nii.gz" in t1_filenames:
                        for filename in t1_filenames:
                            if "T1Brain_noSkull_mask.nii.gz" in filename:
                                new_filename  = f"{subject_id}_{session}_run-01_acq-memprage_T1w_brainmask.nii.gz"
                                dest_filepath = join(NEW_BIDS_DIR_PATH,subject_id,session,"anat",new_filename)
                                og_filepath   = join(anat_dir_path,filename)
                                debug.info("Copying",subject_id,session,"anat",filename)
                                debug.info("to     ",dest_filepath)
                                # os.remove(og_filepath)
                                shutil.copy(og_filepath, dest_filepath)
                                # sys.exit()
                    else:
                        debug.warning("No t1 brain mask found in",subject_id,session)
                else:
                    debug.warning("No anat found in",subject_id,session)                

# move_t1brain()
move_t1brainmask()