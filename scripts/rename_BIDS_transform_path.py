import os, sys
import numpy as np
from tools.datautils import DataUtils
from os.path import join, split
from tools.debug import Debug
import shutil


dutils   = DataUtils()
debug    = Debug()

GROUP = "Mindfulness-Project"

BIDS_ROOT_PATH     = join(dutils.DATAPATH,GROUP)
PATH_2_MNI         = join(dutils.DATAPATH,"MNI","MNI152_T1_1mm_brain.nii.gz")
ANTS_TRANFORM_PATH = join(BIDS_ROOT_PATH,"derivatives","transforms","ants")
METABOLITE_LIST    = ["CrPCr","GluGln","GPCPCh","NAANAAG","Ins"]
METABOLITE_REF     = "GluGln"

def move_glx_trasnforms_to_spectroscopy():
    recording_list = list()
    subject_list = os.listdir(ANTS_TRANFORM_PATH)
    for subject_id in subject_list:
        if "sub-" not in subject_id:continue
        session_list = os.listdir(join(ANTS_TRANFORM_PATH,subject_id))
        for session in session_list:
            if "ses-" in session:
                acq_list = os.listdir(join(ANTS_TRANFORM_PATH,subject_id,session))
                for filename in acq_list:
                    # debug.info("Copying",filename)
                    if "_GluGln_to_t1" in filename:
                        filepath_og = join(ANTS_TRANFORM_PATH,subject_id,session,filename)
                        if "spectroscopy" in acq_list:
                            new_filename = filename.replace("_t1","_t1w")
                            dest_path = join(ANTS_TRANFORM_PATH,subject_id,session,"spectroscopy",new_filename)
                            debug.info("Copying",filepath_og)
                            debug.info("to     ",dest_path)
                            os.remove(filepath_og)
                            # shutil.copy(filepath_og, dest_path)

def rename_t1_to_t1w_spectroscopy():
    recording_list = list()
    subject_list = os.listdir(ANTS_TRANFORM_PATH)
    for subject_id in subject_list:
        if "sub-" not in subject_id:continue
        session_list = os.listdir(join(ANTS_TRANFORM_PATH,subject_id))
        for session in session_list:
            if "ses-" in session:
                acq_list = os.listdir(join(ANTS_TRANFORM_PATH,subject_id,session))
                if "spectroscopy" in acq_list:
                    mrsi_list        = os.listdir(join(ANTS_TRANFORM_PATH,subject_id,session,"spectroscopy"))
                    for filename in mrsi_list:
                        if "_t1." in filename: 
                            new_filename = filename.replace("_t1.","_t1w.")
                            filepath_og  = join(ANTS_TRANFORM_PATH,subject_id,session,"spectroscopy",filename)
                            dest_path    = join(ANTS_TRANFORM_PATH,subject_id,session,"spectroscopy",new_filename)
                            debug.info("Copying",filepath_og)
                            debug.info("to     ",dest_path)
                            # shutil.copy(filepath_og, dest_path)
                            os.remove(filepath_og)

def rename_Cr_to_CrPCr_spectroscopy():
    subject_list = os.listdir(ANTS_TRANFORM_PATH)
    for subject_id in subject_list:
        if "sub-" not in subject_id:continue
        session_list = os.listdir(join(ANTS_TRANFORM_PATH,subject_id))
        for session in session_list:
            if "ses-" in session:
                acq_list = os.listdir(join(ANTS_TRANFORM_PATH,subject_id,session))
                if "spectroscopy" in acq_list:
                    mrsi_list        = os.listdir(join(ANTS_TRANFORM_PATH,subject_id,session,"spectroscopy"))
                    for filename in mrsi_list:
                        if "mrsi_Cr_" in filename: 
                            new_filename = filename.replace("mrsi_Cr_","mrsi_CrPCr_")
                            filepath_og  = join(ANTS_TRANFORM_PATH,subject_id,session,"spectroscopy",filename)
                            dest_path    = join(ANTS_TRANFORM_PATH,subject_id,session,"spectroscopy",new_filename)
                            debug.info("Copying",filepath_og)
                            debug.info("to     ",dest_path)
                            # shutil.copy(filepath_og, dest_path)
                            os.remove(filepath_og)

def rename_spectroscopy_transform():
    subject_list = os.listdir(ANTS_TRANFORM_PATH)
    for subject_id in subject_list:
        if "sub-" not in subject_id:continue
        session_list = os.listdir(join(ANTS_TRANFORM_PATH,subject_id))
        for session in session_list:
            if "ses-" in session:
                acq_list = os.listdir(join(ANTS_TRANFORM_PATH,subject_id,session))
                if "spectroscopy" in acq_list:
                    mrsi_list        = os.listdir(join(ANTS_TRANFORM_PATH,subject_id,session,"spectroscopy"))
                    for filename in mrsi_list:
                        fileprefix = f"{subject_id}_{session}_desc-mrsi_to_t1w"
                        if f"{subject_id}-{session}" in filename:
                            if "syn.nii.gz" in filename: 
                                new_filename = f"{fileprefix}.syn.nii.gz"
                            elif "syn_inv.nii.gz" in filename: 
                                new_filename = f"{fileprefix}.syn_inv.nii.gz"
                            elif "affine.mat" in filename: 
                                new_filename = f"{fileprefix}.affine.mat"
                            elif "affine_inv.mat" in filename: 
                                new_filename = f"{fileprefix}.affine_inv.mat"

                            filepath_og  = join(ANTS_TRANFORM_PATH,subject_id,session,"spectroscopy",filename)
                            dest_path    = join(ANTS_TRANFORM_PATH,subject_id,session,"spectroscopy",new_filename)
                            debug.info("Copying",filepath_og)
                            debug.info("to     ",dest_path)
                            shutil.copy(filepath_og, dest_path)
                            os.remove(filepath_og)
                    # sys.exit()

# move_glx_trasnforms_to_spectroscopy()
# rename_t1_to_t1w_spectroscopy()
# rename_Cr_to_CrPCr_spectroscopy()
rename_spectroscopy_transform()