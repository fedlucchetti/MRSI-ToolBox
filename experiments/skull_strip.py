import os, sys
import numpy as np
import ants
from tqdm import tqdm
from registration.registration import Registration
from registration.tools import Tools
from rich.progress import Progress
from tools.progress_bar import ProgressBar

from os.path import split, join
from tools.filetools import FileTools
from tools.debug import Debug
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '16'


ft     = FileTools()
debug  = Debug()
tools  = Tools("Conc_Mask")
reg    = Registration()
pb     = ProgressBar()


###############################################################################
MRSI_2_T1_DIRNAME ="__MRSI_T1_REG"

################################################################################
METABOLITES    = ["Cr+PCr","Glu+Gln","GPC+PCh","Ins","NAA+NAAG"]
METABOLITE_REF = "Cr+PCr"

MRSI_IN_FILES = [f"OrigRes_{met}_conc" for met in METABOLITES] + \
                [f"OrigRes_{met}_CRLB" for met in METABOLITES] + \
                ["OrigRes_Voxel_SNR", "OrigRes_Voxel_FWHM",
                 "OrigRes_WaterSignal"]

MRSI_OUT_FILES = [f"{met}_Conc" for met in METABOLITES] + \
                 [f"{met}_CRLB" for met in METABOLITES] + \
                 ["Voxel_SNR", "Voxel_FWHM","WaterSignal.nii"]

############ List all subjects ##################
subject_list = ft.list_files_and_extract("MindfullTeen","OrigRes")

# start = int(sys.argv[1])
# end   = min(int(sys.argv[2]),len(subject_list))

# subject_list = subject_list[start:end]
# debug.info("processing",subject_list)
# if input("Continue? [Y,N]")!="Y":
#     sys.exit()

############## Progressbar start durations
for ids, subject_id in enumerate(subject_list):
    debug.separator()
    debug.title(f"Registering {subject_id}")
    debug.warning("Remaining ",len(subject_list)-ids)
    # mrsi_reg_path     = tools.get_mrsi_out_path(subject_id,MRSI_OUT_FILES[0])
    # mrsi_reg_dir_path = split(mrsi_reg_path)[0]
    # os.makedirs(join(mrsi_reg_dir_path,MRSI_2_T1_DIRNAME),exist_ok=True)
    # # mrsi_tmp_dir      = join(mrsi_reg_dir_path,"MRSI2T1")
    # os.makedirs(mrsi_tmp_dir,exist_ok=True)
    # if os.path.exists(mrsi_reg_path):
    #     debug.success("Already processed")
    #     continue
   

    ############ SETUP PATHS ##################
    mrsi_recording_id = tools.get_mrsi_recording(subject_id)
    mrsi_ref_path     = tools.get_mrsi_path(subject_id,"OrigRes_"+METABOLITE_REF+"_conc")
    t1_path           = tools.get_t1_path(subject_id)

    reg.load_images(t1_path ,mrsi_recording_id, METABOLITE_REF)

    ############ N4 BIAS + Skull Strio ##################

    debug.info(" Skull strip T1W ")
    brain_t1_path, brain_t1_mask_path = reg.skull_strip(t1_path,image_type="t1")
    debug.success("DONE")
