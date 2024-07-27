import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import copy
from registration.registration import Registration
from registration.tools import RegTools
from rich.progress import Progress
from tools.progress_bar import ProgressBar
from filters.neuralnet.deepsmoother import DeepSmoother
from tools.datautils import DataUtils
from graphplot.slices import PlotSlices
from os.path import split, join, exists
from tools.filetools import FileTools
from tools.debug import Debug
import nibabel as nib
from tensorflow.keras.models import load_model
from rich.progress import track
from tqdm import  tqdm

from os.path import join, split
from bids.mridata import MRIData
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '16'
import json
GROUP = "Mindfulness-Project"

dutils   = DataUtils()
ftools   = FileTools(GROUP)
dsnn     = DeepSmoother()
debug    = Debug()
regtools = RegTools()
reg      = Registration()
pb       = ProgressBar()
pltsl    = PlotSlices()

BIDS_ROOT_PATH     = join(dutils.DATAPATH,GROUP)
ANTS_TRANFORM_PATH = join(BIDS_ROOT_PATH,"derivatives","transforms","ants")
METABOLITE_LIST    = ["CrPCr","GluGln","GPCPCh","NAANAAG","Ins",
                      "CrPCr-crlb","GluGln-crlb","GPCPCh-crlb","NAANAAG-crlb","Ins-crlb",
                      "snr","fwhm","mask"]


############ Quality check subject list ##################
quality_list_path  = join(dutils.DATAPATH,GROUP,"mrsi_quality_check.json")
with open(quality_list_path,"r") as f:
    quality_list = json.load(f)
############ List all subjects ##################
recording_list = ftools.list_recordings()
# Initialize the Progress context manager outside the loop

# Initialize the Progress context manager outside the loop
with Progress() as progress:
    for ids, recording_id in enumerate(recording_list):
        subject_id, session = recording_id
        if quality_list[subject_id][session]["spectroscopy"] == 0:
            debug.warning(recording_id, "mrsi not found")
            continue
        debug.title(f"Processing {subject_id}-{session} --- {ids}/{len(recording_list)}")
        mridata = MRIData(subject_id, session)
        outpath = mridata.get_path("spectroscopy", "mask", "mni")
        if exists(outpath):
            debug.success(recording_id, "Already registered")
            continue

        # Create a progress task for this recording_id
        task1 = progress.add_task("[red]Transforming...", total=len(METABOLITE_LIST) * 2)  # *2 because there are two updates per metabolite

        for idm, metabolite in enumerate(METABOLITE_LIST):
            # Transform to anat space
            outpath = mridata.get_path("spectroscopy", metabolite, "t1w")
            mrsi_t1wspace_nifti = mridata.get_mrsi_volume(metabolite, "t1w")
            if outpath is not None and mrsi_t1wspace_nifti is not None:
                ftools.save_nii_file(mrsi_t1wspace_nifti.get_fdata(), mrsi_t1wspace_nifti.header, f"{outpath}")
            else:
                debug.error(recording_id, "outpath does not exist", outpath)
            progress.update(task1, advance=1)

            # Transform to mni space
            outpath = mridata.get_path("spectroscopy", metabolite, "mni")
            mrsi_mni_nifti = mridata.get_mrsi_volume(metabolite, "mni")
            if outpath is not None and mrsi_mni_nifti is not None:
                ftools.save_nii_file(mrsi_mni_nifti.get_fdata(), mrsi_mni_nifti.header, f"{outpath}")
            else:
                debug.error(recording_id, "outpath does not exist", outpath)
            progress.update(task1, advance=1)
        
        # Mark the task as complete
        progress.update(task1, completed=len(METABOLITE_LIST) * 2)








        
    #################################
debug.title("DONE")
debug.separator()











    






