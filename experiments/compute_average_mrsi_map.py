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
METABOLITE_LIST    = ["CrPCr","GluGln","GPCPCh","NAANAAG","Ins"]


############ Quality check subject list ##################
quality_list_path  = join(dutils.DATAPATH,GROUP,"mrsi_quality_check.json")
with open(quality_list_path,"r") as f:
    quality_list = json.load(f)
############ List all subjects ##################
recording_list = ftools.list_recordings()
subject_id_arr = recording_list[:,0]
debug.info(subject_id_arr)
subject_id_arr = np.unique(subject_id_arr)
# Initialize the Progress context manager outside the loop

mridata = MRIData("S002", "V2")
_template = mrsi_mni_nifti = mridata.get_mrsi_volume("Ins", "mni").get_fdata()
mrsi_mni_metall_np = np.zeros(_template.shape+(6,))
with Progress() as progress:
    counts = 0
    for ids,recording in enumerate(subject_id_arr):
        subject_id = recording
        if quality_list[subject_id]["V1"]["spectroscopy"]==-1:
            if quality_list[subject_id]["V2"]["spectroscopy"]==-1:
                continue
            else:
                session = "V2"
        else:
            session = "V1"
        # subject_id, session = recording_id
        debug.info(recording,ids,"/",len(subject_id_arr))
        if quality_list[subject_id][session]["spectroscopy"] == 0:
            debug.warning(recording, "mrsi not found")
            continue
        mridata = MRIData(subject_id, session)
        outpath_dir = join(dutils.ANARESULTSPATH,"PLOS","MRSI_avg",GROUP)
        # Create a progress task for this recording_id
        # task1 = progress.add_task("[red]Transforming...", total=len(METABOLITE_LIST) * 2)  # *2 because there are two updates per metabolite
        try:
            for idm, metabolite in enumerate(METABOLITE_LIST):
                outpath = mridata.get_path("spectroscopy", metabolite, "mni")
                mrsi_mni_nifti = mridata.get_mrsi_volume(metabolite, "mni")
                mrsi_mni_metall_np[:,:,:,idm]+=mrsi_mni_nifti.get_fdata()
            counts+=1
        except:print("Error",recording)
        

    os.makedirs(outpath_dir,exist_ok=True)
    outpath = join(outpath_dir,f"average_mrsi_4D_group-{GROUP}.nii.gz")
    mrsi_mni_metall_np[:,:,:,-1] = np.mean(mrsi_mni_metall_np[:,:,:,0:5],axis=-1)
    ftools.save_nii_file(mrsi_mni_metall_np/counts, mrsi_mni_nifti.header, f"{outpath}")

        








        
    #################################
debug.title("DONE")
debug.separator()











    






