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
METABOLITE_REF     = "Ins"


############ OrigResFilter ##################
origres_filter = dict()
for metabolite in METABOLITE_LIST:
    filename  = f"OrigResSmoother_{metabolite}.h5"
    modelpath = os.path.join(dutils.DEVANALYSEPATH,"filters","neuralnet","models",filename)
    origres_filter[metabolite] = load_model(modelpath,compile=False)

recording_list     = ftools.list_recordings()
quality_list_path  = join(dutils.DATAPATH,GROUP,"mrsi_quality_check.json")
with open(quality_list_path,"r") as f:
    quality_list = json.load(f)

############ List all subjects ##################
recording_list = ftools.list_recordings()
for ids, recording_id in enumerate(recording_list):
    subject_id,session = recording_id  
    debug.title(f"Processing {subject_id}-{session} --- {ids}/{len(recording_list)}")
    # if quality_list[subject_id][session]["spectroscopy"]==-1:continue
    mridata = MRIData(subject_id,session)
    outpath = mridata.data["mrsi"][METABOLITE_LIST[0]]["orig"]["path"].replace("orig","origfilt")
    outpath = f"{outpath}"
    if exists(outpath):
        debug.success(recording_id,"Already fitlered")
        debug.success("Exists",outpath)
        continue
    for idm,metabolite in enumerate(METABOLITE_LIST):
        debug.info("Denoise Orig",metabolite)
        __nifti_mask   = mridata.data["mrsi"]["mask"]["orig"]["nifti"]
        __nifti        = mridata.data["mrsi"][metabolite]["orig"]["nifti"]
        origRes_img    = __nifti.get_fdata().squeeze()
        debug.info("origRes_img",origRes_img.shape)
        header_mrsi    = __nifti.header
        img            = np.expand_dims(origRes_img    ,axis=0)
        mask           = np.expand_dims(__nifti_mask.get_fdata().squeeze(),axis=0)
        if mask.shape[1]!=origRes_img.shape[0]:
            mask = np.zeros((1,)+origRes_img.shape)
            mask[:,origRes_img>0] = 1
        debug.info("mask",mask.shape)
        smoothed_mrsi_ref_img, spike_mask = dsnn.proc(origres_filter[metabolite],origRes_img,mask)
        mridata.data["mrsi"][metabolite]["origfilt"]["nifti"] = smoothed_mrsi_ref_img
        mrsi_filt_refout_path = mridata.data["mrsi"][metabolite]["orig"]["path"].replace("orig","origfilt")
        ftools.save_nii_file(smoothed_mrsi_ref_img, header_mrsi  ,f"{mrsi_filt_refout_path}")





        
    #################################
debug.title("DONE")
debug.separator()











    






