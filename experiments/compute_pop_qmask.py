import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import copy
from registration.registration import Registration
from registration.tools import RegTools
from rich.progress import Progress
from tools.progress_bar import ProgressBar
from tools.datautils import DataUtils
from filters.neuralnet.deepsmoother import DeepSmoother
from tools.datautils import DataUtils
from graphplot.slices import PlotSlices
from os.path import split, join, exists
from tools.filetools import FileTools
from tools.debug import Debug
import nibabel as nib
import random
from os.path import join, split
from nilearn import datasets

from bids.mridata import MRIData
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '16'
import json

#############################
GROUP   = "Mindfulness-Project"
SNR_TH  = 4
FWHM_TH = 0.1
CRLB_TH = 20
QMASKPOP_TH = 0.69
#############################

dutils   = DataUtils()
ftools   = FileTools(GROUP)
dsnn     = DeepSmoother()
debug    = Debug()
regtools = RegTools()
reg      = Registration()
pb       = ProgressBar()
pltsl    = PlotSlices() 
###############################

BIDS_ROOT_PATH     = join(dutils.DATAPATH,GROUP)
ANTS_TRANFORM_PATH = join(BIDS_ROOT_PATH,"derivatives","transforms","ants")
OUTDIR             = join(BIDS_ROOT_PATH,"derivatives","group","qmask")

METABOLITE_LIST    = ["CrPCr","GluGln","GPCPCh","NAANAAG","Ins"]

recording_list     = ftools.list_recordings()
quality_list_path  = join(dutils.DATAPATH,GROUP,"mrsi_quality_check.json")
with open(quality_list_path,"r") as f:
    quality_list = json.load(f)

############ List all subjects ##################
recording_list = ftools.list_recordings()
############ Create QMASK template ##################
recording_id   = recording_list[0]
subject_id,session = recording_id
mridata = MRIData(subject_id,session)
mask    = mridata.get_mrsi_volume("mask","mni").get_fdata()
met_qmask = np.zeros((len(METABOLITE_LIST),)+mask.shape)

subject_id_arr = np.unique(recording_list[:,0])
count = 0
for idm,recording in enumerate(subject_id_arr):
    subject_id = recording
    if quality_list[subject_id]["V1"]["spectroscopy"]==-1:
        if quality_list[subject_id]["V2"]["spectroscopy"]==-1:
            continue
        else:
            session = "V2"
    else:
        session = "V1"
    debug.info("Load",subject_id,session,quality_list[subject_id][session]["spectroscopy"])
    mridata = MRIData(subject_id,session)
    try:
        snr        = mridata.get_mrsi_volume("snr","mni").get_fdata()
        fwhm       = mridata.get_mrsi_volume("fwhm","mni").get_fdata()
        mask       = np.ones(snr.shape)
        mask[snr<SNR_TH]   = 0
        mask[fwhm>FWHM_TH] = 0
        for idm,metabolite in enumerate(METABOLITE_LIST):
            crlb        = mridata.get_mrsi_volume(f"{metabolite}-crlb","mni").get_fdata()
            mask[crlb>CRLB_TH] = 0
            met_qmask[idm] += mask
        count+=1
    except:
        debug.warning("No",subject_id,session,"found")

met_qmask /= count
debug.success("Averaged ",count,"recordings")
met_qmask_pop = copy.deepcopy(met_qmask)
met_qmask_pop[met_qmask<=QMASKPOP_TH] = 0
met_qmask_pop[met_qmask>QMASKPOP_TH]  = 1

mni_template    = datasets.load_mni152_template()
os.makedirs(OUTDIR,exist_ok=True)
for idm, metabolite in enumerate(METABOLITE_LIST):
    outpath = join(OUTDIR,f"group_space-mni_acq-qmask_desc-{metabolite}_spectroscopy.nii.gz")
    ftools.save_nii_file(met_qmask_pop[idm],mni_template.header,outpath)



        
    #################################
debug.title("DONE")
debug.separator()











    






