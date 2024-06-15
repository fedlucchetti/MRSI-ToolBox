import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import ants
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from registration.registration import Registration
from registration.tools import RegTools
from rich.progress import Progress
from tools.progress_bar import ProgressBar
from filters.neuralnet.deepsmoother import DeepSmoother
from tools.datautils import DataUtils
from graphplot.slices import PlotSlices
from tensorflow.keras.models import load_model
from os.path import split, join
from tools.filetools import FileTools
from tools.debug import Debug
import nibabel as nib
import random
from os.path import join, split
from connectomics.parcellate import Parcellate
from bids.mridata import MRIData
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '32'

dutils   = DataUtils()
ftools   = FileTools()
dsnn     = DeepSmoother()
debug    = Debug()
regtools = RegTools("Conc")
reg      = Registration()
pb       = ProgressBar()
pltsl    = PlotSlices()
parc     = Parcellate()

BIDS_ROOT_PATH     = "/media/veracrypt2/Connectome/Data/LPN-Project"
ANTS_TRANFORM_PATH = join(BIDS_ROOT_PATH,"derivatives","transforms","ants")
METABOLITE_LIST    = ["CrPCr","GluGln","GPCPCh","NAANAAG","Ins"]
METABOLITE_REF     = "Ins"

recording_list     = ftools.list_recordings()



############ OrigResFilter ##################
origres_filter = dict()
for metabolite in METABOLITE_LIST:
    filename  = f"OrigResSmoother_{metabolite}.h5"
    modelpath = os.path.join(dutils.DEVPATH,"Analytics","filters","neuralnet","models",filename)
    origres_filter[metabolite] = load_model(modelpath,compile=False)

############ List all subjects ##################
recording_list = ftools.list_recordings()
recording_id   = recording_list[0]
for ids, recording_id in enumerate(recording_list):
    subject_id,session = recording_id
    mridata = MRIData(subject_id,session)
    debug.separator()
    debug.title(f"Processing {subject_id}-{session} --- {ids}/{len(recording_list)}")
    ############ Denoise PCr+Cr  ##################
    __nifti        = mridata.data["mrsi"][METABOLITE_REF]["origfilt"]["nifti"]
    if __nifti is None:
        for metabolite in METABOLITE_LIST:
            debug.info("Denoise Orig",metabolite)
            # metabolite_key = metabolite.replace("+","")
            __nifti_mask   = mridata.data["mrsi"]["mask"]["orig"]["nifti"]
            __nifti        = mridata.data["mrsi"][metabolite]["orig"]["nifti"]
            origRes_img    = __nifti.get_fdata().squeeze()
            header_mrsi    = __nifti.header
            img            = np.expand_dims(origRes_img    ,axis=0)
            mask           = np.expand_dims(__nifti_mask.get_fdata().squeeze(),axis=0)
            smoothed_mrsi_ref_img, spike_mask = dsnn.proc(origres_filter[metabolite],origRes_img,mask)
            mridata.data["mrsi"][metabolite]["origfilt"]["nifti"] = smoothed_mrsi_ref_img
            mrsi_filt_refout_path = mridata.data["mrsi"]["mask"]["orig"]["path"].replace("orig","origfilt")
            ftools.save_nii_file(smoothed_mrsi_ref_img, header_mrsi  ,f"{mrsi_filt_refout_path}.nii")
            mridata.data["mrsi"][metabolite]["metabolite"]["path"] = f"{mrsi_filt_refout_path}.nii.gz"
        debug.success("Done")

    #################################
debug.title("DONE")
debug.separator()











    






