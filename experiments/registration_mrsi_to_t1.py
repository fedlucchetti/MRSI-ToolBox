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
from os.path import split, join, exists
from tools.filetools import FileTools
from tools.debug import Debug
import nibabel as nib
import random
from os.path import join, split
from connectomics.parcellate import Parcellate
from bids.mridata import MRIData
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '30'

GROUP = "Mindfulness-Project"

dutils   = DataUtils()
ftools   = FileTools(GROUP)
dsnn     = DeepSmoother()
debug    = Debug()
regtools = RegTools()
reg      = Registration()
pb       = ProgressBar()
pltsl    = PlotSlices()
parc     = Parcellate()

BIDS_ROOT_PATH     = join(dutils.DATAPATH,GROUP)
ANTS_TRANFORM_PATH = join(BIDS_ROOT_PATH,"derivatives","transforms","ants")
METABOLITE_LIST    = ["CrPCr","GluGln","GPCPCh","NAANAAG","Ins"]
METABOLITE_REF     = "Ins"

recording_list     = ftools.list_recordings()



############ OrigResFilter ##################
origres_filter = dict()
for metabolite in METABOLITE_LIST:
    filename  = f"OrigResSmoother_{metabolite}.h5"
    modelpath = os.path.join(dutils.DEVANALYSEPATH,"filters","neuralnet","models",filename)
    origres_filter[metabolite] = load_model(modelpath,compile=False)

############ List all subjects ##################
recording_list = ftools.list_recordings()
recording_id   = recording_list[0]

to_register_list = list()
debug.warning("The following recordings require registration")
for recording_id in recording_list:
    subject_id,session = recording_id
    transform_dir_path        = join(ANTS_TRANFORM_PATH,f"sub-{subject_id}",f"ses-{session}","spectroscopy")
    filename = f"sub-{subject_id}_ses-{session}_desc-mrsi_to_t1w.syn.nii.gz"
    if not exists(join(transform_dir_path,filename)):
        to_register_list.append([subject_id,session])
        debug.info(subject_id,session)

flag_continue = input("Confirm [Y,n]")
if flag_continue!="Y": sys.exit()


##########################################################
for ids, recording_id in enumerate(to_register_list):
    subject_id,session = recording_id
    mridata = MRIData(subject_id,session)
    debug.separator()
    debug.title(f"Processing {subject_id}-{session} --- {ids}/{len(to_register_list)}")
    ############ Denoise PCr+Cr  ##################
    
    __nifti        = mridata.data["mrsi"][METABOLITE_REF]["origfilt"]["nifti"]
    debug.info("__nifti",__nifti)
    if __nifti == 0:
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
            mrsi_filt_refout_path = mridata.data["mrsi"][metabolite]["orig"]["path"].replace("orig","origfilt")
            ftools.save_nii_file(smoothed_mrsi_ref_img, header_mrsi  ,f"{mrsi_filt_refout_path}.nii.gz")
            mridata.data["mrsi"][metabolite]["origfilt"]["path"] = f"{mrsi_filt_refout_path}.nii.gz"
        debug.success("Done")

    ############ MRSIto T1w Registration ##################  
    transform_dir_path        = join(ANTS_TRANFORM_PATH,f"sub-{subject_id}",f"ses-{session}","spectroscopy")
    transform_prefix          = f"sub-{subject_id}_ses-{session}_desc-mrsi_to_t1w"
    
    transform_dir_path        = join(ANTS_TRANFORM_PATH,f"sub-{subject_id}",f"ses-{session}","spectroscopy")
    transform_dir_prefix_path = join(transform_dir_path,f"{transform_prefix}")
    warpfilename              = f"sub-{subject_id}_ses-{session}_desc-mrsi_to_t1w.syn.nii.gz"
    if not exists(join(transform_dir_path,warpfilename)):
        debug.warning(f"{METABOLITE_REF} to T1w Registration not found or not up to date")
        syn_tx,_          = reg.register(fixed_input  = mridata.data["t1w"]["brain"]["orig"]["path"],
                                        moving_input  = mridata.data["mrsi"][METABOLITE_REF]["origfilt"]["path"],
                                        fixed_mask    = None, 
                                        moving_mask   = None,
                                        transform     = "sr",
                                        verbose       = 0)
        # Save Transform
        regtools.save_all_transforms(syn_tx,transform_dir_prefix_path)
    else:
        debug.success("Already registered")
    #################################
debug.title("DONE")
debug.separator()











    






