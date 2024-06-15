import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from registration.registration import Registration
from registration.tools import RegTools
from rich.progress import Progress
from tools.progress_bar import ProgressBar
from tools.datautils import DataUtils
from graphplot.slices import PlotSlices
from os.path import split, join
from tools.filetools import FileTools
from tools.debug import Debug
import nibabel as nib
import random
from os.path import join, split
from connectomics.parcellate import Parcellate
from bids.mridata import MRIData
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '32'
import ants
GROUP = "LPN-Project"


dutils   = DataUtils()
ftools   = FileTools(GROUP)
debug    = Debug()
reg      = Registration()
pb       = ProgressBar()
pltsl    = PlotSlices()


METABOLITE_REF     = sys.argv[1]
PATH_2_MNI         = join(dutils.DATAPATH,"MNI","MNI152_T1_1mm_brain.nii.gz")



OUTDIR      = join("/media","flucchetti","NSA1TB1","4Edgar")
BIDSOUTDIR  = join(dutils.DATAPATH,GROUP,"derivatives","transforms","ants")
SIGNAL_LIST = ["CrPCr","GluGln","GPCPCh","NAANAAG","Ins","maskMRSI"]
ACQ_LIST    = ["conc","crlb","snr","fwhm"]
############ List all subjects ##################
recording_list = ftools.list_recordings()
recording_id   = recording_list[0]
for ids, recording_id in enumerate(recording_list[39::]):
    subject_id,session = recording_id
    subject_session    = f"{subject_id}_{session}"
    mrsiData           = MRIData(subject_id,session)
    debug.separator()
    debug.title(f"Processing {subject_id}-{session} --- {ids}/{len(recording_list)}")
    brain_t1_path  = mrsiData.data["t1w"]["brain"]["orig"]["path"]
    header_t1w     = mrsiData.data["t1w"]["brain"]["orig"]["nifti"].header
    header_mni     = nib.load(PATH_2_MNI).header
    transform_list = mrsiData.get_transform("forward","spectroscopy",metabolite_ref=METABOLITE_REF)
    transform_list_mni = mrsiData.get_transform("forward","anat")
    ###############################
    signal         = SIGNAL_LIST[0]
    image_list_t1w     = [mrsiData.data["t1w"]["brain"]["orig"]["nifti"].get_fdata()]
    title_list_t1w     = ["T1w"]
    image_list_mni     = [nib.load(PATH_2_MNI).get_fdata()]
    title_list_mni     = ["MNI"]
    ###############################
    with Progress() as progress:
        task1 = progress.add_task("[cyan]Starting...", total=len(SIGNAL_LIST)*len(ACQ_LIST))
        for signal in SIGNAL_LIST:
            for acq in ACQ_LIST:
                if   acq=="crlb":space="orig"
                elif acq=="conc":space="origfilt"
                progress.update(task1, description=f"[bold green]Transforming {signal}-{acq}-{space} to T1&MNI[/bold green]")
                __nifiti = mrsiData.data["mrsi"][signal][space]["nifti"]
                __path   = mrsiData.data["mrsi"][signal][space]["path"]
                if __nifiti!=None:
                    ########### T1w ###########
                    transformed_img_np = reg.transform(brain_t1_path,ants.from_nibabel(__nifiti),transform_list).numpy()
                    filename       = f"sub-{subject_id}_ses-{session}_space-t1w_acq-{acq}_desc-{signal}_spectroscopy.nii"
                    outdirpath_t1w = join(OUTDIR,subject_id,session,f"space-t1w_refmet-{METABOLITE_REF}_mrsi")
                    outdirbidspath = join(BIDSOUTDIR,f"sub-{subject_id}",f"ses-{session}","spectroscopy",f"space-t1w_refmet-{METABOLITE_REF}_mrsi")
                    os.makedirs(outdirpath_t1w,exist_ok=True)
                    os.makedirs(outdirbidspath,exist_ok=True)
                    outpath        = join(outdirpath_t1w,filename)
                    outbidspath    = join(outdirbidspath,filename)  
                    ftools.save_nii_file(transformed_img_np, header_t1w  ,outpath )
                    ftools.save_nii_file(transformed_img_np, header_t1w  ,outbidspath )
                    mrsiData.data["mrsi"][signal]["t1w"]["nifti"] = nib.load(f"{outbidspath}.gz")
                    mrsiData.data["mrsi"][signal]["t1w"]["path"]  = f"{outbidspath}.gz"
                    if signal in SIGNAL_LIST[0:-1] and acq=="conc":
                        image_list_t1w.append(transformed_img_np)
                        title_list_t1w.append(signal)
                    progress.advance(task1)
                    ########### MNI ###########
                    transformed_img_np = reg.transform(PATH_2_MNI,f"{outbidspath}.gz",transform_list_mni).numpy()
                    filename       = f"sub-{subject_id}_ses-{session}_space-mni_acq-{acq}_desc-{signal}_spectroscopy.nii"
                    outdirpath_mni     = join(OUTDIR,subject_id,session,f"space-mni_refmet-{METABOLITE_REF}_mrsi")
                    outdirbidspath = join(BIDSOUTDIR,f"sub-{subject_id}",f"ses-{session}","spectroscopy",f"space-mni_refmet-{METABOLITE_REF}_mrsi")
                    os.makedirs(outdirpath_mni,exist_ok=True)
                    os.makedirs(outdirbidspath,exist_ok=True)
                    outpath        = join(outdirpath_mni,filename)
                    outbidspath    = join(outdirbidspath,filename)  
                    ftools.save_nii_file(transformed_img_np, header_mni  ,outpath )
                    ftools.save_nii_file(transformed_img_np, header_mni  ,outbidspath )
                    mrsiData.data["mrsi"][signal]["mni"]["nifti"] = nib.load(f"{outbidspath}.gz")
                    mrsiData.data["mrsi"][signal]["mni"]["path"]  = f"{outbidspath}.gz"
                    if signal in SIGNAL_LIST[0:-1] and acq=="conc":
                        image_list_mni.append(transformed_img_np)
                        title_list_mni.append(signal)
                    progress.advance(task1)


        pltsl.plot_img_slices(image_list_t1w,
                            np.linspace(5,95,8),
                            titles   = title_list_t1w,
                            outpath  = join(outdirpath_t1w,f"sub-{subject_id}_ses-{session}_refmet-{METABOLITE_REF}_to_t1w_all"),
                            PLOTSHOW = False,
                            mask     = mrsiData.data["mrsi"]["mask"]["t1w"]["nifti"].get_fdata())

        pltsl.plot_img_slices(image_list_mni,
                            np.linspace(5,95,8),
                            titles   = title_list_mni,
                            outpath  = join(outdirpath_mni,f"sub-{subject_id}_ses-{session}_refmet-{METABOLITE_REF}_to_mni_all"),
                            PLOTSHOW = False,
                            mask     = mrsiData.data["mrsi"]["mask"]["mni"]["nifti"].get_fdata())
    #################################
debug.title("DONE")
debug.separator()











    






