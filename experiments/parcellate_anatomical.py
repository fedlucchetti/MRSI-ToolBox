import os, sys
import numpy as np
from registration.registration import Registration
from registration.tools import RegTools
from rich.progress import Progress
from tools.progress_bar import ProgressBar
from tools.datautils import DataUtils
from os.path import split, join, exists
from tools.filetools import FileTools
from tools.debug import Debug
from os.path import join, split
from bids.mridata import MRIData
from connectomics.parcellate import Parcellate
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '32'
import nibabel as nib
from nilearn import datasets
import argparse

# GROUP = "LPN-Project"
GROUP = "Mindfulness-Project"
# ATLAS_TYPE = "jhu_icbm_wm"
# ATLAS_TYPE = "aal"
# ATLAS_TYPE = "geometric_cubeK18mm"
# ATLAS_TYPE = "wm_cubeK18mm"
# ATLAS_TYPE = "aal"
ATLAS_TYPE = sys.argv[1]


parc     = Parcellate()
dutils   = DataUtils()
ftools   = FileTools(GROUP)
debug    = Debug()
regtools = RegTools()
reg      = Registration()
pb       = ProgressBar()

BIDS_ROOT_PATH     = join(dutils.DATAPATH,GROUP)
PATH_2_MNI         = join(dutils.DATAPATH,"MNI","MNI152_T1_1mm_brain.nii.gz")
PARC_PATH          = join(BIDS_ROOT_PATH,"derivatives","chimera-atlases")




def main():
    parser = argparse.ArgumentParser(description="Process some input parameters.")

    # Add arguments
    parser.add_argument('--atlas', type=str, required=True, choices=['LFMIHIFIF-2', 'LFMIHIFIF-3', 'LFMIHIFIF-4', 
                                                                     'geometric_cubeK18mm','geometric_cubeK23mm',
                                                                     'aal', 'destrieux','cerebellum'], 
                        help='Atlas choice (must be one of: LFMIHIFIF-2, LFMIHIFIF-3, LFMIHIFIF-4, geometric, aal, destrieux)')
    args       = parser.parse_args()
    ATLAS_TYPE = args.atlas
    ############ List all subjects ##################
    recording_list = ftools.list_recordings()
    recording_id   = recording_list[0]
    for ids, recording_id in enumerate(recording_list):
        subject_id,session      = recording_id
        outdir_path             = join(PARC_PATH,f"sub-{subject_id}",f"ses-{session}","anat")
        prefix_name             =  f"sub-{subject_id}_ses-{session}_run-01_acq-memprage_space-orig_atlas-{ATLAS_TYPE}_dseg"
        origspace_atlas_outpath = join(outdir_path,f"{prefix_name}.nii.gz")
        prefix                  = f"sub-{subject_id}_ses-{session}"
        if exists(origspace_atlas_outpath):
            debug.success("Already processed",prefix)
            # continue
        parcel_outputpath = join(outdir_path,origspace_atlas_outpath)
        mrsiData = MRIData(subject_id,session,group=GROUP)
        debug.separator()
        t1_path = mrsiData.data["t1w"]["brain"]["orig"]["path"]
        if not exists(t1_path):debug.warning("SKIP",prefix);continue
        debug.title(f"Processing {subject_id}-{session} --- {ids}/{len(recording_list)}")
        
        prefix_name =  f"sub-{subject_id}_ses-{session}_run-01_acq-memprage_space-mni_atlas-{ATLAS_TYPE}_dseg"
        # get atlas template
        mni_atlas_outpath = join(outdir_path,f"{prefix_name}.nii.gz")
        header_t1 = nib.load(t1_path).header
        parcel_image_mni, labels, indices, header_mni = parc.create_parcel_image(atlas_string=ATLAS_TYPE)
        # debug.info("np.unique(parcel_image_mni)",np.unique(parcel_image_mni).shape)
        # debug.info("                     labels",labels)
        # debug.info("                    indices",indices)
        os.makedirs(outdir_path,exist_ok=True)
        ftools.save_nii_file(parcel_image_mni,header_mni,mni_atlas_outpath)
        # trasnform atlas to t1 space
        transform_list    = mrsiData.get_transform("inverse","anat")
        parcel_image_orig = reg.transform(t1_path,mni_atlas_outpath,transform_list,interpolator_mode="genericLabel")
        # debug.info("parcel_image_orig labels",np.unique(parcel_image_orig.numpy()))
        ftools.save_nii_file(parcel_image_orig.numpy(),header_t1,origspace_atlas_outpath)
        parc.create_tsv(labels,indices,join(outdir_path,f"{prefix_name}.tsv"))
        prefix_name = prefix_name.replace("mni","orig")
        parc.create_tsv(labels,indices,join(outdir_path,f"{prefix_name}.tsv"))

if __name__ == "__main__":
    main()









    






