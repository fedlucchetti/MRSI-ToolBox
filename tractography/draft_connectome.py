import subprocess
from bids.mridata import MRIData
import time , os, sys
from tools.debug import Debug
import nibabel as nib
from tools.filetools import FileTools
from os.path import join,split, exists
import pandas as pd
import numpy as np 
from tools.datautils import DataUtils

import matplotlib.pyplot as plt
debug = Debug()
dutils = DataUtils
GROUP    = "Mindfulness-Project"
NTHREADS = 32
ftools   = FileTools(GROUP)



recording_list = ftools.list_recordings()
recording = recording_list[42]
# for recording in recording_list:
debug.title(f"Processing {recording}")
subject_id,session=recording
mridata = MRIData(subject_id=recording[0],session=recording[1],group=GROUP)
connectome_dir_path = join(mridata.ROOT_PATH,"derivatives","connectomes",
                           f"sub-{subject_id}",f"ses-{session}","anat")

# Parcellation
# Paths to the original DWI files
dwi_bval = mridata.data["dwi"]["bval"]
dwi_bvec = mridata.data["dwi"]["bvec"]
dwi_nii  = mridata.data["dwi"]["nifti"]
# Paths to the original T1 files
t1_nii          = mridata.data["t1w"]["brain"]["orig"]["path"]
brainmask_nii   = mridata.data["t1w"]["mask"]["orig"]["path"]
# if dwi_nii==0 or t1_nii==0: 
#     debug.error("No DWI or ANAT data found")
#     continue

dwi_rootname    = dwi_nii.replace("dwi.nii.gz","")
anat_rootname   = t1_nii.replace("brain.nii.gz","")

# Convert DWI and T1w images to MRtrix format
dwi_mif       = f"{dwi_rootname}dwi.mif"
t1_mif        = f"{anat_rootname}brain.mif"
brainmask_mif = f"{anat_rootname}brainmask.mif"
dwi_mask      = f"{dwi_rootname}dwi_mask.mif"

# Response times
response_sfwm = f"{dwi_rootname}dwi_response_sfwm.txt"
response_gm   = f"{dwi_rootname}dwi_response_gm.txt"
response_csf  = f"{dwi_rootname}dwi_response_csf.txt"

# FODs
sfwm_fod      = response_sfwm.replace("txt","mif")
gm_fod        = response_gm.replace("txt","mif")
csf_fod       = response_csf.replace("txt","mif")
vf_fod        = response_csf.replace("csf.txt","vf.mif")
sfwm_fod_norm = sfwm_fod.replace(".mif","_norm.mif")
gm_fod_norm   = gm_fod.replace(".mif","_norm.mif")
csf_fod_norm  = csf_fod.replace(".mif","_norm.mif")


# Segmentation files
anat_5tt_nocoreg = t1_mif.replace(".mif","_5tt_nocoreg.mif")
mean_b0          = dwi_mif.replace(".mif","_mean_b0.mif")
mean_b0_nifti    = mean_b0.replace(".mif",".nii.gz")
anat_5tt_nocoreg_dwi = anat_5tt_nocoreg.replace("/anat/","/dwi/")
anat_5tt_nocoreg_dwi_nifti = anat_5tt_nocoreg_dwi.replace(".mif",".nii.gz")
anat_5tt_vol0      = anat_5tt_nocoreg_dwi_nifti.replace("5tt_nocoref.nii.gz","5tt_vol0.nii.gz")
anat_5tt_coreg_dwi = anat_5tt_nocoreg_dwi_nifti.replace("nocoreg.mif","coreg.mif")

# DWI to T1 Registration
diff2struct_transf = f"{dwi_rootname}diff2struct_fsl.mat"
gmwmSeed_coreg     = f"{dwi_rootname}gmwmSeed_coreg.mif"

# Tractography 
tracts_tck         = f"{dwi_rootname}tracts.tck"
sift_mu            = f"{dwi_rootname}sift_mu.txt"
sift_coeffs        = f"{dwi_rootname}sift_coeffs.txt"
sift_2M            = f"{dwi_rootname}sift_2M.txt"
smallerTracks_200k = f"{dwi_rootname}smallerTracks_200k.tck"

# Anatomical parcellation
t1_recon = f"{anat_rootname}recon-all"

# Connectome
anat_parcel_nifti          = mridata.data["parcels"]["LFMIHIFIF-3"]["orig"]["path"]
anat_parcel_dwispace_nifti = anat_parcel_nifti.replace("space-orig","space-dwi")
anat_parcel_dwispace_mif   = anat_parcel_dwispace_nifti.replace(".nii.gz",".mif")
connectome_density         = f"{dwi_rootname}connectome_density.csv"
connectome_length          = f"{dwi_rootname}connectome_length.csv"
connectome_fa              = f"{dwi_rootname}connectome_fa.csv"


assignment_parcel          = join( split(connectome_density)[0], f"assigments_{split(connectome_density)[1]}")








##############################################################################
################################ Connectome ################################
##############################################################################

# Tracks
sift_2M                  = f"{dwi_rootname}sift_2M.txt"
tracts_tck               = f"{dwi_rootname}tracts.tck"
tracts_length            = f"{dwi_rootname}lengths.csv"
# FA
fa_mic = f"{dwi_rootname}fa.mif"
fa_per_stream_csv = f"{dwi_rootname}fa_per_streamline.csv"


os.system(f"tckedit {tracts_tck} -number 200k {smallerTracks_200k} -force")
os.system(f"mrview {dwi_mif} -tractography.load {smallerTracks_200k}")


# Get diffusion tensor then write FA  to fa_mic
os.system(f"dwi2tensor {dwi_mif} - | tensor2metric - -fa {fa_mic}")
# get FA alonmg each streamline
os.system(f"tcksample {tracts_tck} {fa_mic} {fa_per_stream_csv} -stat_tck mean -nthreads {NTHREADS} -force")




# $ tck2connectome tracks.tck nodes.mif connectome.csv -tck_weights_in weights.csv -out_assignments assignments.txt

command = [
    "tck2connectome",
    "-symmetric",
    "-zero_diagonal",
    "-scale_invlength",
    "-scale_invnodevol",
    "-tck_weights_in", sift_2M,
    tracts_tck, anat_parcel_dwispace_mif, connectome_density,
    "-out_assignment", assignment_parcel,
    "-force","-nthreads",f"{NTHREADS}"
]
subprocess.run(command)
connectome_density_np = np.array(pd.read_csv(connectome_density, header=None).values)
connectome_density_np = connectome_density_np[0:172,0:172]


# tck2connectome tracks.tck nodes.mif distances.csv  -scale_length -stat_edge mean
command = [
    "tck2connectome",
    "-symmetric",
    "-zero_diagonal",
    "-scale_invnodevol",
    "-stat_edge","mean",
    tracts_tck, anat_parcel_dwispace_mif, connectome_length,
    "-force","-nthreads",f"{NTHREADS}"
]
subprocess.run(command)
connectome_length_np = np.array(pd.read_csv(connectome_length, header=None).values)
connectome_length_np = connectome_length_np[0:172,0:172]
plt.imshow(connectome_length_np)
plt.show()

command = [
    "tck2connectome",
    "-symmetric",
    "-zero_diagonal",
    "-scale_invnodevol",
    "-scale_file", fa_per_stream_csv,
    "-stat_edge","mean",
    tracts_tck, anat_parcel_dwispace_mif, connectome_fa,
    "-force","-nthreads",f"{NTHREADS}"
]
subprocess.run(command)
connectome_fa_np = np.array(pd.read_csv(connectome_length, header=None).values)
connectome_fa_np = connectome_fa_np[0:172,0:172]
plt.imshow(connectome_fa_np)
plt.show()

filename = split(anat_parcel_nifti)[1].replace("space-dwi_","")
filename = filename.replace(".nii.gz","_simmatrix.npz")

os.makedirs(connectome_dir_path,exist_ok=True)
outpath = join(connectome_dir_path,filename)
np.savez(outpath,
         connectome_density=connectome_density_np,
         connectome_length=connectome_length_np,
         connectome_fa=connectome_fa_np)