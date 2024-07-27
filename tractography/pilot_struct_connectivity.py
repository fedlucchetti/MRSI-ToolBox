import subprocess
from bids.mridata import MRIData
import time , os, sys
import numpy as np 
from tools.debug import Debug
import nibabel as nib
from tools.filetools import FileTools
from os.path import join,split, exists
from tools.datautils import DataUtils
from registration.registration import Registration
from tractography.tracttools import TractTools
import pandas as pd
# from graphplot.simmatrix import SimMatrix
import matplotlib.pyplot as plt
import copy, json

debug    = Debug()
dutils   = DataUtils()
GROUP    = "PilotProject"
NTHREADS = 28
ftools   = FileTools(GROUP)
reg      = Registration()
tctools  = TractTools()
# simplt   = SimMatrix()





LOGPATH = dutils.ANALOGPATH

FREESURFER_PARC_DIR_PATH = join(dutils.DATAPATH,GROUP,"derivatives","freesurfer")
DWIPROC_DIR_PATH         = join(dutils.DATAPATH,GROUP,"derivatives","dwi-preprocessing")
TCKPROC_DIR_PATH         = join(dutils.DATAPATH,GROUP,"derivatives","tractography")
CONNECTOME_DIR_PATH      = join(dutils.DATAPATH,GROUP,"derivatives","connectomes")
CHIMERA_PARC_DIR_PATH    = join(dutils.DATAPATH,GROUP,"derivatives","chimera-atlases")
FREESURFER_LUT_PATH      = join(os.environ["FREESURFER_HOME"],"FreeSurferColorLUT.txt")


########################################
dict_path  = join(dutils.DEVPATH,"MRSI-ToolBox","tractography","progress.json")
with open(dict_path, 'r') as file:
    progress_dict = json.load(file)



subject_id,session = "P001","V1"
recording = subject_id,session
debug.title(f"Processing sub-{subject_id}_ses-{session}")
debug.separator()
mridata = MRIData(subject_id,session,group=GROUP)
dwi_bval = mridata.data["dwi"]["bval"]
dwi_bvec = mridata.data["dwi"]["bvec"]
dwi_nii  = mridata.data["dwi"]["nifti"]
# Paths to the original T1 files
t1_nii          = mridata.data["t1w"]["brain"]["orig"]["path"]
brainmask_nii   = mridata.data["t1w"]["mask"]["orig"]["path"]

if dwi_nii==0 or t1_nii==0: 
    debug.error("No DWI or ANAT data found")
    progress_dict[subject_id][session]["progress"]=f"Error: No DWI or ANAT data found"
    tctools.exit_run(dict_path,progress_dict)


START = time.time()
### Create result folders ###
dwi_rootname    = dwi_nii.replace("dwi.nii.gz","")
tck_rootname    = join(TCKPROC_DIR_PATH,f"sub-{subject_id}",f"ses-{session}")
conn_rootname   = join(CONNECTOME_DIR_PATH,f"sub-{subject_id}",f"ses-{session}","dwi")

anat_rootname   = t1_nii.replace("brain.nii.gz","")

# Convert DWI and T1w images to MRtrix format
# dwi_mif       = join(tck_rootname,"dwi.mif")
dwi_mif       = f"{dwi_rootname}dwi.mif"
t1_mif        = f"{anat_rootname}brain.mif"
brainmask_mif = f"{anat_rootname}brainmask.mif"
dwi_mask      = f"{dwi_rootname}dwi_mask.mif"

# B0 Fields
mean_b0                 = dwi_mif.replace(".mif","_mean_b0.mif")
mean_b0_nii             = mean_b0.replace(".mif",".nii.gz")

# Response times
response_sfwm   = join(tck_rootname,"dwi_response_sfwm.txt")
response_gm     = join(tck_rootname,"dwi_response_gm.txt")
response_csf    = join(tck_rootname,"dwi_response_csf.txt")


# FODs
sfwm_fod      = response_sfwm.replace("txt","mif")
gm_fod        = response_gm.replace("txt","mif")
csf_fod       = response_csf.replace("txt","mif")
vf_fod        = response_csf.replace("csf.txt","vf.mif")
sfwm_fod_norm = sfwm_fod.replace(".mif","_norm.mif")
gm_fod_norm   = gm_fod.replace(".mif","_norm.mif")
csf_fod_norm  = csf_fod.replace(".mif","_norm.mif")
# corrected by YASSER bip-up bip-down DTI
dipyfodf      = join(DWIPROC_DIR_PATH,f"sub-{subject_id}",f"ses-{session}","dwi","odfreconst",
                    f"sub-{subject_id}_ses-{session}_run-01_acq-dsiNdir257_space-dwi_model-SHORE_desc-dipyfodf_res-1x1x1mm_diffmodel.nii.gz")
dipyfodf_norm = dipyfodf.replace(".nii.gz","_norm.nii.gz")


# Segmentation files
subject_id,session
freesurfer_aseg         = join(FREESURFER_PARC_DIR_PATH,f"sub-{subject_id}_ses-{session}_run-01_acq-memprage","mri","aparc+aseg.mgz")
rawavg_vol              = join(FREESURFER_PARC_DIR_PATH,f"sub-{subject_id}_ses-{session}_run-01_acq-memprage","mri","rawavg.mgz")
freesurfer_aseg_anat    = freesurfer_aseg.replace("aseg.mgz","aseg_space-anat.mgz")
anat_5tt_space_anat     = join(tck_rootname,split(t1_mif)[1])
anat_5tt_space_anat     = anat_5tt_space_anat.replace("_acq","_space-anat_acq")
anat_5tt_space_anat     = anat_5tt_space_anat.replace("T1w_brain.mif","5tt_nocoreg.mif")

anat_5tt_space_anat_nii = anat_5tt_space_anat.replace(".mif",".nii.gz")
anat_5tt_vol0           = anat_5tt_space_anat_nii.replace("5tt_nocoreg.nii.gz","5tt_vol0.nii.gz")
anat_5tt_space_dwi      = anat_5tt_space_anat_nii.replace("nocoreg.nii.gz","coreg.mif")
anat_5tt_space_dwi      = anat_5tt_space_dwi.replace("_space-anat","_space-dwi")

# Parcel files
chimera_parcel          = mridata.data["parcels"]["LFMIHIFIF-3"]["orig"]["path"]  
wm_mask_anat_nii        = chimera_parcel.replace("atlas-chimeraLFMIHIFIF","wm_mask")
wm_mask_dwi_nii         = wm_mask_anat_nii.replace("space-orig","space-dwi")
wm_mask_dwi_mif         = wm_mask_dwi_nii.replace(".nii.gz",".mif")

# DWI to T1 Registration
diff2struct_transf = join(tck_rootname,"diff2struct_fsl.mat")
gmwmSeed_coreg     = join(tck_rootname,"gmwmSeed_coreg.mif")


# Tractography 
tracts_tck         = join(tck_rootname,"tracts.tck")
sift_mu            = join(tck_rootname,"sift_mu.txt")
sift_coeffs        = join(tck_rootname,"sift_coeffs.txt")
sift_2M            = join(tck_rootname,"sift_2M.txt")
smallerTracks_200k = join(tck_rootname,"smallerTracks_200k.tck")
smallerTracks_400k = join(tck_rootname,"smallerTracks_400k.tck")
smallerTracks_1000k = join(tck_rootname,"smallerTracks_1000k.tck")


# Connectome
anat_parcel_anatspace_nii  = mridata.data["parcels"]["LFMIHIFIF-3"]["orig"]["path"]
anat_parcel_dwispace_nii   = anat_parcel_anatspace_nii.replace("space-orig","space-dwi")
anat_parcel_dwispace_mif   = anat_parcel_dwispace_nii.replace(".nii.gz",".mif")
anat_parcel_dwispace_csv   = anat_parcel_dwispace_nii.replace(".nii.gz",".csv")
assignment_parcel          = join( split(anat_parcel_dwispace_csv)[0], f"assigments_{split(anat_parcel_dwispace_csv)[1]}")
connectome_density         = f"{dwi_rootname}connectome_density.csv"
connectome_length          = f"{dwi_rootname}connectome_length.csv"

# Connectivity outputfile
connfilename = split(anat_parcel_anatspace_nii)[1].replace("space-dwi_","")
connfilename = connfilename.replace(".nii.gz","_connectivity.npz")
connfilename = connfilename.replace("run-01_acq-memprage_space-orig_","")


# if exists(join(conn_rootname,connfilename)):
#     debug.success(f"{recording} already processed")
#     debug.separator()
#     sys.exit()



########### Create ROOT TCK DIR #############
os.makedirs(tck_rootname,exist_ok=True)
TCKSNAPSHOT_DIR = join(dutils.ANARESULTSPATH,"tmp_tractography")
os.makedirs(TCKSNAPSHOT_DIR,exist_ok=True)
##############################################################################
################### Convert to MIF  #####§##############
##############################################################################
start = time.time()
subprocess.run(["mrconvert", dwi_nii, dwi_mif, "-fslgrad", dwi_bvec, dwi_bval,"-force","-quiet"])
subprocess.run(["mrconvert", t1_nii, t1_mif,"-force","-quiet"])
subprocess.run(["mrconvert", brainmask_nii, brainmask_mif,"-force","-quiet"])
debug.separator()
# Compute DWI brain mask
debug.info("Compute DWI brain mask")
os.system(f"dwi2mask {dwi_mif} {dwi_mask} -nthreads {NTHREADS} -force -quiet")
debug.success("Done in ",round(time.time()-start,1),"sec")


##############################################################################
############## Estimate the response function for SFWM GM CSF #############
##############################################################################
if not exists(response_sfwm): 
    start = time.time()
    debug.info("Estimate the response function for SFWM GM CSF")
    # Estimate the response functions using 'dhollander' algorithm
    subprocess.run([
        "dwi2response", "dhollander", dwi_mif, response_sfwm, response_gm, response_csf,
        "-nthreads", f"{NTHREADS}","-quiet","-force"  # Properly formatted option with separate argument
    ])
    debug.success("Done in ",round(time.time()-start,1),"sec")


# ##############################################################################
# ################################# Compute FOD ################################
# ##############################################################################
start = time.time()
if not exists(sfwm_fod_norm): 
    debug.info("Compute FOD")
    subprocess.run(["dwi2fod", "msmt_csd", dwi_mif, response_sfwm, sfwm_fod, response_gm, gm_fod,response_csf, csf_fod, 
                    "-mask",dwi_mask,"-nthreads", f"{NTHREADS}","-force","-quiet"])
    debug.success("Done DWI")
    os.system(f"mrconvert -coord 3 0 {sfwm_fod} -force - | mrcat {csf_fod} {gm_fod} - {vf_fod} -force  -quiet")
    debug.info("FOD Normalization")
    os.system(f"mtnormalise {sfwm_fod} {sfwm_fod_norm} {gm_fod} {gm_fod_norm} {csf_fod} {csf_fod_norm} -mask {dwi_mask} -force  -quiet")
    debug.success("Done in ",round(time.time()-start,1),"sec")
    debug.separator()
    # os.system(f"mrview {vf_fod} -odf.load_sh {sfwm_fod}")


##############################################################################
################## Generate 5-tissue-type segmentation #######################
##############################################################################
if not exists(anat_5tt_space_anat): 
    start = time.time()
    debug.info("Register dseg from freesurfer to anat space")
    cmd_bashargs = ['mri_vol2vol', '--mov', freesurfer_aseg, '--targ', rawavg_vol, 
                    '--regheader', '--o', freesurfer_aseg_anat, '--no-save-reg', '--interp', 'nearest'] 
    try:
        with open('/dev/null', 'w') as fnull:
            subprocess.run(cmd_bashargs, stdout=fnull, stderr=subprocess.STDOUT, check=True)
    except subprocess.CalledProcessError:
        progress_dict[subject_id][session]["progress"]=f"mri_vol2vol: CalledProcessError"
        tctools.exit_run(dict_path,progress_dict)
    debug.info("Generate 5-tissue-type segmentation from ANAT image")
    try:
        command = [
            "5ttgen", "freesurfer", "-lut",FREESURFER_LUT_PATH, 
            freesurfer_aseg_anat, anat_5tt_space_anat,
            "-nthreads", str(NTHREADS), "-force", "-quiet"
        ]
        # Execute the command with a timeout of 300 seconds
        subprocess.run(command, timeout=100, check=True)
        progress_dict[subject_id][session]["progress"]=f"5tt: Done"
        tctools.exit_run(dict_path,progress_dict,exit=False)
        debug.success("Done in ", round(time.time() - start, 1), "sec")
    except subprocess.TimeoutExpired:
        duration = round(time.time() - start, 1)
        debug.error("Process timed out. Execution took longer than 300 seconds.")
        progress_dict[subject_id][session]["progress"]=f"5tt:  Execution took longer than 100 seconds"
        with open(join(LOGPATH,"run_all_log.txt"), 'a') as file:
            file.write(f"Timeout: {recording[0]}-{recording[1]}: Generate 5-tissue-type segmentation exceeded 300 seconds after {duration} seconds.\n")
        tctools.exit_run(dict_path,progress_dict)
    except subprocess.CalledProcessError:
        debug.error("An error occurred during execution.")
        progress_dict[subject_id][session]["progress"]=f"5tt: CalledProcessError"
        tctools.exit_run(dict_path,progress_dict)
    except KeyboardInterrupt:
        debug.error("KeyboardInterrupt")
        progress_dict[subject_id][session]["progress"]=f"5tt: KeyboardInterrupt"
        tctools.exit_run(dict_path,progress_dict)
    debug.separator()

    ##############################################################################
    ################ Coregistering the Diffusion and Anatomical Images ###############
    ##############################################################################
if not exists(gmwmSeed_coreg) and exists(anat_5tt_space_anat):
    start   = time.time()
    debug.info("Coregistering the Diffusion and Anatomical Images")
    # Convert anat B0 and segmentation images
    os.system(f"dwiextract {dwi_mif} - -bzero | mrmath - mean  {mean_b0} -axis 3 -force -quiet")
    os.system(f"mrconvert {mean_b0} {mean_b0_nii} -force -quiet")
    os.system(f"mrconvert {anat_5tt_space_anat} {anat_5tt_space_anat_nii} -force -quiet")

    os.system(f"fslroi {anat_5tt_space_anat_nii} {anat_5tt_vol0} 0 1 ")
    os.system(f"flirt -in {mean_b0_nii} -ref {anat_5tt_vol0} -interp nearestneighbour -dof 6 -omat {diff2struct_transf}")

    # Convert transform
    os.system(f"transformconvert {diff2struct_transf} {mean_b0_nii} {anat_5tt_space_anat_nii} flirt_import {diff2struct_transf} -force -quiet")

    # Apply transform Anat to DWI
    os.system(f"mrtransform {anat_5tt_space_anat} -linear {diff2struct_transf} -inverse {anat_5tt_space_dwi} -force -quiet")

    # Create seed boundary separate GM from WM
    os.system(f"5tt2gmwmi {anat_5tt_space_dwi} {gmwmSeed_coreg} -force -quiet")
    # os.system(f"mrview {dwi_mif} -overlay.load {gmwmSeed_coreg}")
    debug.success("Done in ",round(time.time()-start,1),"sec")
    debug.separator()
    # os.system(f"mrview {dwi_mif} -overlay.load {anat_5tt_space_anat} -overlay.colourmap 2 -overlay.load {anat_5tt_space_dwi} -overlay.colourmap 1")



##############################################################################
################## Create White Matter Mask ####################
##############################################################################
debug.info(f"Create White Matter Mask")
### Load Parcel and extract WM
parcel_img    = nib.load(chimera_parcel)
parcel_img_np = parcel_img.get_fdata()
_header       = parcel_img.header
# _header = nib.load(t1_nii).header
wm_mask_img   = np.zeros(parcel_img_np.shape)
wm_mask_img[parcel_img_np != 3000] = 0
wm_mask_img[parcel_img_np == 3000] = 1
ftools.save_nii_file(wm_mask_img, _header, wm_mask_anat_nii)

### Transform WM mask from ANAT to DWI space
transform_list     = mridata.get_transform(direction="inverse",space="dwi")
transformed_img_np = reg.transform(f"{mean_b0_nii}",wm_mask_anat_nii,transform_list,interpolator_mode="genericLabel").numpy()
transformed_img_np = transformed_img_np.astype(int)
ftools.save_nii_file(transformed_img_np,
                        nib.load(mean_b0_nii).header,
                        wm_mask_dwi_nii)
os.system(f"mrconvert {wm_mask_dwi_nii} {wm_mask_dwi_mif} -force -quiet")

#debug.info("saved white matter mask to ",wm_mask_dwi_mif)
##############################################################################
################## Anatomically Constrained Tractography ####################
##############################################################################
if exists(anat_5tt_space_dwi) :
    # dipyfodf_mask = dipyfodf.replace(".nii.gz","_mask.nii.gz")
    # fodf_img = nib.load(dipyfodf)
    # fodf_data = fodf_img.get_fdata()
    # # Generate the mask by thresholding the FODF data
    # mask = fodf_data.mean(axis=-1) > 0.001
    # # Convert the mask to uint8 type (binary mask)
    # mask = mask.astype(np.uint8)
    # # Create a new NIfTI image for the mask
    # mask_img = nib.Nifti1Image(mask, fodf_img.affine, fodf_img.header)
    # # Save the mask image to a file
    # nib.save(mask_img, dipyfodf_mask)
    # os.system(f"mtnormalise {dipyfodf} {dipyfodf_norm} -mask {dipyfodf_mask} -force  -quiet")
    start = time.time()
    debug.separator()
    
    debug.title("Anatomically Constrained Tractography")
    try:
        subprocess.run([
            "tckgen",
            "-algorithm", "SD_STREAM",
            "-act", anat_5tt_space_dwi,
            "-seed_gmwmi", gmwmSeed_coreg,
            "-maxlength", "300",
            "-cutoff", "0.06",
            "-select", "10000000",
            "-trials","30",
            dipyfodf,
            "-seed_image",wm_mask_dwi_mif,
            tracts_tck,
            "-nthreads", str(NTHREADS),
            "-force"
        ])
        debug.success("Done in ",round(time.time()-start,1),"sec")
        progress_dict[subject_id][session]["progress"]=f"ACT: Done"
        tctools.exit_run(dict_path,progress_dict,exit=False)
    except subprocess.CalledProcessError:
        debug.error("An error occurred during execution.")
        progress_dict[subject_id][session]["progress"]=f"ACT: An error occurred during execution."
        tctools.exit_run(dict_path,progress_dict)
    except KeyboardInterrupt:
        debug.error("KeyboardInterrupt")
        progress_dict[subject_id][session]["progress"]=f"ACT: KeyboardInterrupt"
        tctools.exit_run(dict_path,progress_dict)
    if not exists(tracts_tck):
        debug.error("ACT failed: SKIP")
        progress_dict[subject_id][session]["progress"]=f"ACT: {tracts_tck} does not exist"
        tctools.exit_run(dict_path,progress_dict)
    # Filter streamlines with tcksift2
    debug.info("Filter streamlines with tcksift2")
    os.system(f"tcksift2 -act {anat_5tt_space_dwi} -out_mu {sift_mu} -out_coeffs {sift_coeffs} -nthreads {NTHREADS} {tracts_tck} {gm_fod_norm} {sift_2M} -force  -quiet")
    ################# View Results ###############
    os.system(f"tckedit {tracts_tck} -number 200k {smallerTracks_200k} -force")
    tctools.tck2trk(mean_b0_nii,smallerTracks_200k,force=True)
    # tctools.take_screenshot(smallerTracks_200k, join(TCKSNAPSHOT_DIR,f"sub-{subject_id}_ses-{session}_tck200k.png"))
    # os.system(f"mrview {dwi_mif} -tractography.load {smallerTracks_200k}")
    debug.separator()



##############################################################################
################### Transform Parcel Image to DWI Space #######################
##############################################################################
debug.info(f"Transforming Parcel image to DWI space")
transform_list     = mridata.get_transform(direction="inverse",space="dwi")
if len(transform_list)==2:
    transformed_img_np = reg.transform(f"{mean_b0_nii}",anat_parcel_anatspace_nii,transform_list,interpolator_mode="genericLabel").numpy()
    transformed_img_np = transformed_img_np.astype(int)
    ftools.save_nii_file(transformed_img_np,nib.load(mean_b0_nii).header,anat_parcel_dwispace_nii.replace(".gz",""))
    d = os.system(f"mrconvert {anat_parcel_dwispace_nii} {anat_parcel_dwispace_mif} -datatype int16 -force -quiet")
    debug.success(f"DONE")
    debug.separator()
else:
    debug.error("Registration DWI->ANAT missing")
    progress_dict[subject_id][session]["progress"]="Registration DWI->ANAT missing"
    tctools.exit_run(dict_path,progress_dict)


##############################################################################
################################ Connectome #################################
##############################################################################
if exists(tracts_tck):
    start = time.time()
    debug.title("Compute streamline density connectome")
    command_density = [
        "tck2connectome",
        "-symmetric",
        "-zero_diagonal",
        "-scale_invlength",
        "-scale_invnodevol",
        "-tck_weights_in", sift_2M,
        tracts_tck, anat_parcel_dwispace_mif, connectome_density,
        "-out_assignment", assignment_parcel,"-quiet",
        "-force","-nthreads",f"{NTHREADS}",
    ]
    command_length = [
        "tck2connectome",
        "-symmetric",
        "-zero_diagonal",
        "-scale_length","-stat_edge", "mean",
        "-tck_weights_in", sift_2M,
        tracts_tck, anat_parcel_dwispace_mif, connectome_length,
        "-out_assignment", assignment_parcel,"-quiet",
        "-force", "-nthreads", f"{NTHREADS}",
    ]
    try:
        debug.info("Density")
        subprocess.run(command_density)
        progress_dict[subject_id][session]["progress"]="Connectome Density:  Done"
        debug.info("Length")
        subprocess.run(command_length)
        progress_dict[subject_id][session]["progress"]="Connectome Length: Done"
        tctools.exit_run(dict_path,progress_dict,exit=False)
        debug.success(f"Done in {round(time.time()-start)} s")
        connectome_density_np = np.array(pd.read_csv(connectome_density, header=None).values)
        connectome_length_np  = np.array(pd.read_csv(connectome_length, header=None).values)
        # Save as NUMPY Array
        debug.info("Saving connectivity matrix to file",connfilename)
        os.makedirs(conn_rootname,exist_ok=True)
        np.savez(join(conn_rootname,connfilename),
                connectome_density=connectome_density_np,
                connectome_length=connectome_length_np)
        debug.info("Removing large tracts_tck file ")
        os.remove(tracts_tck)
    except Exception as e:
        debug.error(e)
        progress_dict[subject_id][session]["progress"]="Error: Compute streamline density connectome"
        tctools.exit_run(dict_path,progress_dict)
        
##############################################################################
################################ Plot Connetivity Results #################################
##############################################################################


# if exists(join(conn_rootname,connfilename)):
#     debug.info("Creating Figures")
#     data = np.load(join(conn_rootname,connfilename))["connectome_density"]
#     connectome_density = np.zeros(data.shape)
#     connectome_density[data>0.0] = 1
#     fig, axs = plt.subplots(1, figsize=(16, 12))  # Adjust size as necessary
#     simplt.plot_simmmatrix(connectome_density[0:280,0:280],ax=axs,titles=f"Structural Connectome sub-{subject_id}_ses-{session}",colormap="inferno",
#                             result_path=join(conn_rootname,connfilename).replace(".npz","") )  
#     fig, axs = plt.subplots(1, figsize=(16, 12))  # Adjust size as necessary
#     simplt.plot_simmmatrix(connectome_density[0:280,0:280],ax=axs,titles=f"Structural Connectome sub-{subject_id}_ses-{session}",colormap="inferno",
#                             result_path=join(TCKSNAPSHOT_DIR,f"sub-{subject_id}_ses-{session}_struct_connectivity.pdf") )  
#     debug.success("Done")

progress_dict[subject_id][session]["progress"]=3
tctools.exit_run(dict_path,progress_dict,exit=False)
debug.success(f"Processed {subject_id}-{session} in ",round(time.time()-START,1),"sec")
debug.separator()
debug.separator()

