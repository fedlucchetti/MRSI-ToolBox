import os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '32'
from bids.mridata import MRIData
import time , os, sys
from tools.debug import Debug
import nibabel as nib
from tools.filetools import FileTools
from os.path import join,split, exists
from registration.registration import Registration
from tools.datautils import DataUtils
from registration.tools import RegTools
import subprocess

debug = Debug()
reg   = Registration()
regtools = RegTools()

GROUP    = "Mindfulness-Project"
dutils   = DataUtils()
BIDS_ROOT_PATH     = join(dutils.DATAPATH,GROUP)
ANTS_TRANFORM_PATH = join(BIDS_ROOT_PATH,"derivatives","transforms","ants")

NTHREADS = 32
ftools   = FileTools(GROUP)
duration = 147

recording_list = ftools.list_recordings()
recording = recording_list[0]
for idr,recording in enumerate(recording_list):
    debug.title(f"Processing {recording} - {idr}/{len(recording_list)}")
    subject_id,session=recording
    mridata = MRIData(subject_id,session,group=GROUP)
    # Parcellation
    # Paths to the original DWI files
    dwi_bval = mridata.data["dwi"]["bval"]
    dwi_bvec = mridata.data["dwi"]["bvec"]
    dwi_nii  = mridata.data["dwi"]["nifti"]
    # Paths to the original T1 files
    t1_nii          = mridata.data["t1w"]["brain"]["orig"]["path"]
    brainmask_nii   = mridata.data["t1w"]["mask"]["orig"]["path"]
    if dwi_nii==0 or t1_nii==0: 
        debug.error("No DWI or ANAT data found")
        continue



    dwi_rootname    = dwi_nii.replace("dwi.nii.gz","")
    anat_rootname   = t1_nii.replace("brain.nii.gz","")

    # Convert DWI and T1w images to MRtrix format
    dwi_mif       = f"{dwi_rootname}dwi.mif"
    t1_mif        = f"{anat_rootname}brain.mif"
    brainmask_mif = f"{anat_rootname}brainmask.mif"
    dwi_mask      = f"{dwi_rootname}dwi_mask.mif"
    mean_b0_mif   = dwi_mif.replace(".mif","_mean_b0.mif")
    mean_b0_nifti = mean_b0_mif.replace(".mif",".nii.gz")


    # Connectome
    anat_parcel_nifti          = mridata.data["parcels"]["LFMIHIFIF-3"]["orig"]["path"]
    anat_parcel_dwispace_nifti = anat_parcel_nifti.replace("space-orig","space-dwi")
    anat_parcel_dwispace_mif   = anat_parcel_dwispace_nifti.replace(".nii.gz",".mif")
    anat_parcel_dwispace_mif.replace(".gz","")


    if os.path.exists(anat_parcel_dwispace_mif):
        debug.success(f"Parcel already transformed to DWI space")
        continue

    ########### Compute B0 volume ############
    debug.info("Compute B0 volume")
    subprocess.run(["mrconvert", dwi_nii, dwi_mif, "-fslgrad", dwi_bvec, dwi_bval,"-force","-quiet"])
    os.system(f"dwiextract {dwi_mif} - -bzero | mrmath - mean  {mean_b0_mif} -axis 3 -force -quiet")
    d = os.system(f"mrconvert {mean_b0_mif} {mean_b0_nifti} -force -quiet")
    debug.success("Done")
    if not os.path.exists(mean_b0_nifti):
        continue

    ############ DWI T1w Registration ##################  
    transform_dir_path        = join(ANTS_TRANFORM_PATH,f"sub-{subject_id}",f"ses-{session}","dwi")
    transform_prefix          = f"sub-{subject_id}_ses-{session}_desc-dwi_to_t1w"
    transform_dir_prefix_path = join(transform_dir_path,f"{transform_prefix}")
    if not os.path.exists(f"{transform_dir_prefix_path}.syn.nii.gz") :
        proc=None
        try:
            debug.warning(f"DWI to T1w Registration not found or not up to date")
            proc = subprocess.Popen(['python3', f"tools/progress_bar.py ",f"{str(int(duration))}" ])
            pid = proc.pid
            syn_tx,duration  = reg.register(fixed_input=t1_nii,moving_input=mean_b0_nifti,transform="sr")
            proc.terminate()
            regtools.save_all_transforms(syn_tx,transform_dir_prefix_path)
        except KeyboardInterrupt:
            if proc:
                proc.terminate()
    else:
        debug.success("Already registered")


    debug.info(f"Transforming Parcel image to DWI space")
    transform_list = mridata.get_transform(direction="inverse",space="dwi")
    anat_parcel_np = nib.load(anat_parcel_nifti).get_fdata()
    transformed_img_np = reg.transform(f"{mean_b0_nifti}",anat_parcel_nifti,transform_list,interpolator_mode="genericLabel").numpy()
    transformed_img_np = transformed_img_np.astype(int)
    ftools.save_nii_file(transformed_img_np,nib.load(mean_b0_nifti).header,anat_parcel_dwispace_nifti.replace(".gz",""))
    d = os.system(f"mrconvert {anat_parcel_dwispace_nifti} {anat_parcel_dwispace_mif} -datatype int16 -force -quiet")
    debug.success(f"DONE")
    debug.separator()


# sift_2M                  = f"{dwi_rootname}sift_2M.txt"
# tracts_tck               = f"{dwi_rootname}tracts.tck"
# anat_parcel_dwispace_csv = anat_parcel_dwispace_nifti.replace(".nii.gz",".csv")
# assignment_parcel        = join( split(anat_parcel_dwispace_csv)[0], f"assigments_{split(anat_parcel_dwispace_csv)[1]}")

# command = [
#     "tck2connectome",
#     "-symmetric",
#     "-zero_diagonal",
#     "-scale_invnodevol",
#     "-tck_weights_in", sift_2M,
#     tracts_tck, anat_parcel_dwispace_mif, anat_parcel_dwispace_csv,
#     "-out_assignment", assignment_parcel,
#     "-force","-nthreads",f"{NTHREADS}"
# ]
# subprocess.run(command)


# # Load the connectome data
# connectome_df = pd.read_csv(anat_parcel_dwispace_csv, header=None)  # Assuming there's no header
# connectome_gm = np.array(connectome_df)[0:172,0:172]
# connectome_gm = connectome_gm/np.max(connectome_gm)
# plt.imshow(connectome_gm,cmap="plasma")
# plt.show()

# plt.hist(connectome_gm.flatten())
# plt.show()


# import networkx as nx

# # Create a graph from the connectome matrix
# G = nx.from_numpy_array(connectome_gm)

# # Draw the graph
# plt.figure(figsize=(12, 10))
# pos = nx.spring_layout(G)  # positions for all nodes
# nx.draw_networkx_nodes(G, pos, node_size=70)
# nx.draw_networkx_edges(G, pos, width=0.5)
# nx.draw_networkx_labels(G, pos, font_size=8)
# plt.title('Connectome Graph Visualization')
# plt.show()