import numpy as np
from tools.datautils import DataUtils
from tools.debug import Debug
from os.path import join, split 
import os , math
import matplotlib.pyplot as plt
from tools.filetools import FileTools
from graphplot.simmatrix import SimMatrix
from bids.mridata import MRIData
from bids.mridata import MRIData
import nibabel as nib
from connectomics.nettools import NetTools
from connectomics.netcluster import NetCluster
from nilearn import plotting, image, datasets
import ants

import copy, sys
from registration.registration import Registration
from connectomics.network import NetBasedAnalysis
from connectomics.parcellate import Parcellate
from connectomics.robustness import NetRobustness
from graphplot.nodal_simmilarity import NodalSimilarity
from rich.console import Console
from rich.table import Table

dutils    = DataUtils()
resultdir = join(dutils.ANARESULTSPATH,"connectomes_M_vs_S")
OUTDIR    = join(dutils.ANARESULTSPATH,"PLOS")
os.makedirs(OUTDIR,exist_ok=True)

GROUP     = "Mindfulness-Project"
simplt    = SimMatrix()
reg       = Registration()
ftools    = FileTools(GROUP)
debug     = Debug()
nettools  = NetTools()
nba       = NetBasedAnalysis()
parc      = Parcellate()
netrobust = NetRobustness()
netclust  = NetCluster()
pltnodal  = NodalSimilarity()

############ Meta stat sign #########3
ALPHA = 0.05
MAX_FIBRE_LENGTH = 250
############ Labels ############
subject_id,session = "S001","V1"
mridata  = MRIData(subject_id,session ,group=GROUP)
m_connectome_dir_path = join(mridata.ROOT_PATH,"derivatives","connectomes",
                        f"sub-{subject_id}",f"ses-{session}","spectroscopy")
filename = f"sub-{subject_id}_ses-{session}_run-01_acq-memprage_atlas-chimeraLFMIHIFIF_desc-scale3grow2mm_dseg_simmatrix.npz"
data                 = np.load(join(m_connectome_dir_path,filename))
parcel_labels        = data["labels"]

####################################

data = np.load(join(resultdir,"simM_metab_struct.npz"))
# length_con_arr      = data["distance_matrix_arr"]
metab_con_arr              = data["metab_con_sp_arr"]
subject_id_arr             = data["subject_id_arr"]
session_arr                = data["session_arr"]
qmask_arr                  = data["qmask_arr"]
label_indices              = data["label_indices"]
simmatrix_sp_leave_out_arr = data["simmatrix_sp_leave_out_arr"]
outpath            = join(dutils.ANARESULTSPATH,f"{GROUP}_threshold_list.npz")
data_th               = np.load(outpath)
subject_id_arr_th  = data_th["subject_id_arr"]
session_arr_th     = data_th["session_arr"]
threshold_arr      = data_th["best_thresholds"]



n_subjects      = metab_con_arr.shape[0]
METABOLITES           = ["NAANAAG", "Ins", "GPCPCh", "GluGln", "CrPCr"]

sel_parcel_list = ["ctx-rh","subc-rh","thal-rh","amygd-rh","hipp-rh",
                   "ctx-lh","subc-lh","thal-lh","amygd-lh","hipp-lh","stem"]
FONTSIZE = 16

weighted_metab_sim            = list()
for idm,metab_sim in enumerate(metab_con_arr):
    idt = np.where((subject_id_arr[idm] == subject_id_arr_th) & (session_arr[idm] == session_arr_th))[0]
    if len(idt)==0: continue
    weighted_metab_sim.append(metab_sim)

weighted_metab_sim         = np.array(weighted_metab_sim)


density = 0.10
simmatrix_pop  = weighted_metab_sim.mean(axis=0)
zero_diag_indices = np.where(np.diag(simmatrix_pop) == 0)[0]
simmatrix_pop_clean = np.delete(simmatrix_pop, zero_diag_indices, axis=0)
simmatrix_pop_clean = np.delete(simmatrix_pop_clean, zero_diag_indices, axis=1)
binarized_matrix_pos = nba.binarize(simmatrix_pop_clean,threshold=density,mode="pos",threshold_mode="density",binarize=False)
binarized_matrix_neg  = nba.binarize(simmatrix_pop_clean,threshold=density,mode="neg",threshold_mode="density",binarize=False)
binarized_matrix_abs  = nba.binarize(simmatrix_pop_clean,threshold=density,mode="abs",threshold_mode="density")
binarized_matrix_posneg  = nba.binarize(simmatrix_pop_clean,threshold=density,mode="posneg",threshold_mode="density")

_simmatrix_subj = weighted_metab_sim[0]
zero_diag_indices = np.where(np.diag(_simmatrix_subj) == 0)[0]
_simmatrix_subj = np.delete(_simmatrix_subj, zero_diag_indices, axis=0)
simmatrix_subj = np.delete(_simmatrix_subj, zero_diag_indices, axis=1)
binarized_matrix_sub_posneg  = nba.binarize(simmatrix_subj,threshold=density,mode="posneg",threshold_mode="density")


label_indices = label_indices[0:-1]
label_indices = np.delete(label_indices, zero_diag_indices)
parcel_labels = parcel_labels[0:-1]
parcel_labels = np.delete(parcel_labels, zero_diag_indices)

n_parcels = len(parcel_labels)


# outpath = join(OUTDIR,"Simmatrix_MFT_group_subj")
# fig, axs = plt.subplots(2,2, figsize=(16, 12))  # Adjust size as necessary
np.fill_diagonal(simmatrix_pop_clean, 0, wrap=False)
# simplt.plot_simmatrix(simmatrix_subj,ax=axs[0,0],titles=f"Weighted Matrix",
#                     colormap="bluewhitered",show_colorbar=True,
#                     result_path=None,scale_factor=0.3)
# simplt.plot_simmatrix(binarized_matrix_sub_posneg,ax=axs[0,1],titles=f"Binarized Matrix {round(100*density)}% Connection Density",
#                     colormap="bluewhitered",show_colorbar=True,
#                     result_path=None,scale_factor=0.3)
# np.fill_diagonal(simmatrix_pop_clean, 0, wrap=False)
# simplt.plot_simmatrix(simmatrix_pop_clean,ax=axs[1,0],titles=f"Weighted Matrix",
#                     colormap="bluewhitered",show_colorbar=True,
#                     result_path=None,scale_factor=0.3)
# simplt.plot_simmatrix(binarized_matrix_posneg,ax=axs[1,1],titles=f"Binarized Matrix {round(100*density)}% Connection Density",
#                     colormap="bluewhitered",show_colorbar=True,
#                     result_path=outpath,scale_factor=0.3)
# plt.show()
# sys.exit()


# cluster_labels         = netclust.consensus_clustering(np.abs(simmatrix_pop), n_clusters=4, n_iterations=10)
# sorted_indices = np.argsort(cluster_labels)
# modular_simmatrix_pop = simmatrix_pop[sorted_indices, :][:, sorted_indices]



mni_template    = datasets.load_mni152_template()


parcel_t1w_path = mridata.data["parcels"]["LFMIHIFIF-3"]["orig"]["path"]
transform_list  = mridata.get_transform("forward","anat")
parcel_mni_img  = reg.transform(fixed_image=ants.from_nibabel(mni_template),moving_image=parcel_t1w_path,
                                interpolator_mode="genericLabel",transform=transform_list)
parcel_mni_img_nii = nib.Nifti1Image(parcel_mni_img.numpy(), mni_template.affine)




colormap="bluewhitered"
colormap=pltnodal.color_bars(colormap)
mni_template            = datasets.load_mni152_template()
parcellation_data_np    = parcel_mni_img_nii.get_fdata()
nodal_similarity_matrix = pltnodal.nodal_similarity(binarized_matrix_posneg)

# Ensure that the number of parcels matches the number of similarity values

# Create an empty 3D image to store the similarity values
nodal_strength_map_np = np.zeros(parcellation_data_np.shape)

# Fill the similarity map with the similarity values
for i, value in enumerate(label_indices):
    nodal_strength_map_np[parcellation_data_np == value] = nodal_similarity_matrix[i]  # Assuming parcel indices start from 1

# Convert to Nifti1Image

# similarity_img = nib.Nifti1Image(nodal_strength_map_np, mni_template.affine)


# output_file=join(OUTDIR,"NodalSimmilarity_MFT_group_weighted.pdf")
# # Save the resulting image (Optional)
# plotting.plot_stat_map(similarity_img, cmap=colormap, 
#                         vmin=np.min(nodal_similarity_matrix),
#                         vmax=np.max(nodal_similarity_matrix),
#                         bg_img=mni_template,
#                         colorbar=True)
# plt.show()
ftools.save_nii_file(nodal_strength_map_np,mni_template.header,join(OUTDIR,"nodal_strength_map_weighted.nii.gz"))
sys.exit()












nodal_strength_map_np_weighted = pltnodal.plot(parcel_mni_img_nii,simmatrix_pop_clean,label_indices,
                                      vmin=None,vmax=None,colormap="bluewhitered",
                                      output_file=join(OUTDIR,"NodalSimmilarity_MFT_group_weighted.pdf"))

ftools.save_nii_file(nodal_strength_map_np_weighted,mni_template.header,join(OUTDIR,"nodal_strength_map_weighted.nii.gz"))

outpath = join(OUTDIR,"Simmatrix_MFT_group_subj")

density=0.20
# nodal_strength_np_weighted = pltnodal.nodal_similarity(simmatrix_pop_clean)
# nodal_strength_np_posneg   = nba.binarize(nodal_strength_np_weighted/(n_parcels-1),threshold=density,mode="posneg",threshold_mode="density")
# nodal_strength_np_pos      = nba.binarize(nodal_strength_np_weighted/(n_parcels-1),threshold=density,mode="pos",threshold_mode="density")
# nodal_strength_np_neg      = nba.binarize(nodal_strength_np_weighted/(n_parcels-1),threshold=density,mode="neg",threshold_mode="density")
# nodal_strength_np_abs      = nba.binarize(nodal_strength_np_weighted/(n_parcels-1),threshold=density,mode="abs",threshold_mode="density")

# nodal_strength_map_np_weighted = pltnodal.plot(parcel_mni_img_nii,nodal_strength_np_weighted,label_indices,
#                                       vmin=None,vmax=None,colormap="bluewhitered",
#                                       output_file=join(OUTDIR,"NodalSimmilarity_MFT_group_weighted.pdf"))

# nodal_strength_map_np = pltnodal.plot(parcel_mni_img_nii,nodal_strength_np_pos,label_indices,
#                                       vmin=None,vmax=None,colormap="Reds",
#                                       output_file=join(OUTDIR,"NodalSimmilarity_MFT_group_pos.pdf"))

# nodal_strength_map_np = pltnodal.plot(parcel_mni_img_nii,nodal_strength_np_neg,label_indices,
#                                       vmin=None,vmax=None,colormap="Blues",
#                                       output_file=join(OUTDIR,"NodalSimmilarity_MFT_group_neg.pdf"))

# nodal_strength_map_np = pltnodal.plot(parcel_mni_img_nii,nodal_strength_np_posneg,label_indices,
#                                       vmin=None,vmax=None,colormap="bluewhitered",
#                                       output_file=join(OUTDIR,"NodalSimmilarity_MFT_group_posneg.pdf"))


# ftools.save_nii_file(nodal_strength_map_np_weighted,mni_template.header,join(OUTDIR,"nodal_strength_map_weighted.nii.gz"))
# ftools.save_nii_file(nodal_strength_np_posneg,mni_template.header,join(OUTDIR,"nodal_strength_map.nii.gz"))
# ftools.save_nii_file(nodal_strength_np_pos,mni_template.header,join(OUTDIR,"nodal_strength_map_pos.nii.gz"))
# ftools.save_nii_file(nodal_strength_np_neg,mni_template.header,join(OUTDIR,"nodal_strength_map_neg.nii.gz"))


sys.exit()








fig, axs = plt.subplots(1,2, figsize=(16, 12))  # Adjust size as necessary
simplt.plot_simmatrix(simmatrix_pop,ax=axs[0],titles=f"M",
                    colormap="jet",show_colorbar=False, show_parcels="VH",
                    result_path=None,scale_factor=0.3)
simplt.plot_simmatrix(simmatrix_pop_clean,ax=axs[1],titles=f"M+,M-",
                    colormap="jet",show_colorbar=False,parcel_labels=parcel_labels, show_parcels="VH",
                    result_path=None,scale_factor=0.3)
plt.show()