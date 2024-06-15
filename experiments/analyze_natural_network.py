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
import networkx as nx
from sklearn.cluster import KMeans

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
parcel_labels_group        = copy.deepcopy(data["labels"])
parcel_labels_subj         = copy.deepcopy(data["labels"])

####################################

data = np.load(join(resultdir,"simM_metab_struct.npz"))
for k in data.keys():print(k)
# length_con_arr      = data["distance_matrix_arr"]
metab_con_arr              = data["metab_con_sp_arr"]
subject_id_arr             = data["subject_id_arr"]
session_arr                = data["session_arr"]
qmask_arr                  = data["qmask_arr"]
label_indices_group              = data["label_indices"]
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


label_indices_group       = label_indices_group[0:-1]
label_indices_group       = np.delete(label_indices_group, zero_diag_indices)
parcel_labels_group = parcel_labels_group[0:-1]
parcel_labels_group = np.delete(parcel_labels_group, zero_diag_indices)
n_parcels_group     = len(parcel_labels_group)


_simmatrix_subj = weighted_metab_sim[0]
zero_diag_indices = np.where(np.diag(_simmatrix_subj) == 0)[0]
_simmatrix_subj = np.delete(_simmatrix_subj, zero_diag_indices, axis=0)
simmatrix_subj = np.delete(_simmatrix_subj, zero_diag_indices, axis=1)
binarized_matrix_sub_posneg  = nba.binarize(simmatrix_subj,threshold=density,mode="posneg",threshold_mode="density")
label_indices_subj = data["label_indices"]
label_indices_subj = label_indices_subj[0:-1]

label_indices_subj = np.delete(label_indices_subj, zero_diag_indices)
parcel_labels_subj = parcel_labels_subj[0:-1]
parcel_labels_subj = np.delete(parcel_labels_subj, zero_diag_indices)
n_parcels_subj     = len(parcel_labels_subj)



############### Plot Sim Matrices ###############
np.fill_diagonal(simmatrix_pop_clean, 0, wrap=False)
# outpath = join(OUTDIR,"Simmatrix_MFT_group_subj")
# fig, axs = plt.subplots(2,2, figsize=(16, 12))  # Adjust size as necessary
# simplt.plot_simmatrix(simmatrix_subj,ax=axs[0,0],titles=f"Individual Weighted Matrix",
#                     colormap="bbo",show_colorbar=True,
#                     result_path=None,scale_factor=0.3)
# simplt.plot_simmatrix(binarized_matrix_sub_posneg,ax=axs[0,1],titles=f"Individual Binarized Matrix {round(100*density)}% Edge Density",
#                     colormap="bbo",show_colorbar=True,
#                     result_path=None,scale_factor=0.3)
# np.fill_diagonal(simmatrix_pop_clean, 0, wrap=False)
# simplt.plot_simmatrix(simmatrix_pop_clean,ax=axs[1,0],titles=f"Group Weighted Matrix",
#                     colormap="bbo",show_colorbar=True,
#                     result_path=None,scale_factor=0.3)
# simplt.plot_simmatrix(binarized_matrix_posneg,ax=axs[1,1],titles=f"Group Binarized Matrix {round(100*density)}% Edge Density",
#                     colormap="bbo",show_colorbar=True,
#                     result_path=outpath,scale_factor=0.3)
# plt.show()
# sys.exit()


# cluster_labels         = netclust.consensus_clustering(np.abs(simmatrix_pop), n_clusters=4, n_iterations=10)
# sorted_indices = np.argsort(cluster_labels)
# modular_simmatrix_pop = simmatrix_pop[sorted_indices, :][:, sorted_indices]


############## GET MNI Parcellation ###############
mni_template    = datasets.load_mni152_template()
parcel_t1w_path = mridata.data["parcels"]["LFMIHIFIF-3"]["orig"]["path"]
transform_list  = mridata.get_transform("forward","anat")
parcel_mni_img  = reg.transform(fixed_image=ants.from_nibabel(mni_template),moving_image=parcel_t1w_path,
                                interpolator_mode="genericLabel",transform=transform_list)
parcel_mni_img_nii = nib.Nifti1Image(parcel_mni_img.numpy(), mni_template.affine)
###########################################################################



density=0.20
nodal_strength_np_weighted = pltnodal.nodal_similarity(simmatrix_pop_clean)
simmatrix_pop_weighted_plus = copy.deepcopy(simmatrix_pop_clean)
simmatrix_pop_weighted_neg  = copy.deepcopy(simmatrix_pop_clean)
simmatrix_pop_weighted_plus[simmatrix_pop_clean<0] = 0
simmatrix_pop_weighted_neg[simmatrix_pop_clean>0]  = 0

nodal_strength_np_weighted = pltnodal.nodal_similarity(simmatrix_pop_clean)
nodal_strength_np_weighted_plus = pltnodal.nodal_similarity(simmatrix_pop_weighted_plus)
nodal_strength_np_weighted_neg  = pltnodal.nodal_similarity(simmatrix_pop_weighted_neg)

nodal_strength_np_weighted/=(n_parcels_group-1)
nodal_strength_map_np_weighted = pltnodal.plot(parcel_mni_img_nii,nodal_strength_np_weighted,label_indices_group,
                                      vmin=None,vmax=None,colormap="bbo",slices=[-5,-13,8],
                                      output_file=join(OUTDIR,"NodalSimmilarity_MFT_weighted_group.pdf"))

nodal_strength_map_np_weighted_plus = pltnodal.plot(parcel_mni_img_nii,nodal_strength_np_weighted_plus,label_indices_group,
                                      vmin=None,vmax=None,colormap="Reds",slices=[-5,-13,8],
                                      output_file=join(OUTDIR,"NodalSimmilarity_MFT_weighted_group_plus.pdf"))

nodal_strength_map_np_weighted_neg = pltnodal.plot(parcel_mni_img_nii,-nodal_strength_np_weighted_neg,label_indices_group,
                                      vmin=None,vmax=None,colormap="Blues",slices=[-5,-13,8],
                                      output_file=join(OUTDIR,"NodalSimmilarity_MFT_weighted_group_neg.pdf"))

ftools.save_nii_file(nodal_strength_map_np_weighted,mni_template.header,join(OUTDIR,"nodal_strength_map_weighted_group.nii.gz"))


density=0.20
nodal_strength_np_weighted_subj = pltnodal.nodal_similarity(simmatrix_subj)
nodal_strength_np_weighted_subj/=(n_parcels_subj-1)
debug.info("nodal_strength_np_weighted_subj",nodal_strength_np_weighted_subj.shape)
debug.info("label_indices_subj             ",label_indices_subj.shape)

nodal_strength_map_np_weighted_subj = pltnodal.plot(parcel_mni_img_nii,nodal_strength_np_weighted_subj,label_indices_subj,
                                      vmin=None,vmax=None,colormap="bbo",slices=[-5,-13,8],
                                      output_file=join(OUTDIR,"NodalSimmilarity_MFT_subj_weighted_subj.pdf"))

ftools.save_nii_file(nodal_strength_np_weighted,mni_template.header,join(OUTDIR,"nodal_strength_map_weighted_group_sum.nii.gz"))
ftools.save_nii_file(nodal_strength_map_np_weighted_subj,mni_template.header,join(OUTDIR,"nodal_strength_map_weighted_subj.nii.gz"))
ftools.save_nii_file(nodal_strength_map_np_weighted_plus,mni_template.header,join(OUTDIR,"nodal_strength_map_weighted_group_pos.nii.gz"))
ftools.save_nii_file(nodal_strength_map_np_weighted_neg,mni_template.header,join(OUTDIR,"nodal_strength_map_weighted_group_neg.nii.gz"))


def rgb_to_grayscale_with_zero(R, B ):
    grayscale = 0.299 * R  + 0.114 * B
    return grayscale

def vector_to_scalar(x, y):
    # Calculate the angle in radians
    theta = math.atan2(y, x)
    # Normalize the gle to [0, 1]
    theta_normalized = (theta + math.pi) / (2 * math.pi)
    # Scale the normalized angle to [0, 255]an
    scalar_value = 255 * theta_normalized
    return scalar_value

vectorized_to_scalar = np.vectorize(vector_to_scalar)
vectorized_rgb_to_gray = np.vectorize(rgb_to_grayscale_with_zero)

nodal_strength_np_weighted_hsv = vectorized_rgb_to_gray(nodal_strength_map_np_weighted_plus/np.max(nodal_strength_map_np_weighted_plus)*255,
                                                        nodal_strength_map_np_weighted_neg/np.max(nodal_strength_map_np_weighted_neg)*255)

nodal_strength_np_weighted_scalar = vectorized_to_scalar(nodal_strength_map_np_weighted_plus/np.max(nodal_strength_map_np_weighted_plus), 
                                                         nodal_strength_map_np_weighted_neg/np.max(nodal_strength_map_np_weighted_neg))
ftools.save_nii_file(nodal_strength_np_weighted_scalar,mni_template.header,join(OUTDIR,"nodal_strength_map_weighted_group_scalar.nii.gz"))
ftools.save_nii_file(nodal_strength_np_weighted_scalar,mni_template.header,join(OUTDIR,"nodal_strength_map_weighted_group_scalar.nii.gz"))



# sys.exit()






######### RichClub ########
rc_degrees_M,rc_coefficients_M,mean_rc_M, std_rc_M, rc_deg_cutoff_M,pvalues = nba.get_rc_distribution(np.abs(binarized_matrix_abs))
rich_club_node_indices, rich_club_node_degrees = nba.extract_rich_club_network(binarized_matrix_abs,rc_deg_cutoff_M)

rich_club_node_degrees = binarized_matrix_abs[rich_club_node_indices].sum(axis=0)
network_node_degrees   = binarized_matrix_abs.sum(axis=0)
parcel_positions       = np.array(parc.compute_centroids(parcel_mni_img.numpy(),label_indices_group))
parcel_positions.shape

parcel_data_clust = np.zeros([len(label_indices_group),4])
for idx, rc_idx in enumerate(label_indices_group):
    # print(parcel_labels_group[rc_idx],",",rich_club_node_degrees[rc_idx],",",round(nodal_strength_np_weighted[rc_idx],2))
    x,y,z = parcel_positions[idx]
    parcel_data_clust[idx]= [nodal_strength_np_weighted[idx],x,y,z]


num_clusters_arr = np.arange(1,23)  # For example, testing 1 through 10 clusters
wcss = np.zeros(num_clusters_arr.shape)  # List to store the within-cluster sums of squares
for i,n_clusters in enumerate(num_clusters_arr):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=10000, n_init=10, random_state=42)
    kmeans.fit(parcel_data_clust)
    wcss[i] = kmeans.inertia_
plt.plot(num_clusters_arr,wcss,"-o")
plt.show()

kmeans = KMeans(n_clusters=6, init='k-means++', max_iter=10000, n_init=10, random_state=42)
kmeans.fit(parcel_data_clust)
parcel_data = np.zeros([len(label_indices_group),5])
for idx, rc_idx in enumerate(label_indices_group):
    print(kmeans.labels_[idx],",",network_node_degrees[idx],",",round(nodal_strength_np_weighted[idx],2),",",parcel_labels_group[idx])
    x,y,z = parcel_positions[idx]
    parcel_data[idx]= [network_node_degrees[idx],nodal_strength_np_weighted[idx],x,y,z]








fig, axs = plt.subplots(1, figsize=(16, 12))  # Adjust size as necessary
axs.plot(rc_degrees_M, rc_coefficients_M, label='Metabolic Network', color='blue')
axs.plot(rc_degrees_M, pvalues, label='pvalues', color='red')

axs.fill_between(rc_degrees_M, mean_rc_M - std_rc_M, mean_rc_M + std_rc_M, color='gray', alpha=0.5, label='Random Network ±1σ')
axs.vlines(rc_deg_cutoff_M, 0, 1, colors='k', linestyles='dashed')

axs.set_xlabel('Degree',fontsize=FONTSIZE)
axs.set_ylabel('Rich-Club Coefficient',fontsize=FONTSIZE)
axs.legend()
axs.grid()
#axs.set_yscale("log")
plt.tight_layout() 
plt.show()


sys.exit()







