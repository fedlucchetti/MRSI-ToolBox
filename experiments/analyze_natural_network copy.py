import numpy as np
from tools.datautils import DataUtils
from tools.debug import Debug
from os.path import join, split 
import os , math
import matplotlib.pyplot as plt
from tools.filetools import FileTools
from graphplot.simmatrix import SimMatrixPlot
from bids.mridata import MRIData
import nibabel as nib
from connectomics.nettools import NetTools
from connectomics.netcluster import NetCluster
from nilearn import plotting, image, datasets
import ants
import networkx as nx
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
simplt    = SimMatrixPlot()
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
# for k in data.keys():print(k)
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
parcellation_data_np    = parcel_mni_img_nii.get_fdata()
ftools.save_nii_file(parcellation_data_np,mni_template.header,join(OUTDIR,"parcellation_mi152.nii.gz"))

###########################################################################



density=0.10
nodal_strength_np_weighted = pltnodal.nodal_similarity(simmatrix_pop_clean)
simmatrix_pop_weighted_plus = copy.deepcopy(simmatrix_pop_clean)
simmatrix_pop_weighted_neg  = copy.deepcopy(simmatrix_pop_clean)
simmatrix_pop_weighted_plus[simmatrix_pop_clean<0] = 0
simmatrix_pop_weighted_neg[simmatrix_pop_clean>0]  = 0

nodal_strength_np_weighted      = pltnodal.nodal_similarity(simmatrix_pop_clean)
nodal_strength_np_weighted_plus = pltnodal.nodal_similarity(simmatrix_pop_weighted_plus)
nodal_strength_np_weighted_neg  = pltnodal.nodal_similarity(simmatrix_pop_weighted_neg)


M_p =  copy.deepcopy(nodal_strength_np_weighted_plus)
M_n =  copy.deepcopy(nodal_strength_np_weighted_neg)
feature_matrix_2D = np.array([M_p, M_n]).T  # Example, adjust as per your actual vectors


##################### Perform Spectral Clustering #####################
n_clusters = 4  # Number of clusters
sc = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', assign_labels='kmeans')
feature_labels = sc.fit_predict(feature_matrix_2D)
unique_labels = set(feature_labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    class_member_mask = (feature_labels == k)
    xy = feature_matrix_2D[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)
plt.title('Spectral Clustering')
plt.show()
cluster_averages = np.array([feature_matrix_2D[feature_labels == k].mean(axis=0) for k in range(n_clusters)])
X_transformed_plusneg = np.array([cluster_averages[label] for label in feature_labels])

nodal_strength_map_np_weighted_plus = pltnodal.plot(parcel_mni_img_nii,X_transformed_plusneg[:,0],label_indices_group,
                                      vmin=None,vmax=None,colormap="Reds",slices=[-5,-13,8],
                                      output_file=join(OUTDIR,"NodalSimmilarity_MFT_weighted_group_plus_clusters.pdf"))
nodal_strength_map_np_weighted_neg = pltnodal.plot(parcel_mni_img_nii,-X_transformed_plusneg[:,1],label_indices_group,
                                      vmin=None,vmax=None,colormap="Blues",slices=[-5,-13,8],
                                      output_file=join(OUTDIR,"NodalSimmilarity_MFT_weighted_group_neg_clusters.pdf"))


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
nodal_strength_np_scalar_clusters = vectorized_to_scalar(nodal_strength_map_np_weighted_plus/np.max(nodal_strength_map_np_weighted_plus), 
                                                         nodal_strength_map_np_weighted_neg/np.max(nodal_strength_map_np_weighted_neg))


ftools.save_nii_file(nodal_strength_np_scalar_clusters,mni_template.header,join(OUTDIR,"nodal_strength_map_weighted_group_2D_SpectralClusteredPCA.nii.gz"))



# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(X_transformed_plusneg)
# Apply PCA to reduce to 1 component
pca = PCA(n_components=1)
pca_components2D = pca.fit_transform(scaled_features)





principal_component = pca_components2D.flatten()
NS_2D_map_np = pltnodal.nodal_strength_map(principal_component,parcellation_data_np,label_indices_group)
NS_2D_map_np_norm = 255*(NS_2D_map_np-np.max(NS_2D_map_np))/(np.min(NS_2D_map_np)-np.max(NS_2D_map_np))
similarity2D_img = nib.Nifti1Image(NS_2D_map_np_norm, mni_template.affine)
ftools.save_nii_file(NS_2D_map_np_norm,mni_template.header,join(OUTDIR,"nodal_strength_map_weighted_group_2D_SpectralClusteredPCA.nii.gz"))




# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(feature_matrix_2D)
# Apply PCA to reduce to 1 component
pca = PCA(n_components=1)
pca_components = pca.fit_transform(scaled_features)



# ##################### Higher Orders ######################
# M_pp, M_mm, M_pm, M_mp = pltnodal.compute_higher_order_correlations(simmatrix_pop_clean)
# NS_pp_map_np = pltnodal.nodal_strength_map(M_pp,parcellation_data_np,label_indices_group)
# NS_mm_map_np = pltnodal.nodal_strength_map(M_mm,parcellation_data_np,label_indices_group)
# NS_pm_map_np = pltnodal.nodal_strength_map(M_pm,parcellation_data_np,label_indices_group)
# NS_mp_map_np = pltnodal.nodal_strength_map(M_mp,parcellation_data_np,label_indices_group)
# feature_matrix_6D = np.array([M_p, M_n, M_pp, M_mm, M_pm, M_mp]).T  # Example, adjust as per your actual vectors
# feature_matrix_4D = np.array([M_p, M_n, M_pp, M_mm]).T  # Example, adjust as per your actual vectors

# #################################################################
# # Standardize the features
# scaler = StandardScaler()
# scaled_features = scaler.fit_transform(feature_matrix_4D)
# # Apply PCA to reduce to 1 component
# pca = PCA(n_components=1)
# pca_components_6D = pca.fit_transform(scaled_features)
# principal_component_6D = pca_components_6D.flatten()
# NS_6D_map_np = pltnodal.nodal_strength_map(principal_component_6D,parcellation_data_np,label_indices_group)
# NS_6D_map_np_norm = 255*(NS_6D_map_np-np.max(NS_6D_map_np))/(np.min(NS_6D_map_np)-np.max(NS_6D_map_np))
# similarity2D_img = nib.Nifti1Image(NS_6D_map_np_norm, mni_template.affine)
# ftools.save_nii_file(NS_6D_map_np_norm,mni_template.header,join(OUTDIR,"nodal_strength_map_weighted_group_4DPCA.nii.gz"))
# sys.exit()
#################################################################


# The principal components after PCA transformation
principal_component = pca_components.flatten()
NS_2D_map_np = pltnodal.nodal_strength_map(principal_component,parcellation_data_np,label_indices_group)
NS_2D_map_np_norm = 255*(NS_2D_map_np-np.max(NS_2D_map_np))/(np.min(NS_2D_map_np)-np.max(NS_2D_map_np))
similarity2D_img = nib.Nifti1Image(NS_2D_map_np_norm, mni_template.affine)
ftools.save_nii_file(NS_2D_map_np_norm,mni_template.header,join(OUTDIR,"nodal_strength_map_weighted_group_2DPCA.nii.gz"))

# Cluster PCA
num_clusters_arr = np.arange(1,23)  # For example, testing 1 through 10 clusters
wcss = np.zeros(num_clusters_arr.shape)  # List to store the within-cluster sums of squares
for i,n_clusters in enumerate(num_clusters_arr):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=10000, n_init=10, random_state=42)
    kmeans.fit(feature_matrix_2D)
    wcss[i] = kmeans.inertia_
plt.plot(num_clusters_arr,wcss,"-o")
plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=10000, n_init=10, random_state=42)
kmeans.fit(feature_matrix_2D)
homotypic_parcels = np.zeros(principal_component.shape)
for label in kmeans.labels_:
    ids = np.where(kmeans.labels_==label)[0]
    homotypic_parcels[ids] = principal_component[ids].mean()

homotypic_map_np = pltnodal.nodal_strength_map(homotypic_parcels,parcellation_data_np,label_indices_group)
homotypic_map_np_norm = 255*(homotypic_map_np-np.max(homotypic_map_np))/(np.min(homotypic_map_np)-np.max(homotypic_map_np))
homotypic_map_np_norm_img = nib.Nifti1Image(homotypic_map_np_norm, mni_template.affine)
ftools.save_nii_file(homotypic_map_np_norm,mni_template.header,join(OUTDIR,"homotypic_map_map_weighted_group_clustered.nii.gz"))
###################### Plot ######################
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




nodal_strength_np_weighted_hsv = vectorized_rgb_to_gray(nodal_strength_map_np_weighted_plus/np.max(nodal_strength_map_np_weighted_plus)*255,
                                                        nodal_strength_map_np_weighted_neg/np.max(nodal_strength_map_np_weighted_neg)*255)

nodal_strength_np_weighted_scalar = vectorized_to_scalar(nodal_strength_map_np_weighted_plus/np.max(nodal_strength_map_np_weighted_plus), 
                                                         nodal_strength_map_np_weighted_neg/np.max(nodal_strength_map_np_weighted_neg))
ftools.save_nii_file(nodal_strength_np_weighted_scalar,mni_template.header,join(OUTDIR,"nodal_strength_map_weighted_group_scalar.nii.gz"))
ftools.save_nii_file(nodal_strength_np_weighted_scalar,mni_template.header,join(OUTDIR,"nodal_strength_map_weighted_group_scalar.nii.gz"))



# sys.exit()






######### RichClub ########
rc_degrees_M,rc_coefficients_M,mean_rc_M, std_rc_M, rc_deg_cutoff_M,pvalues = nba.get_rc_distribution(np.abs(binarized_matrix_abs))
rich_club_node_indices, rich_club_node_degrees = nba.extract_subnetwork(binarized_matrix_abs,rc_deg_cutoff_M)
_                     , network_node_degrees = nba.extract_subnetwork(binarized_matrix_abs,0)

binarized_matrix_abs[rich_club_node_indices].sum(axis=0)

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


np.savez("rc_network_plot_data.npz",
         simmatrix_pop_clean    = simmatrix_pop_clean,
         binarized_matrix_abs   = binarized_matrix_abs,
         rich_club_node_indices = rich_club_node_indices, 
         rich_club_node_degrees = rich_club_node_degrees,
         network_node_degrees   = network_node_degrees,
         parcel_labels_group    = parcel_labels_group,
         parcel_positions       = parcel_positions 
         )





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








