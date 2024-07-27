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
from connectomics.simmilarity import Simmilarity
from graphplot.nodal_simmilarity import NodalSimilarity
from rich.console import Console
from rich.table import Table
from scipy.stats import linregress, spearmanr
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
dutils    = DataUtils()
resultdir = join(dutils.ANARESULTSPATH,"connectomes_M_vs_S")
OUTDIR    = join(dutils.ANARESULTSPATH,"PLOS")
os.makedirs(OUTDIR,exist_ok=True)

GROUP     = "Mindfulness-Project"
simm      = Simmilarity()
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
ALPHA            = 0.05
# PARC_SCHEME      = "aal"
PARC_SCHEME      = "LFMIHIFIF-2"
# PARC_SCHEME      = "geometric_cubeK23mm"

INCLUDE_WM       = True
############ Labels ############

####################################

METABOLITES     = ["NAANAAG", "Ins", "GPCPCh", "GluGln", "CrPCr"]
FONTSIZE        = 16



############# Load 4D Homotopy features
resultssubdir = join(OUTDIR,"simmatrix",GROUP)
os.makedirs(resultssubdir,exist_ok=True)
filename      = f"group-{GROUP}_atlas-{PARC_SCHEME}_desc-simmatrix_WM_{int(INCLUDE_WM)}.npz"
outpath       = join(resultssubdir,filename)
data          = np.load(outpath)
# Extract variables
parcel_concentrations5D = data['parcel_concentrations5D']
weighted_metab_sim_list = data['weighted_metab_sim']
weighted_metab_sim_avg  = data['weighted_metab_sim_avg']
label_indices_group     = data['label_indices_group']
parcel_labels_group     = data['parcel_labels_group']
subject_id_arr          = data['subject_id_arr']
session_arr             = data['session_arr']
n_recordings            = len(session_arr)

weighted_metab_sim_list_1 = weighted_metab_sim_list[:,:,:,0]
weighted_metab_sim_list_2 = weighted_metab_sim_list[:,:,:,1]


############## GET MNI Parcellation ###############
mni_template    = datasets.load_mni152_template()
mridata  = MRIData("S002","V3",group=GROUP)
parcel_t1w_path = mridata.data["parcels"][PARC_SCHEME]["orig"]["path"]
transform_list  = mridata.get_transform("forward","anat")
parcel_mni_img  = reg.transform(fixed_image=ants.from_nibabel(mni_template),moving_image=parcel_t1w_path,
                                interpolator_mode="genericLabel",transform=transform_list)
parcel_mni_img_nii   = nib.Nifti1Image(parcel_mni_img.numpy(), mni_template.affine)
parcellation_data_np = parcel_mni_img_nii.get_fdata()
ftools.save_nii_file(parcellation_data_np,mni_template.header,join(OUTDIR,f"parcellation_mi152_{PARC_SCHEME}.nii.gz"))
mnimask = datasets.load_mni152_brain_mask().get_fdata()

############## Feature 4D ##############
features4D    = simm.get_4D_feature_nodal_similarity(weighted_metab_sim_avg,parcellation_data_np,label_indices_group)
features2D_1  = simm.get_feature_nodal_similarity(weighted_metab_sim_avg[:,:,0],parcellation_data_np,label_indices_group)
features2D_2  = simm.get_feature_nodal_similarity(weighted_metab_sim_avg[:,:,1],parcellation_data_np,label_indices_group)

features4D_subj_list = np.zeros((weighted_metab_sim_list_1.shape[0],)+(features2D_2.shape[0],)+(4,))
for i,_weighted_metab_sim_1 in enumerate(weighted_metab_sim_list_1):
    subject_id,session = subject_id_arr[i],session_arr[i]
    _weighted_metab_sim_2  = weighted_metab_sim_list_2[i]
    features_1             = simm.get_feature_nodal_similarity(_weighted_metab_sim_1,parcellation_data_np,label_indices_group)
    features_2             = simm.get_feature_nodal_similarity(_weighted_metab_sim_2,parcellation_data_np,label_indices_group)
    features4D_subj_list[i,:,0:2]    = features_1
    features4D_subj_list[i,:,2:4]    = features_2
features4D_subj_avg = features4D_subj_list.mean(axis=0)


########## Dimensionality Reduction ##########
methods = ['isomap', 'lle', 'hessian_lle', 'laplacian_eigenmaps', 'tsne', 'umap', 'pca', 'lda', 'kernel_pca', 'sparse_pca']
for method in methods:
    debug.info(method)
    try:
        projected_data = netclust.project_4d_to_1d(features4D, method)
        projected_data = 255*(projected_data-np.min(projected_data))/(np.max(projected_data)-np.min(projected_data))
        projected_data_3D = simm.nodal_strength_map(projected_data[:,0],parcellation_data_np,label_indices_group)
        projected_data_3D[mnimask<1] = 0
        outpath = join(OUTDIR,f"homotopy_group_4D_{PARC_SCHEME}-WM_{INCLUDE_WM}-proj_{method}.nii.gz")
        ftools.save_nii_file(projected_data_3D,mni_template.header,outpath)
    except Exception as e: debug.warning(method,"SKIP")





########## Homotypic CLuster Detection ##########
outsubdir = join(OUTDIR,"homotypic_clusters",PARC_SCHEME)
os.makedirs(outsubdir,exist_ok=True)
n_clusters_list= [3,4]
for n_clusters in n_clusters_list:
    cluster_labels = netclust.cluster_all_algorithms(features4D,n_clusters=n_clusters)
    for clust_alg in cluster_labels.keys():
        projected_data_3D = simm.nodal_strength_map(cluster_labels[clust_alg]+1,parcellation_data_np,label_indices_group)
        outpath = join(outsubdir,f"homotopy_group_4D_{PARC_SCHEME}-WM_{INCLUDE_WM}-NR_{n_recordings}-clust_{clust_alg}-nclust_{n_clusters}.nii.gz")
        ftools.save_nii_file(projected_data_3D,mni_template.header,outpath)
    cluster_labels = netclust.cluster_all_algorithms(features2D_1,n_clusters=n_clusters)
    for clust_alg in cluster_labels.keys():
        debug.info(clust_alg)
        projected_data_3D = simm.nodal_strength_map(cluster_labels[clust_alg]+1,parcellation_data_np,label_indices_group)
        outpath = join(outsubdir,f"homotopy_group_2D_{PARC_SCHEME}-WM_{INCLUDE_WM}-NR_{n_recordings}-clust_{clust_alg}-nclust_{n_clusters}.nii.gz")
        ftools.save_nii_file(projected_data_3D,mni_template.header,outpath)
    cluster_labels = netclust.cluster_all_algorithms(features4D_bin,n_clusters=n_clusters)
    for clust_alg in cluster_labels.keys():
        debug.info(clust_alg)
        projected_data_3D = simm.nodal_strength_map(cluster_labels[clust_alg]+1,parcellation_data_np,label_indices_group)
        outpath = join(outsubdir,f"homotopy_network_group_4D_{PARC_SCHEME}-WM_{INCLUDE_WM}-NR_{n_recordings}-clust_{clust_alg}-nclust_{n_clusters}.nii.gz")
        ftools.save_nii_file(projected_data_3D,mni_template.header,outpath)
    cluster_labels = netclust.cluster_all_algorithms(features4D_subj_avg,n_clusters=n_clusters)
    for clust_alg in cluster_labels.keys():
        debug.info(clust_alg)
        projected_data_3D = simm.nodal_strength_map(cluster_labels[clust_alg]+1,parcellation_data_np,label_indices_group)
        outpath = join(outsubdir,f"homotopy_network_subjavg_4D_{PARC_SCHEME}-WM_{INCLUDE_WM}-NR_{n_recordings}-clust_{clust_alg}-nclust_{n_clusters}.nii.gz")
        ftools.save_nii_file(projected_data_3D,mni_template.header,outpath)










sys.exit()



########### Show Cluster pairwise correlation ################
cluster_labels = netclust.cluster_all_algorithms(features4D)["KMeans"]
n_clusters = max(cluster_labels)+1
parcel_concentrations4D = parcel_concentrations5D.mean(axis=0)
parcel_concentrations3D = np.zeros((n_clusters,)+parcel_concentrations4D.shape[1::])
for cluster_label in range(max(cluster_labels)+1):
    ids = np.where(cluster_labels==cluster_label)[0]
    parcel_concentrations3D[cluster_label] = np.mean(parcel_concentrations4D[ids],axis=0)


def plot_cluster_correlations(parcel_concentrations3D,cluster_x_id,cluster_y_id,axs,plot_met_legend=True):
    x_array = parcel_concentrations3D[cluster_x_id].flatten()
    y_array = parcel_concentrations3D[cluster_y_id].flatten()
    for i_met,met in enumerate(METABOLITES):
        parcel_conc_x = parcel_concentrations3D[cluster_x_id,i_met]
        parcel_conc_y = parcel_concentrations3D[cluster_y_id,i_met]
        if plot_met_legend:
            axs.plot(parcel_conc_x,parcel_conc_y,".",label=f"{met}")
        else:
            axs.plot(parcel_conc_x,parcel_conc_y,".")
    slope, intercept, r, p, se = linregress(x_array, y_array)
    res = spearmanr(x_array, y_array)
    res2 = parc.speanman_corr_quadratic(x_array, y_array)
    corr2 = round(res2["corr"],2)
    x_range = np.linspace(x_array.min(),x_array.max(),100)
    y1_pred = intercept + slope * x_range
    y1_extra = intercept + slope * x_array
    # Quadratic
    X = x_array.reshape(-1, 1)
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model2 = LinearRegression()
    model2.fit(X_poly, y_array)
    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    X_range_poly = poly.transform(X_range)
    Y_range_pred = model2.predict(X_range_poly)
    # Plot regression
    axs.plot(x_range, y1_pred, 'k', linewidth=2, 
            label=fr"$\rho_{{XY}} = {round(res.statistic, 2)}$")
    axs.plot(X_range, Y_range_pred,'r', linewidth=1, 
             label=fr"$\rho^2_{{XY}} = {corr2}$")
    axs.legend(fontsize=FONTSIZE)
    axs.grid()
    return axs

fig, axs = plt.subplots(1,3, figsize=(16, 12))  # Adjust size as necessary
plot_cluster_correlations(parcel_concentrations3D,0,1,axs[0],plot_met_legend=True)
plot_cluster_correlations(parcel_concentrations3D,0,2,axs[1],plot_met_legend=False)
plot_cluster_correlations(parcel_concentrations3D,1,2,axs[2],plot_met_legend=False)
plt.show()

sys.exit()






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








