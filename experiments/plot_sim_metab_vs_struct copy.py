import numpy as np
from tools.datautils import DataUtils
from tools.debug import Debug
from os.path import join, split 
import os , math
import matplotlib.pyplot as plt
from tools.filetools import FileTools
from graphplot.simmatrix import SimMatrixPlot
from bids.mridata import MRIData
from scipy.stats import norm
from math import tan, tanh, atanh
from connectomics.nettools import NetTools
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from sklearn.cluster import KMeans

sel_parcel_list = ["ctx-lh","subc-lh","thal-lh","hipp-lh","stem",
                   "ctx-rh","subc-rh","thal-rh","hipp-rh"]

merge_parcels_dict = dict()


merge_parcels_dict[28]  = {"label":["ctx","rh","caudalmiddlefrontal_12"] ,"merge":[28,29]}
merge_parcels_dict[74]  = {"label":["ctx","rh","pericalcarine"] ,"merge":[74,75]}
merge_parcels_dict[91]  = {"label":["ctx","rh","inferiortemporal_12"] ,"merge":[91,92]}
merge_parcels_dict[111] = {"label":["subc","rh","pallidum_accumbens"] ,"merge":[111,112]}
merge_parcels_dict[120] = {"label":["amygd","rh"],"merge":[120,128]}
merge_parcels_dict[129] = {"label":["hipp","rh"] ,"merge":[129,132]}

merge_parcels_dict[166]  = {"label":["ctx","lh","caudalmiddlefrontal_12"] ,"merge":[166,167]}
merge_parcels_dict[212]  = {"label":["ctx","lh","pericalcarine"] ,"merge":[212,213]}
merge_parcels_dict[229]  = {"label":["ctx","lh","inferiortemporal_12"] ,"merge":[229,230]}
merge_parcels_dict[249]  = {"label":["subc","lh","pallidum_accumbens"] ,"merge":[249,250]}
merge_parcels_dict[258]  = {"label":["amygd","lh"],"merge":[258,266]}
merge_parcels_dict[267]  = {"label":["hipp","lh"] ,"merge":[267,270]}

dutils = DataUtils()
resultdir = join(dutils.ANARESULTSPATH,"connectomes_M_vs_S")
GROUP    = "Mindfulness-Project"
simplt   = SimMatrixPlot()
ftools   = FileTools(GROUP)
debug    = Debug()
nettools = NetTools()

############ Meta stat sign #########3
ALPHA = 0.05
############ Labels ############
subject_id,session = "S001","V1"
mridata  = MRIData(subject_id,session ,group=GROUP)
m_connectome_dir_path = join(mridata.ROOT_PATH,"derivatives","connectomes",
                        f"sub-{subject_id}",f"ses-{session}","spectroscopy")
filename = f"sub-{subject_id}_ses-{session}_run-01_acq-memprage_atlas-chimeraLFMIHIFIF_desc-scale3grow2mm_dseg_simmatrix.npz"
data                 = np.load(join(m_connectome_dir_path,filename))
label_list_concat    = data["labels"]
parcel_labels_ignore = data["parcel_labels_ignore"]
n_labels             = len(label_list_concat)
####################################

data = np.load(join(resultdir,"simM_metab_struct.npz"))
struct_con_arr  = data["struct_con_arr"]
metab_con_arr   = data["metab_con_arr"]
metab_pvalues   = data["metab_pvalues"]
n_subjects      = metab_pvalues.shape[0]
####################################
parcel_ids_positions=dict()
for sel_parcel in sel_parcel_list:
    parcel_id_coord_list=list()
    for idp,parcel_label in enumerate(label_list_concat):
        if sel_parcel in parcel_label:
            parcel_id_coord_list.append(idp)
    a,b = min(parcel_id_coord_list),max(parcel_id_coord_list)+1
    parcel_ids_positions[sel_parcel] = [a,b]



metab_con_bin_merged, metab_con_merged = nettools.construct_metabolic_simmilarity(metab_con_arr,metab_pvalues,ALPHA=0.05,threshold=0.69)
# metab_adj       = (np.abs(metab_sim_corr) > 0.7).astype(int)
fig, axs = plt.subplots(1,3, figsize=(16, 12))  # Adjust size as necessary
for idm in range(3):
    simplt.plot_simmmatrix(metab_con_merged[idm],ax=axs[idm],titles=f"Metabolic Connectome id {idm}",
                           parcel_ids_positions=parcel_ids_positions,colormap="magma") 
plt.tight_layout() 
plt.show() 

################### MErge Demo###################
for ids,simmatrix in enumerate(struct_con_arr):
    merged_matrix = nettools.merge_matrix_elements(simmatrix,merge_parcels_dict)

################################## Optimal thresholding ##################################

for threshold in np.linspace(0.5,0.99,50):
    bin_adj = np.zeros(metab_con_merged.shape)
    # bin_adj[metab_con_merged>threshold]=1
    bin_adj[metab_con_merged<threshold]=0
    spectrum_list = list()
    for im,matrix in enumerate(bin_adj):
        s = nettools.laplacian_spectrum(matrix)
        spectrum_list.append(s)
    debug.info(threshold,"sigma",np.array(spectrum_list).std())

################################## Spectral Clustering ##################################
spectrum_list = list()
for im,matrix in enumerate(metab_con_bin_merged):
    s = nettools.laplacian_spectrum(matrix)
    debug.info(np.sum(matrix))
    spectrum_list.append(s)
plt.show()
spectrum_list = np.array(spectrum_list)



num_clusters = np.arange(1,31)  # For example, testing 1 through 10 clusters
wcss = []  # List to store the within-cluster sums of squares
for k in num_clusters:
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(spectrum_list)
    wcss.append(kmeans.inertia_)
plt.plot(num_clusters, wcss, '-o', color='blue')
plt.show()
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
kmeans.fit(spectrum_list)
# Getting the labels assigned to each instance (i.e., each of your flattened_matrices)
cluster_label = kmeans.labels_
cluster_x = list()
colors = ["r","k","g","b"]
for cluster_id in range(optimal_clusters):
    cluster_idc = np.where(cluster_id==cluster_label)[0]
    cluster_x.append(cluster_idc)
    for idc in cluster_idc:
        plt.plot(spectrum_list[idc],color=colors[cluster_id],label=f"Cluster id {cluster_id}")
plt.show()
##################################################################################################



struct_adj      = (struct_con_arr > 0).astype(int)


# metab_adj       = (np.abs(metab_sim_corr) > 0.7).astype(int)
fig, axs = plt.subplots(1,3, figsize=(16, 12))  # Adjust size as necessary
simplt.plot_simmmatrix(metab_sim_corr,ax=axs[0],titles="Metabolic Connectome",parcel_ids_positions=parcel_ids_positions,colormap="magma") 
simplt.plot_simmmatrix(metab_sim_corr,ax=axs[1],titles="Metabolic Bin Connectome",colormap="magma")   
simplt.plot_simmmatrix(struct_adj.mean(axis=0),ax=axs[2],titles="Structural Connectome",colormap="magma")   
plt.tight_layout()  # Adjust layout to make sure everything fits well
plt.show()


# metab_adj       = (np.abs(metab_bin_arr) > 0.7).astype(int)
metab_adj       = (np.abs(metab_con_merged) > 0.7).astype(int)

K               = struct_adj.shape[0]
# average total metav correlated/uncorrelated
# corr_metab = metab_adj.mean(axis=0)
# count_metab_corr   = len(np.where(corr_metab>0.67)[0])
# count_metab_uncorr = len(np.where(corr_metab<=0.67)[0])



prob_11, prob_10, prob_01, prob_00 = compute_joint_probability_distributions(metab_con_bin_merged,struct_adj)
prob_11, prob_10, prob_01, prob_00 = compute_joint_probability_distributions(metab_adj,struct_adj)

################################## TH=0.75 max prob con ##################################
alpha = 0.05
for threshold in np.linspace(0.5,0.99,50):
    metab_sim_corr = np.tanh(np.mean(np.arctanh(metab_con_merged),axis=0))
    metab_adj       = (np.abs(metab_con_merged) > threshold).astype(int)
    prob_11, prob_10, prob_01, prob_00 = compute_joint_probability_distributions(metab_adj,struct_adj)
    prob_con  = np.zeros(prob_11.shape)
    prob_dys  = np.zeros(prob_11.shape)
    prob_caus = np.zeros(prob_11.shape)
    prob_con[prob_11>prob_10] = 1
    prob_dys[prob_00>prob_01] = 1
    prob_caus[(prob_con==1) & (prob_dys ==1)] = 1
    debug.info(threshold,int(prob_con.sum()),int(prob_dys.sum()))
######################################################################################################

# prob_con = test_binary(prob_11,prob_10,K)
# prob_dys = test_binary(prob_00,prob_01,K)

prob_con  = np.zeros(prob_11.shape)
prob_dys  = np.zeros(prob_11.shape)
prob_caus = np.zeros(prob_11.shape)

prob_con[prob_11>prob_10] = 1
prob_dys[prob_00>prob_01] = 1


alpha = 0.05

# prob_con[prob_con<alpha] = 1
# prob_con[prob_con>=alpha] = 0
# prob_dys[prob_dys<alpha] = 1
# prob_dys[prob_dys>=alpha] = 0


prob_caus[(prob_con==1) & (prob_dys ==0)] = 1

fig, axs = plt.subplots(2,3, figsize=(16, 12))  # Adjust size as necessary
simplt.plot_simmmatrix(prob_11,
                       ax=axs[0,0], 
                       titles="prob M=1|S=1")
simplt.plot_simmmatrix(prob_10,
                       ax=axs[0,1], 
                       titles="prob M=1|S=0")
simplt.plot_simmmatrix(prob_00,
                       ax=axs[1,0], 
                       titles="prob M=0|S=0")
simplt.plot_simmmatrix(prob_01,
                       ax=axs[1,1], 
                       titles="prob M=0|S=1")
simplt.plot_simmmatrix(prob_con,
                       ax=axs[0,2], 
                       titles="prob S=1 -> C=1")
simplt.plot_simmmatrix(prob_dys,
                       ax=axs[1,2], 
                       titles="prob S=0 -> C=0")

plt.tight_layout()  # Adjust layout to make sure everything fits well
plt.show()
laplace_spectrum = ftools.laplacian_spectrum(prob_caus)
plt.plot(laplace_spectrum,label="density")
plt.ylabel("Laplace Spectrum")
plt.legend()
plt.show()





fig, axs = plt.subplots(1,3, figsize=(24, 20))  # Adjust size as necessary

simplt.plot_simmmatrix(prob_caus,
                       parcel_ids_positions=parcel_ids_positions,
                       ax=axs[2], 
                       titles="Causal Structure to Metabolic")
simplt.plot_simmmatrix(struct_adj.mean(axis=0),
                       parcel_ids_positions=parcel_ids_positions,
                       ax=axs[0], 
                       titles="Structrual Connectome")
simplt.plot_simmmatrix(np.tanh(np.mean(np.arctanh(metab_con_merged),axis=0)),
                       parcel_ids_positions=parcel_ids_positions,
                       ax=axs[1], 
                       titles="Metabolic Connectome")                                  

plt.tight_layout()  # Adjust layout to make sure everything fits well
plt.show()




