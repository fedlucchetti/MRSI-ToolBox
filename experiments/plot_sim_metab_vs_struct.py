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
import networkx as nx

from math import tan, tanh, atanh
from connectomics.nettools import NetTools
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from sklearn.cluster import KMeans
import copy, sys
from connectomics.network import NetBasedAnalysis
from sklearn.linear_model import RANSACRegressor
from connectomics.parcellate import Parcellate

from pyemd import emd


dutils = DataUtils()
resultdir = join(dutils.ANARESULTSPATH,"connectomes_M_vs_S")
GROUP    = "Mindfulness-Project"
simplt   = SimMatrixPlot()
ftools   = FileTools(GROUP)
debug    = Debug()
nettools = NetTools()
nba      = NetBasedAnalysis()
parc     = Parcellate()

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
####################################

data = np.load(join(resultdir,"simM_metab_struct.npz"))
struct_con_arr  = data["struct_con_arr"]
metab_con_arr   = data["metab_con_arr"]
metab_pvalues   = data["metab_pvalues"]
subject_id_arr  = data["subject_id_arr"]
session_arr     = data["session_arr"]
# simmatrix_ids_to_delete_arr = data["simmatrix_ids_to_delete_arr"]

outpath            = join(dutils.ANARESULTSPATH,f"{GROUP}_threshold_list.npz")
data               = np.load(outpath)
subject_id_arr_th  = data["subject_id_arr"]
session_arr_th     = data["session_arr"]
threshold_arr      = data["best_thresholds"]



n_subjects      = metab_pvalues.shape[0]
sel_parcel_list = ["ctx-rh","subc-rh","thal-rh","amygd-rh","hipp-rh",
                   "ctx-lh","subc-lh","thal-lh","amygd-lh","hipp-lh","stem"]
parcel_ids_positions, label_list_concat = parc.get_main_parcel_plot_positions(sel_parcel_list,label_list_concat)

def compute_laplacian_spectrum(matrix):
    G = nx.from_numpy_array(matrix)
    L = nx.laplacian_matrix(G).toarray()
    eigenvalues = np.linalg.eigvalsh(L)
    return eigenvalues

def get_degree_distribution(simmatrix_binarized):
    degree_distribution = nba.degree_distribution(simmatrix_binarized)
    degrees             = np.array(list(degree_distribution.keys()))
    ids                 = np.argsort(degrees)
    degree_counts       = np.array(list(degree_distribution.values()))
    degrees,degree_counts = degrees[ids[1::]],degree_counts[ids[1::]]
    ransac = RANSACRegressor()
    ransac.fit(degrees.reshape(-1, 1), np.log(degree_counts))
    X_fit        = np.linspace(degrees.min(), degrees.max(), 100).reshape(-1, 1)
    y_pred_huber = ransac.predict(X_fit)
    return degrees,degree_counts,X_fit,y_pred_huber

def get_clustering_distribution(simmatrix_binarized):
    G = nx.from_numpy_array(simmatrix_binarized)
    # Compute the clustering coefficient for each node
    return nx.clustering(G).values()

def get_betweeness_distribution(simmatrix_binarized):
    G = nx.from_numpy_array(simmatrix_binarized)
    # Compute the clustering coefficient for each node
    return nx.betweenness_centrality(G).values()

def get_rc_distribution(simmatrix_binarized):
    ######### RichClub ########
    adjacency_matrix = np.where(simmatrix_binarized == -1, 1, simmatrix_binarized)
    np.fill_diagonal(adjacency_matrix,0)
    G = nx.from_numpy_array(adjacency_matrix)
    reference_degrees = np.array(sorted(set(d for n, d in G.degree())))
    # Compute rich-club coefficient distribution for Metabolic network
    rc_coefficients = nba.rich_club_coefficient_curve(G, reference_degrees)
    # Compute rich-club coefficient distribution for random network
    mean_rc, std_rc = nba.rich_club_distribution(G, reference_degrees)
    degree_at_rc_1 = None
    for degree, rc_coeff in zip(reference_degrees, rc_coefficients):
        if rc_coeff >= 1:
            degree_at_rc_1 = degree
            break
    return reference_degrees,rc_coefficients,mean_rc, std_rc, degree_at_rc_1

def extract_rich_club_submatrix(simmatrix_binarized,degree_at_rc_1):
    # _, _, _, _, degree_at_rc_1 = get_rc_distribution(simmatrix_binarized)
    rich_club_submatrix = np.zeros(simmatrix_binarized.shape)
    if degree_at_rc_1 is not None:
        adjacency_matrix = np.where(simmatrix_binarized == -1, 1, simmatrix_binarized)
        np.fill_diagonal(adjacency_matrix, 0)
        G = nx.from_numpy_array(adjacency_matrix)
        # Identify rich-club nodes
        rich_club_nodes = [n for n, d in G.degree() if d >= degree_at_rc_1]
        # Create submatrix for rich-club nodes
        rich_club_indices = np.array(rich_club_nodes)
        rich_club_submatrix = simmatrix_binarized[np.ix_(rich_club_indices, rich_club_indices)]
    return rich_club_submatrix


def compute_emd(distribution1, distribution2):
    """
    Computes the Earth Mover's Distance (EMD) between two distributions.
    
    Parameters:
    distribution1 (np.array): First distribution (histogram).
    distribution2 (np.array): Second distribution (histogram).
    
    Returns:
    float: The Earth Mover's Distance between the two distributions.
    """
    
    # Ensure the distributions are normalized
    distribution1 = distribution1 / np.sum(distribution1)
    distribution2 = distribution2 / np.sum(distribution2)
    
    # Number of bins
    num_bins = len(distribution1)
    
    # Compute the distance matrix
    distance_matrix = np.zeros((num_bins, num_bins))
    for i in range(num_bins):
        for j in range(num_bins):
            distance_matrix[i, j] = abs(i - j)
    # Compute the EMD
    emd_value = emd(distribution1, distribution2, distance_matrix)
    
    return emd_value

th = 0.73
FONTSIZE = 16
for idm,metab_sim in enumerate(metab_con_arr):

    idt = np.where((subject_id_arr[idm] == subject_id_arr_th) & (session_arr[idm] == session_arr_th))[0]
    if len(idt)==0: continue
    subject_id,session     = subject_id_arr_th[idt][0],session_arr_th[idt][0]
    prefix                 = f"sub-{subject_id}_ses-{session}"
    debug.title(f"Generating figures for {prefix}")
    m_connectome_dir_path = join(mridata.ROOT_PATH,"derivatives","connectomes",
                            f"sub-{subject_id}",f"ses-{session}","spectroscopy")
    metab_filepath          = join(m_connectome_dir_path,f"{prefix}_run-01_acq-memprage_atlas-chimeraLFMIHIFIF_desc-scale3grow2mm_dseg_simmatrix.npz")
    simmatrix_ids_to_delete = np.load(join(m_connectome_dir_path,metab_filepath))["simmatrix_ids_to_delete"]
    threshold               = threshold_arr[idt][0]
    
    resultdir_subject_path = join(resultdir,prefix)
    os.makedirs(resultdir_subject_path,exist_ok=True)
                                  
    struct_con          = struct_con_arr[idm]
    metabmatrix_binarized = copy.deepcopy(metab_sim)
    structmatrix_binarized = copy.deepcopy(struct_con)
    metabmatrix_binarized[np.abs(metab_sim)<threshold]   = 0
    metabmatrix_binarized[np.abs(metab_sim)>=threshold]  = np.sign(metabmatrix_binarized[np.abs(metab_sim)>=threshold])
    structmatrix_binarized[struct_con>0]  = 1


    ## 1 Connectivty Matrix
    debug.info("1 Connectivty Matrix")
    fig, axs = plt.subplots(1,2, figsize=(16, 12))  # Adjust size as necessary
    outpath = join(resultdir_subject_path,f"{prefix}-connectivty_matrix")
    _structmatrix_binarized = copy.deepcopy(structmatrix_binarized)
    _structmatrix_binarized[0,0] = -1
    simplt.plot_simmmatrix(_structmatrix_binarized,ax=axs[0],titles=f"Structural Connectome",
                        parcel_ids_positions=parcel_ids_positions,colormap="seismic") 
    simplt.plot_simmmatrix(metabmatrix_binarized,ax=axs[1],titles=f"Metabolic Connectome",
                        parcel_ids_positions=parcel_ids_positions,colormap="seismic",
                        result_path=outpath,show_parcels="H")
    plt.show() 

    ## 2 Degree, Cluster and InBetweeness
    debug.info("2. Degree, Cluster and InBetweeness")
    arr = copy.deepcopy(metabmatrix_binarized)
    array_after_row_deletion = np.delete(arr, simmatrix_ids_to_delete, axis=0)
    metabmatrix_binarized    = np.delete(array_after_row_deletion, simmatrix_ids_to_delete, axis=1)

    fig, axs = plt.subplots(2,2, figsize=(16, 12))  # Adjust size as necessary

    degrees_S,degree_counts_S,X_fit_S,y_pred_huber_S = get_degree_distribution(structmatrix_binarized)
    degrees_M,degree_counts_M,X_fit_M,y_pred_huber_M = get_degree_distribution(metabmatrix_binarized)

    cluster_distr_S = get_clustering_distribution(structmatrix_binarized)
    cluster_distr_M = get_clustering_distribution(metabmatrix_binarized)

    btw_distr_S = get_betweeness_distribution(structmatrix_binarized)
    btw_distr_M = get_betweeness_distribution(metabmatrix_binarized)
  

    # debug.info(degree_counts_S.shape, degree_counts_M.shape)
    # debug.info("Delta Degree Distribution     ",compute_emd(degree_counts_S, degree_counts_M))
    # debug.info("Delta ClusterCoef Distribution",compute_emd(cluster_distr_S, cluster_distr_M))
    # debug.info("Delta Betweenness Distribution",compute_emd(btw_distr_S, btw_distr_M))


    axs[0,0].plot(degrees_S, degree_counts_S,".", color='r', alpha=0.7)
    axs[0,0].plot(X_fit_S, np.exp(y_pred_huber_S), color='red', label='Structural')
    axs[0,0].set_xlabel('Degree',fontsize=FONTSIZE)
    axs[0,0].set_ylabel('Counts',fontsize=FONTSIZE)
    axs[0,0].set_yscale('log')
    axs[0,0].legend()
    axs[0,0].set_ylim(1,100)
    axs[0,0].grid()


    axs[0,1].plot(degrees_M, degree_counts_M,".", color='b', alpha=0.7)
    axs[0,1].plot(X_fit_M, np.exp(y_pred_huber_M), color='blue', label='Metabolic')
    axs[0,1].set_xlabel('Degree',fontsize=FONTSIZE)
    axs[0,1].set_ylabel('Counts',fontsize=FONTSIZE)
    axs[0,1].set_yscale('log')
    axs[0,1].legend()
    axs[0,1].set_ylim(1,100)
    axs[0,1].grid()

    axs[1,0].hist(cluster_distr_S, color='r', label='Structural', alpha=0.7)
    axs[1,0].hist(cluster_distr_M, color='b', label='Metabolic', alpha=0.7)
    axs[1,0].set_xlabel('Cluster Coefficient',fontsize=FONTSIZE)
    axs[1,0].set_ylabel('Counts',fontsize=FONTSIZE)
    # axs[0,0].set_yscale('log')
    axs[1,0].legend()
    # axs[1,0].set_ylim(1,100)
    axs[1,0].grid()

    axs[1,1].hist(btw_distr_S, color='r', label='Structural', alpha=0.7)
    axs[1,1].hist(btw_distr_M, color='b', label='Metabolic', alpha=0.7)
    axs[1,1].set_xlabel('In-Betweenness',fontsize=FONTSIZE)
    axs[1,1].set_ylabel('Counts',fontsize=FONTSIZE)
    # axs[1,1].set_yscale('log')
    # axs[1,1].set_ylim(1,100)
    axs[1,1].legend()
    axs[1,1].grid()
    plt.tight_layout()
    # plt.show()
    outpath = join(resultdir_subject_path,f"{prefix}-topological")
    fig.savefig(f"{outpath}.pdf")
    debug.success("Saved to ",outpath) 

    ## 3 Rich Club analysis
    debug.info("3. Rich Club analysis")
    rc_degrees_S,rc_coefficients_S,mean_rc_S, std_rc_S, rc_deg_cutoff_S = get_rc_distribution(structmatrix_binarized)
    rc_degrees_M,rc_coefficients_M,mean_rc_M, std_rc_M, rc_deg_cutoff_M = get_rc_distribution(np.abs(metabmatrix_binarized))
    structmatrix_binarized_rc = extract_rich_club_submatrix(structmatrix_binarized,rc_deg_cutoff_S)
    metabmatrix_binarized_rc  = extract_rich_club_submatrix(np.abs(metabmatrix_binarized),rc_deg_cutoff_M)

    bin_threshold_arr = np.linspace(0.6,0.85,100)
    deltas = np.ones(bin_threshold_arr.shape)
    for idt,bin_threshold in enumerate(bin_threshold_arr):
        metabmatrix_binarized = copy.deepcopy(metab_sim)
        metabmatrix_binarized[np.abs(metab_sim)<bin_threshold]   = 0
        metabmatrix_binarized[np.abs(metab_sim)>=bin_threshold]  =  np.sign(metabmatrix_binarized[np.abs(metab_sim)>=bin_threshold])
        spectrum_S    = compute_laplacian_spectrum(structmatrix_binarized)
        spectrum_M    = compute_laplacian_spectrum(np.abs(metabmatrix_binarized))
        deltas[idt]   = np.abs(np.mean(spectrum_M-spectrum_S))
        debug.info(bin_threshold,deltas[idt])

    debug.success("Best threshold",bin_threshold_arr[np.argmin(deltas)])
    bin_threshold = bin_threshold_arr[np.argmin(deltas)]
    metabmatrix_binarized = copy.deepcopy(metab_sim)
    metabmatrix_binarized[np.abs(metab_sim)<bin_threshold]   = 0
    metabmatrix_binarized[np.abs(metab_sim)>=bin_threshold]  =  np.sign(metabmatrix_binarized[np.abs(metab_sim)>=bin_threshold])
    spectrum_S_rc = compute_laplacian_spectrum(structmatrix_binarized_rc)
    spectrum_M_rc = compute_laplacian_spectrum(metabmatrix_binarized_rc)
 

    fig, axs = plt.subplots(2,2, figsize=(16, 12))  # Adjust size as necessary 
    axs[0,0].plot(rc_degrees_S, rc_coefficients_S, label='Structural Connectome RC', color='red')
    axs[0,0].fill_between(rc_degrees_S, mean_rc_S - std_rc_S, mean_rc_S + std_rc_S, color='gray', alpha=0.5, label='Random Network ±1σ')
    axs[0,0].vlines(rc_deg_cutoff_S, 0, 1, colors='k', linestyles='dashed')
    axs[0,0].set_xlabel('Degree',fontsize=FONTSIZE)
    axs[0,0].set_ylabel('Rich-Club Coefficient',fontsize=FONTSIZE)
    axs[0,0].legend()
    axs[0,0].grid()

    axs[0,1].plot(rc_degrees_M, rc_coefficients_M, label='Metabolic Connectome RC', color='blue')
    axs[0,1].fill_between(rc_degrees_M, mean_rc_M - std_rc_M, mean_rc_M + std_rc_M, color='gray', alpha=0.5, label='Random Network ±1σ')
    axs[0,1].vlines(rc_deg_cutoff_M, 0, 1, colors='k', linestyles='dashed')
    axs[0,1].set_xlabel('Degree',fontsize=FONTSIZE)
    axs[0,1].set_ylabel('Rich-Club Coefficient',fontsize=FONTSIZE)
    axs[0,1].legend()
    axs[0,1].grid()

    axs[1,0].plot(spectrum_S, label='Structural Network', color='red')
    axs[1,0].plot(spectrum_M, label='Metabolic Network', color='blue')
    axs[1,0].set_xlabel('Eigenvalue ID',fontsize=FONTSIZE)
    axs[1,0].set_ylabel('Spectrum',fontsize=FONTSIZE)
    axs[1,0].legend()
    axs[1,0].grid()

    axs[1,1].plot(spectrum_S_rc, label='Structural RC Network', color='red')
    axs[1,1].plot(spectrum_M_rc, label='Metabolic RC Network', color='blue')
    axs[1,1].set_xlabel('Eigenvalue ID',fontsize=FONTSIZE)
    axs[1,1].set_ylabel('Spectrum',fontsize=FONTSIZE)
    axs[1,1].legend()
    axs[1,1].grid()
    # plt.show()
    outpath = join(resultdir_subject_path,f"{prefix}-topological_RC")
    plt.tight_layout()
    # plt.show()
    fig.savefig(f"{outpath}.pdf")
    debug.success("Saved to ",outpath) 
    debug.separator()

sys.exit()

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




