import numpy as np
from tools.datautils import DataUtils
from tools.debug import Debug
from os.path import join, split 
import os , math
import matplotlib.pyplot as plt
from tools.filetools import FileTools
from graphplot.simmatrix import SimMatrix
from bids.mridata import MRIData
from scipy.stats import norm
import networkx as nx
from rich.progress import Progress,track
from bids.mridata import MRIData
import nibabel as nib
from math import tan, tanh, atanh
from connectomics.nettools import NetTools
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from sklearn.cluster import KMeans
import copy, sys
from connectomics.network import NetBasedAnalysis
from connectomics.parcellate import Parcellate
from pyemd import emd
from scipy.stats import chi2_contingency

dutils = DataUtils()
resultdir = join(dutils.ANARESULTSPATH,"connectomes_M_vs_S")
GROUP    = "Mindfulness-Project"
simplt   = SimMatrix()
ftools   = FileTools(GROUP)
debug    = Debug()
nettools = NetTools()
nba      = NetBasedAnalysis()
parc     = Parcellate()

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
label_list_concat    = data["labels"]
label_indices        = data["labels_indices"]

####################################

data = np.load(join(resultdir,"simM_metab_struct.npz"))
struct_con_arr  = data["density_con_arr"]
length_con_arr  = data["length_con_arr"]
# length_con_arr  = data["distance_matrix_arr"]
metab_con_arr   = data["metab_con_sp_arr"]
metab_pvalues   = data["metab_pvalues_sp"]
subject_id_arr  = data["subject_id_arr"]
session_arr     = data["session_arr"]
qmask_arr       = data["qmask_arr"]
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




FONTSIZE = 16
S_matrices,M_matrices         = list(), list()
M_matrices_pos,M_matrices_neg = list(), list()
M_matrices_ful = list()
length_arr = list()
weighted_metab_sim = list()
for idm,metab_sim in enumerate(metab_con_arr):
    idt = np.where((subject_id_arr[idm] == subject_id_arr_th) & (session_arr[idm] == session_arr_th))[0]
    if len(idt)==0: continue
    subject_id,session        = subject_id_arr_th[idt][0],session_arr_th[idt][0]
    prefix                    = f"sub-{subject_id}_ses-{session}"
    m_connectome_dir_path     = join(mridata.ROOT_PATH,"derivatives","connectomes",
                                f"sub-{subject_id}",f"ses-{session}","spectroscopy")
    metab_filepath            = join(m_connectome_dir_path,f"{prefix}_run-01_acq-memprage_atlas-chimeraLFMIHIFIF_desc-scale3grow2mm_dseg_simmatrix.npz")
    simmatrix_ids_to_delete   = np.load(join(m_connectome_dir_path,metab_filepath))["simmatrix_ids_to_delete"]
    threshold                 = threshold_arr[idt][0]
    struct_con                = struct_con_arr[idm]
    length_con                = length_con_arr[idm]
    length_con[length_con>MAX_FIBRE_LENGTH] = -1
    metabmatrix_binarized     = nba.binarize(metab_sim,threshold,mode="abs")
    metabmatrix_binarized_ful = nba.binarize(metab_sim,threshold,mode="posneg")
    metabmatrix_binarized_pos = nba.binarize(metab_sim,threshold,mode="pos")
    metabmatrix_binarized_neg = nba.binarize(metab_sim,threshold,mode="neg")
    structmatrix_binarized    = copy.deepcopy(struct_con)
    structmatrix_binarized[struct_con>0]  = 1
    weighted_metab_sim.append(metab_sim)
    S_matrices.append(structmatrix_binarized)
    M_matrices.append(metabmatrix_binarized)
    M_matrices_pos.append(metabmatrix_binarized_pos)
    M_matrices_neg.append(metabmatrix_binarized_neg)
    M_matrices_ful.append(metabmatrix_binarized_ful)
    length_arr.append(length_con)


M_matrices = np.array(M_matrices)
S_matrices = np.array(S_matrices)
M_matrices_pos = np.array(M_matrices_pos)
M_matrices_neg = np.array(M_matrices_neg)
M_matrices_ful = np.array(M_matrices_ful)
weighted_metab_sim = np.array(weighted_metab_sim)
struct_con_arr       = np.array(struct_con_arr)
length_arr           = np.array(length_arr)
length_arr[S_matrices!=1] = -1
# length_matrix        = np.nan_to_num(length_arr/(np.sqrt(length_arr*S_matrices)), nan=0.0)


############ Degree vs Distances ############
distances_all,X_meas,X_fit,y_pred_huber,params = nba.edge_distance(M_matrices[0],length_arr[0])
distance_pdf,_    = np.histogram(distances_all,bins="auto")
slope = params["slope"]
r2    = params["r_squared"]
distances_all,X_meas_p,X_fit_p,y_pred_huber_p,params_p = nba.edge_distance(M_matrices_pos[0],length_arr[0])
distance_pdf_p,_    = np.histogram(distances_all,bins="auto")
slope = params_p["slope"]
r2    = params_p["r_squared"]
distances_all,X_meas_n,X_fit_n,y_pred_huber_n,params_n = nba.edge_distance(M_matrices_neg[0],length_arr[0])
distance_pdf_n,_    = np.histogram(distances_all,bins="auto")
slope = params_n["slope"]
r2    = params_n["r_squared"]
plt.figure(figsize=(10, 6))
plt.plot(X_fit,np.exp(y_pred_huber),"k",label=f"slope:{round(slope,1)} -- R2:{round(r2,2)}")
plt.plot(X_fit_p,np.exp(y_pred_huber_p),"r-",label=f"slope:{round(slope,1)} -- R2:{round(r2,2)}")
plt.plot(X_fit_n,np.exp(y_pred_huber_n),"b-",label=f"slope:{round(slope,1)} -- R2:{round(r2,2)}")
plt.plot(X_meas,distance_pdf,"k.")
plt.plot(X_meas_p,distance_pdf_p,"r.")
plt.plot(X_meas_n,distance_pdf_n,"b.")

plt.yscale("log")
plt.show()
############ Degree vs Distances ############

all_differences_p = list()
all_differences_n = list()
all_differences   = list()
for k in range(M_matrices_ful.shape[0]):
    indices_all = np.argwhere(M_matrices[k] == 1)
    indices_p   = np.argwhere(M_matrices_pos[k] == 1)
    indices_n   = np.argwhere(M_matrices_neg[k] == 1)
    # Calculate the differences of their indices
    # distances_all = indices_all[:, 0]-indices_all[:, 1]
    # distances_p   = indices_p[:, 0]-indices_p[:, 1]
    # distances_n   = indices_n[:, 0]-indices_n[:, 1]
    distances_all = length_arr[k,indices_all[:, 0],indices_all[:, 1]]
    distances_p   = length_arr[k,indices_p[:, 0],indices_p[:, 1]]
    distances_n   = length_arr[k,indices_n[:, 0],indices_n[:, 1]]
    all_differences.extend(distances_all[distances_all!=-1])   
    all_differences_p.extend(distances_p[distances_p!=-1])
    all_differences_n.extend(distances_n[distances_n!=-1])
# delete NaN values from non connected edges
all_differences   = np.array(all_differences)
all_differences   = all_differences[~np.isnan(all_differences)]
all_differences_p = np.array(all_differences_p)
all_differences_p = all_differences_p[~np.isnan(all_differences_p)]
all_differences_n = np.array(all_differences_n)
all_differences_n = all_differences_n[~np.isnan(all_differences_n)]



# Calculate the differences of their indices
distance_pdf,bins    = np.histogram(all_differences,bins="auto")
distances = (bins[:-1] + bins[1:]) / 2
X_fit,y_pred_huber,params = lin_regression(distances,np.log(distance_pdf))
slope = params["slope"]
r2    = params["r_squared"]
# Calculate the differences of their + indices
distance_pdf_p,bins = np.histogram(all_differences_p,bins="auto")
distances_p = (bins[:-1] + bins[1:]) / 2
X_fit_p,y_pred_huber_p,params_p = lin_regression(distances_p,np.log(distance_pdf_p))
slope_p = params_p["slope"]
r2_p    = params_p["r_squared"]
# Calculate the differences of their - indices
distance_pdf_n,bins = np.histogram(all_differences_n,bins="auto")
distances_n = (bins[:-1] + bins[1:]) / 2
X_fit_n,y_pred_huber_n,params_n = lin_regression(distances_n,np.log(distance_pdf_n))
slope_n = params_n["slope"]
r2_n    = params_n["r_squared"]

plt.figure(figsize=(10, 6))
plt.plot(X_fit,np.exp(y_pred_huber),"k",label=f"slope:{round(slope,1)} -- R2:{round(r2,2)}")
plt.plot(distances,distance_pdf,"k")
# plt.plot(X_fit_p,np.exp(y_pred_huber_p),"r",label=f"slope:{round(slope_p,1)} -- R2:{round(r2_p,2)}")
plt.plot(distances_p,distance_pdf_p,"r")
# plt.plot(X_fit_n,np.exp(y_pred_huber_n),"b",label=f"slope:{round(slope_n,1)} -- R2:{round(r2_n,2)}")
plt.plot(distances_n,distance_pdf_n,"b")
# plt.hist(np.array(all_differences_n),color="blue",alpha=0.23,bins=125,label="positive edges")
# plt.hist(np.array(all_differences_p),color="red",alpha=0.2,bins=125,label="negative edges")
# plt.hist(np.array(all_differences),color="black",alpha=0.2,bins=125)

plt.title('Degree PDF vs Distance')
plt.xlabel('Distance')
# plt.xlim([0,170])
plt.ylabel('PDF')
# plt.yscale("log")
plt.legend()
plt.show()


colors = [
    (0, 0, 1),    # Blue
    (0, 0, 0),    # Black
    (1, 0, 0)     # Red
]
colors_p = [
    (0, 0, 0),    # Black
    (1, 0, 0)     # Red
]
colors_n = [
    (0, 0, 0),    # Black
    (0, 0, 1),    # Blue
]

# Define the positions of each color segment
positions = [0, 0.5, 1]
positions_b = [0, 1]

# Create the colormap
from matplotlib.colors import LinearSegmentedColormap
custom_cmap = LinearSegmentedColormap.from_list("blue_black_red", list(zip(positions, colors)))
custom_cmap_p = LinearSegmentedColormap.from_list("blue_black_red_1", list(zip(positions_b, colors_p)))
custom_cmap_n = LinearSegmentedColormap.from_list("blue_black_red_1", list(zip(positions_b, colors_n)))

fig, axs = plt.subplots(1,3, figsize=(16, 12))  # Adjust size as necessary
M_matrix_pos,_ = nba.modularize(M_matrices_pos[0])
M_matrix_neg,_ = nba.modularize(M_matrices_neg[0])
M_matrix_full_arranged,_ = nba.modularize(M_matrices_ful.mean(axis=0))
simplt.plot_simmmatrix(M_matrices_ful[0],ax=axs[0],titles=f"Simmilarity Matrix",
                    colormap=custom_cmap,show_colorbar=False,parcel_ids_positions=parcel_ids_positions, show_parcels="VH",
                    result_path=None)
simplt.plot_simmmatrix(M_matrices_pos[0],ax=axs[1],titles=f"Correlations",
                    colormap=custom_cmap_p,show_colorbar=False,
                    result_path=None)
simplt.plot_simmmatrix(M_matrices_neg[0],ax=axs[2],titles=f"Anticorelations",
                    colormap=custom_cmap_n,show_colorbar=False,
                    result_path=None)
plt.show()




fig, axs = plt.subplots(2,2, figsize=(16, 12))  # Adjust size as necessary

M_matrix_pos,_ = nba.modularize(M_matrices_pos[0])
M_matrix_neg,_ = nba.modularize(M_matrices_neg[0])

simplt.plot_simmmatrix(M_matrices_ful.mean(axis=0),ax=axs[0,0],titles=f"M",
                    colormap="jet",show_colorbar=False,parcel_ids_positions=parcel_ids_positions, show_parcels="VH",
                    result_path=None)
simplt.plot_simmmatrix(M_matrix_pos-M_matrix_neg,ax=axs[0,1],titles=f"M+,M-",
                    colormap="magma",show_colorbar=False,
                    result_path=None)
simplt.plot_simmmatrix(M_matrix_pos,ax=axs[1,0],titles=f"M+",
                    colormap="magma",show_colorbar=False,
                    result_path=None)
simplt.plot_simmmatrix(M_matrix_neg,ax=axs[1,1],titles=f"M-",
                    colormap="magma",show_colorbar=False,
                    result_path=None)
plt.show()

nodes_0_edges = np.where(M_matrices_ful.mean(axis=0).sum(axis=0)==0)
label_indices[nodes_0_edges]
np.where()

# permuted_ids = reverse_cuthill_mckee(csr_matrix(M_matrices[0]))
metabmatrix_binarized   = copy.deepcopy(metab_sim)
metabmatrix_binarized[np.abs(metab_sim)<0.7]   = 0
metabmatrix_binarized[np.abs(metab_sim)>=0.7]  = 1
permuted_matrix = copy.deepcopy(metabmatrix_binarized)
permuted_matrix = permuted_matrix[permuted_ids,:]
permuted_matrix = permuted_matrix[:,permuted_ids]
simplt.plot_simmmatrix(negative_edges,ax=axs[2],titles=f"OG",
                    colormap="magma",
                    result_path=None)
plt.show()

debug.info(S_matrices.shape,M_matrices.shape)
order_arr = np.arange(1,10)
p_values = list()
for order in order_arr:
    debug.info("Order",order)
    
    outpath = join(resultdir,f"S_to_M-Causal_order_{order}")
    _pvalue_order = calculate_p_value_matrix(S_matrices, M_matrices, order)
    fig, axs = plt.subplots(1, figsize=(16, 12))  # Adjust size as necessary
    oder = 1
    simplt.plot_simmmatrix(_pvalue_order,ax=axs,titles=f"S -> M O{order}",
                        colormap="magma",
                        result_path=outpath,show_parcels="H")
    p_values.append(_pvalue_order)
    # plt.imshow(_pvalue_order)
    # plt.show()
p_values = np.array(p_values)
