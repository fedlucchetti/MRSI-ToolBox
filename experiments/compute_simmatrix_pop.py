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

GROUP     = "Mindfulness-Project"
OUTDIR    = join(dutils.ANARESULTSPATH,"PLOS","simmatrix",GROUP)
os.makedirs(OUTDIR,exist_ok=True)

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

INCLUDE_WM       = False
############ Labels ############

####################################

METABOLITES     = ["NAANAAG", "Ins", "GPCPCh", "GluGln", "CrPCr"]
sel_parcel_list = ["ctx-rh","subc-rh","thal-rh","amygd-rh","hipp-rh",
                   "ctx-lh","subc-lh","thal-lh","amygd-lh","hipp-lh","stem"]
FONTSIZE        = 16

weighted_metab_sim_list   = list()
weighted_metab_sim_list2  = list()
parcel_concentrations5D   = list()
recording_list = np.array(ftools.list_recordings())
recording      = recording_list[42]
subject_id_arr,session_arr=list(),list()
for idm,recording in enumerate(recording_list):
    subject_id,session=recording
    prefix = f"sub-{subject_id}_ses-{session}"
    debug.info(prefix)
    mridata  = MRIData(subject_id,session,group=GROUP)
    con_path = mridata.data["connectivity"]["spectroscopy"][PARC_SCHEME]["path"]
    con_data = np.load(con_path)
    sim_matrix = con_data["simmatrix_sp"]
    p_values   = con_data["pvalue_sp"]
    sim_matrix[p_values>0.001] = 0
    sim_matrix2 = con_data["simmatrix_s2"]
    p_values   = con_data["pvalue_sp2"]
    sim_matrix2[p_values>0.001] = 0
    parcel_concentration = con_data["parcel_concentrations"]
    parcel_concentrations5D.append(parcel_concentration)
    weighted_metab_sim_list.append(sim_matrix)
    weighted_metab_sim_list2.append(sim_matrix2)
    subject_id_arr.append(subject_id)
    session_arr.append(session)


label_indices_group     = copy.deepcopy(con_data["labels_indices"])[0:-1]
parcel_labels_group     = copy.deepcopy(con_data["labels"])[0:-1]
parcel_concentrations5D = np.array(parcel_concentrations5D)

mridata  = MRIData("S002","V3",group=GROUP)
con_path = mridata.data["connectivity"]["spectroscopy"][PARC_SCHEME]["path"]
label_indices_subj = copy.deepcopy(con_data["labels_indices"])[0:-1]
parcel_labels_subj = copy.deepcopy(con_data["labels"])[0:-1]
_simmatrix_subj    = con_data["simmatrix_sp"]

########## Clean simmilarity matrices ##########

# Filter empy connectivty matrices 
weighted_metab_sim_list   = np.array(weighted_metab_sim_list)
weighted_metab_sim_list2  = np.array(weighted_metab_sim_list2)

weighted_metab_sim,i,e    = simm.filter_sparse_matrices(weighted_metab_sim_list)
weighted_metab_sim2       = np.delete(weighted_metab_sim_list2,e,axis=0)
parcel_concentrations5D   = np.delete(parcel_concentrations5D,e,axis=0)
session_arr               = np.delete(session_arr,e,axis=0)
subject_id_arr            = np.delete(subject_id_arr,e,axis=0)

weighted_metab_sim        = np.array(weighted_metab_sim)
weighted_metab_sim2        = np.array(weighted_metab_sim2)

weighted_metab_sim_4D     = np.zeros((weighted_metab_sim_list.shape)+(2,))
weighted_metab_sim_4D[:,:,:,0] = weighted_metab_sim_list
weighted_metab_sim_4D[:,:,:,1] = weighted_metab_sim_list2

simmatrix_pop             = weighted_metab_sim.mean(axis=0)
simmatrix_pop2            = weighted_metab_sim2.mean(axis=0)

n_recordings              = weighted_metab_sim.shape[0]
debug.success("Aggregated n",n_recordings,"recordings")

# Detect empty correlations 
zero_diag_indices         = np.where(np.diag(simmatrix_pop) == 0)[0]
if INCLUDE_WM:
    # Inlcude WM parcels and exclude total WM
    wm_exclude            = np.where(label_indices_group >3000)[0]
else:
    # Exclude all WM 
    wm_exclude            = np.where(label_indices_group >=3000)[0]
mask_indices             = np.concatenate([zero_diag_indices,wm_exclude])

# delete rowd/cols of empty correlations 
simmatrix_pop_clean           = np.delete(simmatrix_pop, mask_indices, axis=0)
simmatrix_pop_clean           = np.delete(simmatrix_pop_clean, mask_indices, axis=1)
simmatrix_pop_clean2          = np.delete(simmatrix_pop2, mask_indices, axis=0)
simmatrix_pop_clean2          = np.delete(simmatrix_pop_clean2, mask_indices, axis=1)
parcel_concentrations5D          = np.delete(parcel_concentrations5D, mask_indices, axis=1)

# Average all
weighted_metab_sim_4D_avg     = np.zeros((simmatrix_pop_clean.shape)+(2,))
weighted_metab_sim_4D_avg[:,:,0] = simmatrix_pop_clean
weighted_metab_sim_4D_avg[:,:,1] = simmatrix_pop_clean2

label_indices_group  = np.delete(label_indices_group, mask_indices)
parcel_labels_group  = np.delete(parcel_labels_group, mask_indices)
n_parcels_group      = len(parcel_labels_group)

_simmatrix_subj   = np.delete(_simmatrix_subj, mask_indices, axis=0)
simmatrix_subj    = np.delete(_simmatrix_subj, mask_indices, axis=1)


label_indices_subj = np.delete(label_indices_subj, mask_indices)
parcel_labels_subj = np.delete(parcel_labels_subj, mask_indices)
n_parcels_subj     = len(parcel_labels_subj)






############## Binarize ############## 
density = 0.10
binarized_matrix_pos         = nba.binarize(simmatrix_pop_clean,threshold=density,mode="pos",threshold_mode="density",binarize=False)
binarized_matrix_neg         = nba.binarize(simmatrix_pop_clean,threshold=density,mode="neg",threshold_mode="density",binarize=False)
binarized_matrix_abs         = nba.binarize(simmatrix_pop_clean,threshold=density,mode="abs",threshold_mode="density")
binarized_matrix_posneg      = nba.binarize(simmatrix_pop_clean,threshold=density,mode="posneg",threshold_mode="density")
binarized_matrix_sub_posneg  = nba.binarize(simmatrix_pop_clean,threshold=density,mode="posneg",threshold_mode="density")
binarized_matrix_posneg_2    = nba.binarize(weighted_metab_sim_4D_avg[:,:,1],threshold=0.05,mode="posneg",threshold_mode="density")


############## Plot Sim Matrices ###############
np.fill_diagonal(simmatrix_pop_clean, 0, wrap=False)
outpath = join(OUTDIR,f"Simmatrix_groupavg_atlas-{PARC_SCHEME}")
fig, axs = plt.subplots(2,2, figsize=(16, 12))  # Adjust size as necessary
simplt.plot_simmatrix(simmatrix_subj,ax=axs[0,0],titles=f"Individual Weighted Matrix",
                    colormap="bbo",show_colorbar=True,
                    result_path=None,scale_factor=0.3)
simplt.plot_simmatrix(binarized_matrix_sub_posneg,ax=axs[0,1],titles=f"Individual Binarized Matrix {round(100*density)}% Edge Density",
                    colormap="bbo",show_colorbar=True,
                    result_path=None,scale_factor=0.3)
np.fill_diagonal(simmatrix_pop_clean, 0, wrap=False)
simplt.plot_simmatrix(simmatrix_pop_clean,ax=axs[1,0],titles=f"Group Weighted Matrix",
                    colormap="bbo",show_colorbar=True,
                    result_path=None,scale_factor=0.3)
simplt.plot_simmatrix(binarized_matrix_posneg,ax=axs[1,1],titles=f"Group Binarized Matrix {round(100*density)}% Edge Density",
                    colormap="bbo",show_colorbar=True,
                    result_path=outpath,scale_factor=0.3)
plt.show()
############## Plot Sim Matrices Quadratic ###############
outpath = join(OUTDIR,f"Simmatrix_groupavg_quadratic_atlas-{PARC_SCHEME}")
fig, axs = plt.subplots(2,2, figsize=(16, 12))  # Adjust size as necessary
simplt.plot_simmatrix(weighted_metab_sim_4D_avg[:,:,0],ax=axs[0,0],titles=f"Order 1",
                    colormap="bbo",show_colorbar=True,
                    result_path=None,scale_factor=0.3)
simplt.plot_simmatrix(binarized_matrix_posneg,ax=axs[0,1],titles=f"Order 1 Binarized Matrix {round(100*density)}% Edge Density",
                    colormap="bbo",show_colorbar=True,
                    result_path=None,scale_factor=0.3)
simplt.plot_simmatrix(weighted_metab_sim_4D_avg[:,:,1],ax=axs[1,0],titles=f"Order 2",
                    colormap="bbo",show_colorbar=True,
                    result_path=None,scale_factor=0.3)
simplt.plot_simmatrix(binarized_matrix_posneg_2,ax=axs[1,1],titles=f"Order 2 Binarized Matrix {round(100*density)}% Edge Density",
                    colormap="bbo",show_colorbar=True,
                    result_path=outpath,scale_factor=0.3)
plt.show()


