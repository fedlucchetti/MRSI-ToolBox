import numpy as np
from tools.datautils import DataUtils
from tools.debug import Debug
from os.path import join, split 
import os , math
import scipy.stats as stats
from rich.console import Console
from rich.table import Table
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
from sklearn.cluster import KMeans
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
distance_matrix_arr        = data["distance_matrix_arr"]
metab_con_arr              = data["metab_con_sp_arr"]
subject_id_arr             = data["subject_id_arr"]
session_arr                = data["session_arr"]
qmask_arr                  = data["qmask_arr"]
label_indices_group        = data["label_indices"]
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




############## GET MNI Parcellation ###############
mni_template    = datasets.load_mni152_template()
parcel_t1w_path = mridata.data["parcels"]["LFMIHIFIF-3"]["orig"]["path"]
transform_list  = mridata.get_transform("forward","anat")
parcel_mni_img  = reg.transform(fixed_image=ants.from_nibabel(mni_template),moving_image=parcel_t1w_path,
                                interpolator_mode="genericLabel",transform=transform_list)
parcel_mni_img_nii = nib.Nifti1Image(parcel_mni_img.numpy(), mni_template.affine)
parcellation_data_np    = parcel_mni_img_nii.get_fdata()
ftools.save_nii_file(parcellation_data_np,mni_template.header,join(OUTDIR,"parcellation_mi152.nii.gz"))
centroids               = nettools.compute_centroids(parcellation_data_np, label_indices_group)
distance_matrix         = nettools.compute_distance_matrix(centroids)
##########################################################################

parcel_group_list = list()
sel_parcel_list = [["ctx-rh"],["subc-rh","thal-rh","amygd-rh","hipp-rh"],
                   ["ctx-lh"],["subc-lh","thal-lh","amygd-lh","hipp-lh"]]
for ipl, parcel_label in enumerate(parcel_labels_group):
    for i, big_parcel_label in enumerate(sel_parcel_list):
        for sub_parcel_label in big_parcel_label:
            if sub_parcel_label in parcel_label:
                parcel_group_list.append([ipl,i])

sel_parcel_list = ["ctx-rh","subc-rh","ctx-lh","subc-lh"]
parcel_group_list = np.array(parcel_group_list)
corr_list = list()
avg = 0
fig, axs = plt.subplots(4, figsize=(16, 12))  # Adjust size as necessary
for i, big_parcel_label in enumerate(sel_parcel_list):
    ids = np.where(parcel_group_list[:,1]==i)[0]
    distances = distance_matrix[np.ix_(ids,ids)].flatten()
    correlations = simmatrix_pop_clean[np.ix_(ids,ids)].flatten()
    id0 = np.where(distances!=0)[0]
    axs[i].plot(distances[id0],correlations[id0],".",label=big_parcel_label,markersize=0.5)
    r = stats.pearsonr(distances[id0],correlations[id0])
    corr_list.append(r)
    avg += r.statistic
plt.legend()
plt.show()

distances = distance_matrix.flatten() 
correlations = simmatrix_pop_clean.flatten()
idd0 = np.where(distances!=0)[0]

distances = distances[idd0]
correlations = correlations[idd0]

idc0 = np.where(correlations!=0)[0]
distances = distances[idc0]
correlations = correlations[idc0]
whole_brain_corr = stats.pearsonr(distances,correlations)
whole_brain_log_corr = stats.pearsonr(distances,np.log(np.abs((correlations))))

plt.plot(distances,correlations,".",markersize=0.5)
plt.show()

# Create the table
console = Console()

table = Table(title="Correlations vs Distance")
# Add columns
table.add_column("Parcel", justify="center", style="cyan", no_wrap=True)
table.add_column("r", justify="center", style="magenta")
table.add_column("SD", justify="center", style="green")
table.add_column("p-value", justify="center", style="red")

for i,corr in enumerate(corr_list):
    sd = corr.confidence_interval().high-corr.confidence_interval().low
    table.add_row(sel_parcel_list[i],f"{round(corr.correlation,2)}", f"{round(sd,2)}", f"{round(corr.pvalue,5)}")
table.add_row("Total",f"{avg/4}","" ,"")
sd = whole_brain_corr.confidence_interval().high-whole_brain_corr.confidence_interval().low
table.add_row("Whole Brain",f"{round(whole_brain_corr.correlation,2)}", f"{round(sd,2)}", f"{round(whole_brain_corr.pvalue,5)}")
sd = whole_brain_log_corr.confidence_interval().high-whole_brain_log_corr.confidence_interval().low
table.add_row("Whole Brain",f"{round(whole_brain_log_corr.correlation,2)}", f"{round(sd,2)}", f"{round(whole_brain_log_corr.pvalue,5)}")

console.print(table)








