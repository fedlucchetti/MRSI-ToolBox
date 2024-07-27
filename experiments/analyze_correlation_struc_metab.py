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
density_matrix_arr         = data["density_con_arr"]
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


density = 0.10
weighted_metab_sim_list       = list()
weighted_struct_sim_list      = list()
binarized_metab_pop_list      = list()
binarized_struct_pop_list     = list()

for idm,metab_sim in enumerate(metab_con_arr):
    idt = np.where((subject_id_arr[idm] == subject_id_arr_th) & (session_arr[idm] == session_arr_th))[0]
    if len(idt)==0  : continue
    if metab_sim.sum()==0 or density_matrix_arr[idm].sum()==0: continue
    weighted_metab_sim_list.append(metab_sim)
    weighted_struct_sim_list.append(density_matrix_arr[idm])
    binarized_metab_pop_list.append(nba.binarize(metab_sim,threshold=density,threshold_mode="density",mode="abs"))
    binarized_struct_pop_list.append(nba.binarize(density_matrix_arr[idm],threshold=density,threshold_mode="density",mode="abs"))

weighted_metab_sim_list   = np.array(weighted_metab_sim_list)
weighted_struct_sim_list  = np.array(weighted_struct_sim_list)
binarized_metab_pop_list  = np.array(binarized_metab_pop_list)
binarized_struct_pop_list = np.array(binarized_struct_pop_list)

weighted_metab_sim_avg   = weighted_metab_sim_list.mean(axis=0)
weighted_struct_sim_avg  = weighted_struct_sim_list.mean(axis=0)

zero_diag_indices        = np.where(np.diag(weighted_metab_sim_avg) == 0)[0]
weighted_metab_sim_list  = np.delete(weighted_metab_sim_list, zero_diag_indices, axis=1)
weighted_metab_sim_list  = np.delete(weighted_metab_sim_list, zero_diag_indices, axis=2)
weighted_struct_sim_list = np.delete(weighted_struct_sim_list, zero_diag_indices, axis=1)
weighted_struct_sim_list = np.delete(weighted_struct_sim_list, zero_diag_indices, axis=2)

binarized_metab_pop_list  = np.delete(binarized_metab_pop_list, zero_diag_indices, axis=1)
binarized_metab_pop_list  = np.delete(binarized_metab_pop_list, zero_diag_indices, axis=2)
binarized_struct_pop_list = np.delete(binarized_struct_pop_list, zero_diag_indices, axis=1)
binarized_struct_pop_list = np.delete(binarized_struct_pop_list, zero_diag_indices, axis=2)

rw = stats.pearsonr(weighted_metab_sim_list.flatten(),weighted_struct_sim_list.flatten())
rb = stats.pearsonr(binarized_metab_pop_list.flatten(),binarized_struct_pop_list.flatten())







# Create the table
console = Console()

table = Table(title="Structure Vs Metabolic")
# Add columns
table.add_column("Matrix", justify="center", style="cyan", no_wrap=True)
table.add_column("Metric", justify="center", style="cyan", no_wrap=True)
table.add_column("r", justify="center", style="magenta")
table.add_column("SD", justify="center", style="green")
table.add_column("p-value", justify="center", style="red")

sd = rw.confidence_interval().high-rw.confidence_interval().low
table.add_row("Weighted","Correlation",f"{round(rw.correlation,2)}", f"{round(sd,2)}", f"{round(rw.pvalue,5)}")
sd = rb.confidence_interval().high-rb.confidence_interval().low
table.add_row("Binarized","Correlation",f"{round(rb.correlation,2)}", f"{round(sd,2)}", f"{round(rb.pvalue,5)}")
console.print(table)








