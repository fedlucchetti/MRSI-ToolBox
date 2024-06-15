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
import copy, sys
from connectomics.network import NetBasedAnalysis
from connectomics.parcellate import Parcellate
from connectomics.robustness import NetRobustness
from rich.console import Console
from rich.table import Table

dutils    = DataUtils()
resultdir = join(dutils.ANARESULTSPATH,"connectomes_M_vs_S")
GROUP     = "Mindfulness-Project"
simplt    = SimMatrix()
ftools    = FileTools(GROUP)
debug     = Debug()
nettools  = NetTools()
nba       = NetBasedAnalysis()
parc      = Parcellate()
netrobust = NetRobustness()

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

# length_con_arr      = data["distance_matrix_arr"]
metab_con_arr          = data["metab_con_sp_arr"]
subject_id_arr         = data["subject_id_arr"]
session_arr            = data["session_arr"]
qmask_arr              = data["qmask_arr"]
simmatrix_sp_leave_out_arr = data["simmatrix_sp_leave_out_arr"]

outpath            = join(dutils.ANARESULTSPATH,f"{GROUP}_threshold_list.npz")
data               = np.load(outpath)
subject_id_arr_th  = data["subject_id_arr"]
session_arr_th     = data["session_arr"]
threshold_arr      = data["best_thresholds"]



n_subjects      = metab_con_arr.shape[0]
METABOLITES           = ["NAANAAG", "Ins", "GPCPCh", "GluGln", "CrPCr"]

sel_parcel_list = ["ctx-rh","subc-rh","thal-rh","amygd-rh","hipp-rh",
                   "ctx-lh","subc-lh","thal-lh","amygd-lh","hipp-lh","stem"]
parcel_ids_positions, label_list_concat = parc.get_main_parcel_plot_positions(sel_parcel_list,label_list_concat)
FONTSIZE = 16
S_matrices,M_matrices         = list(), list()
M_matrices_pos,M_matrices_neg = list(), list()
M_matrices_ful                = list()
length_arr                    = list()
weighted_metab_sim            = list()
simmatrix_sp_leave_out_list   = list()

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
    # simmatrix_sp_leave_out    = simmatrix_sp_leave_out_arr[idm]
    weighted_metab_sim.append(metab_sim)
    simmatrix_sp_leave_out_list.append(simmatrix_sp_leave_out_arr[idm])


weighted_metab_sim         = np.array(weighted_metab_sim)
simmatrix_sp_leave_out_arr = np.array(simmatrix_sp_leave_out_list)


# Calculate the average matrix
weighted_metab_sim_mean = netrobust.average_matrices(weighted_metab_sim)

# Create a console object
console = Console()

# Create the table
table = Table(title="Correlations")

# Add columns
table.add_column("Comparison", justify="center", style="cyan", no_wrap=True)
table.add_column("r", justify="center", style="magenta")
table.add_column("SD", justify="center", style="green")
table.add_column("p-value", justify="center", style="red")


# Individual MSN edge weights VS sample mean MSN edge weights
table.add_row(f"Individual vs Sample Mean")
table.add_row(f"------------------------------------------", "-------------------","-------------------", "-------------------")


avg_edge_corr, std_edge_corr, p_value_edge_corr = netrobust.edge_weight_correlations(weighted_metab_sim, weighted_metab_sim_mean)
table.add_row("Edge weights", f"{avg_edge_corr:.2f}", f"{std_edge_corr:.2f}", f"{p_value_edge_corr:.5f}")

# Individual MSN nodal similarities VS sample mean MSN nodal similarities
avg_nodal_corr, std_nodal_corr, p_value_nodal_corr = netrobust.nodal_similarity_correlations(weighted_metab_sim, weighted_metab_sim_mean)
table.add_row("Nodal similarities ", f"{avg_nodal_corr:.2f}", f"{std_nodal_corr:.2f}", f"{p_value_nodal_corr:.5f}")

# Individual MSN edge weights VS individual leave-one-out MSN edge weights
table.add_row(f"------------------------------------------", "-------------------","-------------------", "-------------------")
table.add_row(f"Edge Weights Leave-Out")
table.add_row(f"------------------------------------------", "-------------------","-------------------", "-------------------")

for ids, metabolite in enumerate(METABOLITES):
    avg_set_corr, std_set_corr, p_value_set_corr = netrobust.compare_two_sets(weighted_metab_sim, simmatrix_sp_leave_out_arr[:, ids, :, :])
    table.add_row(f"{metabolite}", f"{avg_set_corr:.2f}", f"{std_set_corr:.2f}", f"{p_value_set_corr:.5f}")

table.add_row(f"------------------------------------------", "-------------------","-------------------", "-------------------")
table.add_row(f"Nodal Simmilarity Leave-Out")
table.add_row(f"------------------------------------------", "-------------------","-------------------", "-------------------")


# Individual MSN nodal similarities VS individual leave-one-out MSN nodal similarities
for ids, metabolite in enumerate(METABOLITES):
    avg_nodal_set_corr, std_nodal_set_corr, p_value_nodal_set_corr = netrobust.compare_nodal_similarities(weighted_metab_sim, simmatrix_sp_leave_out_arr[:, ids, :, :])
    table.add_row(f"{metabolite}", f"{avg_nodal_set_corr:.2f}", f"{std_nodal_set_corr:.2f}", f"{p_value_nodal_set_corr:.5f}")

# Print the table
console.print(table)














# fig, axs = plt.subplots(1,2, figsize=(16, 12))  # Adjust size as necessary
# simplt.plot_simmatrix(weighted_metab_sim[0],ax=axs[0],titles=f"M",
#                     colormap="jet",show_colorbar=False,parcel_ids_positions=parcel_ids_positions, show_parcels="VH",
#                     result_path=None)
# simplt.plot_simmatrix(weighted_metab_sim_clust,ax=axs[1],titles=f"M+,M-",
#                     colormap="jet",show_colorbar=False,
#                     result_path=None)
# plt.show()