import os, sys, copy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import ants
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import networkx as nx
from scipy.stats import pareto
from graphplot import circular as circplt
from graphplot.simmatrix import SimMatrixPlot
from tqdm import tqdm
from registration.registration import Registration
from rich.progress import Progress,track
from tools.progress_bar import ProgressBar
from tools.datautils import DataUtils
# from graphplot.slices import PlotSlices
from os.path import split, join
from tools.filetools import FileTools
from tools.debug import Debug
from connectomics.parcellate import Parcellate
from connectomics.nettools import Tools
from connectomics.network import NetBasedAnalysis
import scipy.stats as stats
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '32'
import matplotlib.pyplot as plt

dutils = DataUtils()
ftools = FileTools()
ctools = Tools()

debug  = Debug()
reg    = Registration()
pb     = ProgressBar()
# pltsl  = PlotSlices()
parc   = Parcellate()
pltsim = SimMatrixPlot()
netba  = NetBasedAnalysis()
PLOTDEBUG = False
FONTSIZE=16

###############################################################################
RESULTS_PATH = join(dutils.ANARESULTSPATH,"simmatrix")
RESULTS_PLOT_PATH = join(dutils.ANARESULTSPATH,"simmatrix_plots")

###############################################################################
ignore_list  = ["stem","wm"]
main_parcels = ["ctx","subc","thal","amygd","hipp","hypo","cer"]
hemispheres  = ["lh","rh"]
################################################################################
############ List all subjects ##################
retain_list_arr = np.load(join(dutils.ANARESULTSPATH,"Qcheck","retain_list.npz"))
subject_id_arr  = retain_list_arr["subject_id"]
lipid           = retain_list_arr["lipid"]
sim_matrix_list = np.array(ctools.find_sim_matrix_files(RESULTS_PATH))

for idx , subject_id in enumerate(subject_id_arr):
    debug.separator()
    lipid_match = f"{int(lipid[idx]):02}"
    match_subject_id = f"{subject_id}_LipRem{lipid_match}"

    match_id = np.where(sim_matrix_list[:,1]==match_subject_id)[0]
    if len(match_id)==0:
        debug.warning(match_subject_id,"not found")
        continue
    
    match_id = match_id[0]
    # for ids, sim_path_id in enumerate(sim_matrix_list):

    path , subject_id    = sim_matrix_list[match_id]
    debug.info(subject_id)
    try:
        simmatrix            = np.load(path)["simmatrix"].mean(axis=-1)
        parcel_label_list    = np.load(path)["labels"]
        parcel_labels_ignore = np.load(path)["parcel_labels_ignore"]
        parcel_labels_ignore = parcel_labels_ignore[:,0]
        n_labels             = len(parcel_label_list)
        debug.success(match_subject_id,"found")
    except Exception as e:
        debug.error(e,"Opening simmatrix for ",match_subject_id)
        continue

    ###### Create result path
    result_dir_path = join(RESULTS_PLOT_PATH,subject_id)
    os.makedirs(result_dir_path,exist_ok=True)
    # Remove rows
    simmatrix         = np.delete(simmatrix, parcel_labels_ignore, axis=0)
    parcel_label_list = np.delete(parcel_label_list, parcel_labels_ignore, axis=0)
    # Remove columns from the result of the previous step
    simmatrix         = np.delete(simmatrix, parcel_labels_ignore, axis=1)

    # for th in np.linspace(0.05,1,20):
    #     activated_matrix, degrees = parc.activate_matrix_and_count_degrees(simmatrix, threshold=th*np.max(simmatrix))
    #     observed_rc, permuted_max_rc, p_value = netba.get_stat_rich_club(activated_matrix, num_permutations=1000)
    #     debug.info(th,p_value)


    ####### Rich Club ########
    activated_matrix, degrees,rich_club_nodes, stats = netba.get_stat_rich_club(simmatrix,bin_th=0.3)
    observed_rc     = stats["observed_rc"]
    permuted_max_rc = stats["permuted_max_rc"]
    p_value         = stats["p_value"]
    # rich_club_dict      = nx.algorithms.rich_club_coefficient(G, normalized=False, Q=100)
    rich_club_labels    = [parcel_label_list[node] for node in rich_club_nodes]
    debug.success("rich_club parcels",rich_club_labels)

    ####### Fit Pareto ########
    b, loc, scale = pareto.fit(simmatrix.flatten(), loc=0)
    x = np.linspace(0.0, np.max(simmatrix), 100)  # Adjust as necessary for your data range
    pareto_fit  = pareto.pdf(x, b, loc, scale)

    ####### Select RichClub Nodes ########
    selective_richclub_labels = list()
    for idx, parcel_label in enumerate(parcel_label_list):
        if parcel_label in rich_club_labels :
            selective_richclub_labels.append(parcel_label)
        else:
            selective_richclub_labels.append("")





    debug.info("Rich Club Analysis")
    ############# Rich Club Analysis #############
    fig, axs = plt.subplots(3,1, figsize=(6, 10))  # Adjust the figsize as needed
    # Histogram of Connectivity Strength
    # th_simmatrix = copy.deepcopy(simmatrix)
    axs[0].hist(simmatrix.flatten(), bins=100, density=True, label="Histogram")
    axs[0].plot(x, pareto_fit, 'r', label="Fitted Pareto distribution")
    axs[0].set_xlabel('Connectivity Strength',fontsize=FONTSIZE-2)
    axs[0].set_ylabel('Number of nodes',fontsize=FONTSIZE-2)  # Correcting this line
    axs[0].set_yscale('log')
    axs[0].set_ylim([10e-2,10e2])
    axs[0].grid(True)
    # Observed Rich-Club Coefficient Curve
    axs[1].plot(list(observed_rc.keys()), list(observed_rc.values()), marker='.', linestyle='-', color='b')
    axs[1].plot(list(permuted_max_rc.keys()), list(permuted_max_rc.values()), marker='.', linestyle='-', color='b')
    axs[1].axvline(x=th_degree, color='red', linestyle='--', label='RichClub Threshold')
    axs[1].set_xlabel('Degree Threshold',fontsize=FONTSIZE-2)
    axs[1].set_ylabel('Rich-Club Coefficient',fontsize=FONTSIZE-2)
    # axs[1].set_title('Observed Rich-Club Coefficient')
    axs[1].legend()
    axs[1].grid(True)
    # Histogram of Permuted Max Rich-Club Coefficients
    axs[2].hist(permuted_max_rc, bins=20, color='gray', alpha=0.7, label='Permuted Networks')
    axs[2].axvline(x=max(observed_rc.values()), color='red', linestyle='--', label='Observed Max RC')
    axs[2].set_xlabel('Max Rich-Club Coefficient',fontsize=FONTSIZE-2)
    axs[2].set_ylabel('Frequency',fontsize=FONTSIZE-2)
    # axs[2].set_title('Permuted Networks Max RC Distribution')
    axs[2].grid(True)
    axs[2].legend()
    plt.tight_layout()
    outpath=join(result_dir_path,'rich_club_analysis.pdf')
    fig.savefig(outpath)  #
    # plt.show()
    ###############################################



######### SimMatrix ########
debug.info("SimMatrix")
fig, axs = plt.subplots(figsize=(12, 10))  # Adjust the figsize as needed
cax = axs.matshow(pearson_mat, interpolation='nearest', cmap="viridis")  # Use a colormap that fits your data
axs.grid(False)  # It's usually better to disable the grid for matshow
axs.set_title('Similarity Matrix')
axs.set_xticks(range(len(selective_richclub_labels)))
axs.set_xticklabels(selective_richclub_labels, rotation=90, fontsize=16)
axs.set_yticks(range(len(selective_richclub_labels)))
axs.set_yticklabels(selective_richclub_labels, fontsize=16)
# Add a color bar to indicate the strength
fig.colorbar(cax, ax=axs, fraction=0.046, pad=0.04,label="Metabolite Correlation Stength")  # Adjust fraction and pad to fit your layout
outpath=join(result_dir_path,'SimmilarityMatrix.pdf')
fig.savefig(outpath)  #
plt.show()


    ########## Circular ########
    ## Rich Club
    debug.info("Circular Network Plot")
    fig,_ = circplt.plot_connectivity_circle(con=simmatrix,node_names=selective_richclub_labels,interactive=True,
                                    facecolor="k",textcolor="w",title="Metabolic Connectome - Rich Club Nodes",
                                    fontsize_title=FONTSIZE+2,colorbar=True,colorbar_size=1,show=False)
    outpath=join(result_dir_path,'CircularNetwork.pdf')
    fig.savefig(outpath)
    # plt.show()

    debug.success("Done")
    debug.separator()
    # ALL
    # circplt.plot_connectivity_circle(con=simmatrix,node_names=parcel_label_list,interactive=True)



