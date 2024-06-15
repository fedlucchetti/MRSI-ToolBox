import os, sys, copy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import powerlaw  # You may need to install this package
from graphplot import circular as circplt
from graphplot.simmatrix import SimMatrix
# from tqdm import tqdm
from registration.registration import Registration
from rich.progress import Progress,track
from tools.progress_bar import ProgressBar
from tools.datautils import DataUtils
# from graphplot.slices import PlotSlices
from os.path import split, join
from tools.filetools import FileTools
from tools.debug import Debug
from connectomics.parcellate import Parcellate
from mrsi.data import MRSIData
from randomize.randomize import Randomize
from tqdm.auto import tqdm  # For progress bar
import networkx as nx
from connectomics.network import NetBasedAnalysis
from scipy.stats import percentileofscore
from scipy.stats import pareto
import random
from registration.tools import RegTools
from scipy.stats import chi2
from graphplot.springnetwork import SpringNetwork

dutils = DataUtils()
ftools = FileTools()
debug  = Debug()
reg    = Registration()
pb     = ProgressBar()
# pltsl  = PlotSlices()
parc   = Parcellate()
pltsim = SimMatrix()
rand   = Randomize
netba  = NetBasedAnalysis()
regtools = RegTools("Conc")
simplt = SimMatrix()
netplt = SpringNetwork()
PLOTDEBUG = False
N_RANDOM  = 20

###############################################################################
GROUP = "MindfullTeen"
RESULTS_PATH = join(dutils.ANARESULTSPATH,"simmatrix",GROUP)
output_dir_path = join(RESULTS_PATH,"Population")
MRSI_REG_RESULT_PATH = "/media/flucchetti/NSA1TB1/Connectome/Data/MindfullTeen/Reg"
os.makedirs(RESULTS_PATH,exist_ok=True)
os.makedirs(output_dir_path,exist_ok=True)

###############################################################################
###############################################################################
# ignore_list  = ["stem","wm","cer"]
ignore_list  = ["wm","hypo","medulla","scp"]
main_parcels = ["ctx","subc","thal","amygd","hipp","hypo"]
fuse_parcels = ["amygd","hipp","hypo"]
sel_parcel_list = ["ctx-lh","subc-lh","thal-lh","amygd-lh","hipp-lh","stem",
                   "ctx-rh","subc-rh","thal-rh","amygd-rh","hipp-rh"]

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


################################################################################
METABOLITES  = ["Cr+PCr","Glu+Gln","GPC+PCh","Ins","NAA+NAAG"]
############ List all subjects ##################
retain_list_arr   = np.load(join(dutils.ANARESULTSPATH,"Qcheck","retain_list.npz"))
subject_id_arr    = retain_list_arr["subject_id"]
lipid_arr         = retain_list_arr["lipid"]
subject_list = [[subject_id, str(int(lipid_arr[ids]))] for ids, subject_id in enumerate(subject_id_arr)]
############ List all subjects ##################
subject_list = np.array(subject_list)
subject_list = subject_list[np.argsort(subject_list[:,0])]
subject_id   = subject_list[23]
# ids          = 23
# mrsiData     = MRSIData(subject_id,restype="OrigResFilt")
subject_loaded = 0
parcel_concentrations = None
# subject_list = subject_list[0:5]
with Progress() as progress:
    task1 = progress.add_task("[red]Parcellating...", total=len(subject_list))
    for ids, subject_id in enumerate(subject_list):
        progress.update(task1, advance=1)
        subject_id[1]            = str(int(subject_id[1] ))
        S,V,Lipid                = subject_id[0][0:4],subject_id[0][5:] ,subject_id[1]
        if V[1]!="1": continue
        mrsi_subject_reg_path    = join(MRSI_REG_RESULT_PATH,S+"_"+V,"LipRem_"+str(int(Lipid)))
        debug.title(f"Processing {subject_id}")
        mrsiData   = MRSIData(subject_id,restype="OrigResFilt")
        mrsirand   = rand(mrsiData,METABOLITES)
        ############ Get parcels and mask outside MRSI region   #############
        debug.info("Load parcels")
        transform_path           = join(mrsi_subject_reg_path,"Transform_MRSI2T1")
        transform_MRSI_T1_inv    = regtools.load_transform(transform_path,"inverse")
        parcel_image3D, parcel_header_dict = parc.get_parcel(subject_id,
                                                            group=GROUP,
                                                            transform=transform_MRSI_T1_inv,
                                                            target_image_path=mrsiData.path["Cr+PCr"])
        # parcel_image3D ,parcel_header_dict = parc.filter_parcel(parcel_image3D,parcel_header_dict ,ignore_list=ignore_list)
        parcel_image3D ,parcel_header_dict = parc.merge_parcels(parcel_image3D,parcel_header_dict, merge_parcels_dict)
        mrsiData.data["parcel"] = parcel_image3D
        # label_list_concat = ["-".join(sublist) for sublist in parcel_label_list]
        t1mask = np.zeros(mrsiData.data["t1"].shape)
        t1mask[mrsiData.data["t1"]>0] = 1
        parcel_header_dict = parc.count_voxels_per_parcel(parcel_image3D,mrsiData.data["mask"],
                                                                        t1mask,parcel_header_dict)
        # Extracting all label values without filtering on 'mask'
        all_labels_list         = [sub_dict['label'] for sub_dict in parcel_header_dict.values()]
        voxels_outside_mrsi     = {k: v for k, v in parcel_header_dict.items() if v['count'][-1] <= 5}
        # Extracting all 'label' values into a single list
        parcel_labels_ignore    = [sub_dict['label'] for sub_dict in voxels_outside_mrsi.values()]
        parcel_label_ids_ignore = [keys for keys in voxels_outside_mrsi.keys()]
        label_list_concat       = ["-".join(sublist) for sublist in all_labels_list]
        parcel_labels_ignore_concat = ["-".join(sublist) for sublist in parcel_labels_ignore]
        n_parcels               = len(parcel_header_dict)
        ############ Parcellate and SimMatrix   #############
        met_image4D_data           = mrsirand.sample_noisy_img4D()
        # parcel_concentrations      = parc.parcellate(met_image4D_data,mrsiData.data["parcel"],parcel_labels_ids)
        parcel_concentrations      = parc.parcellate_vectorized(met_image4D_data,mrsiData.data["parcel"],
                                                                parcel_header_dict,rescale=True,
                                                                parcel_concentrations=parcel_concentrations)
    



simmatrix, _pvalue_mat     = parc.compute_simmatrix_pearson(parcel_concentrations,
                                                            parcel_label_ids_ignore,
                                                            corr_mode = "spearman",show_progress=True)








# pvalue_mat.append(_pvalue_mat)
# simmatrix_list.append(simmatrix)

# pvalue_mat      = parc.combine_p_values_chi2(np.array(pvalue_mat))
# simmatrix_list  = np.array(simmatrix_list)
# pearson_mat     = np.mean(simmatrix_list,axis=0)
# pearson_mat[pvalue_mat>0.05] = 0
# pearson_mat[np.abs(pearson_mat)<np.std(simmatrix_list,axis=0)] = 0


### get parcel positions for 2d plot
parcel_ids_positions=dict()
for sel_parcel in sel_parcel_list:
    parcel_id_coord_list=list()
    for idp,parcel_label in enumerate(label_list_concat):
        if sel_parcel in parcel_label:
            parcel_id_coord_list.append(idp)
    a,b = min(parcel_id_coord_list),max(parcel_id_coord_list)+1
    parcel_ids_positions[sel_parcel] = [a,b]

label_list_concat = np.array(label_list_concat)
# sys.exit()
result_path = join(output_dir_path,"SimMatrix")
np.savez(f"{result_path}.npz",simmatrix=simmatrix,pvalue_mat=_pvalue_mat,
            labels=label_list_concat,parcel_labels_ignore=parcel_labels_ignore_concat)
debug.success(f"Saved to {result_path}")


######### SimMatrix ########
outpath = join(output_dir_path,"PopSimMatrix")
simmatrix[_pvalue_mat>0.05]=0
simplt.plot_simmmatrix(simmatrix,parcel_ids_positions, 
                        titles="Metabolic Simmilarity Matrix - MFT" , 
                        result_path=f"{outpath}", plotshow=True)
debug.separator()

# sys.exit()
    




####### Rich Club ########
adj_matrix          = copy.deepcopy(np.abs(simmatrix))
adj_matrix[adj_matrix < 0.5*np.max(simmatrix)] = 0
np.fill_diagonal(adj_matrix,0)
# Create graphs
adj_G               = nx.from_numpy_array(adj_matrix)
degrees             = dict(adj_G.degree())
degrees_arr = np.array(list(degrees.values()))
print(np.sort(degrees_arr))
# Rich club coeffs
observed_rich_club_coeff = nx.rich_club_coefficient(adj_G, normalized=False)

rand_G                  = netba.get_random_graphs(adj_matrix,N=1000)
random_rich_club_coeffs = [nx.rich_club_coefficient(rg, normalized=False) for rg in rand_G]
random_rc_arr           = np.array(list(random_rich_club_coeffs[0].values()))
observed_rc_arr         = np.array(list((observed_rich_club_coeff.values())))[0:len(random_rc_arr)]
# Statistical Significane
p_values                 = dict()
for k in observed_rich_club_coeff:
    random_coeffs_k  = np.array([rc[k] for rc in random_rich_club_coeffs if k in rc])
    observed_coeff_k = observed_rich_club_coeff[k]
    # Calculate the p-value for this degree k
    p_value     = (100 - percentileofscore(random_coeffs_k, observed_coeff_k, kind='strict')) / 100
    p_values[k] = p_value

rich_club_nodes     = [node for node, deg in degrees.items() if deg >= 7]
# rich_club_dict      = nx.algorithms.rich_club_coefficient(G, normalized=False, Q=100)
rich_club_labels    = [label_list_concat[node] for node in rich_club_nodes]
debug.success("rich_club parcels",rich_club_labels)

####### Fit Pareto ########
b, loc, scale = pareto.fit(simmatrix.flatten(), loc=0)
x = np.linspace(0.0, np.max(simmatrix), 100)  # Adjust as necessary for your data range
pareto_fit  = pareto.pdf(x, b, loc, scale)

# ####### Fit Pareto Connectivity########
b, loc, scale = pareto.fit(simmatrix.flatten(), loc=0)
x_con = np.linspace(-1, np.max(simmatrix), 100)  # Adjust as necessary for your data range
pareto_fit_con  = pareto.pdf(x_con, b, loc, scale)
####### Fit Pareto Degree Distribution ########
b, loc, scale = pareto.fit(list(degrees.values()))
x_deg = np.linspace(0.0, max(list(degrees.values())), 100)  # Adjust as necessary for your data range
pareto_fit_deg  = pareto.pdf(x_deg, b, loc, scale)


####### Select RichClub Nodes ########
selective_richclub_labels = list()
for idx, parcel_label in enumerate(label_list_concat):
    if parcel_label in rich_club_labels :
        selective_richclub_labels.append(parcel_label)
    else:
        selective_richclub_labels.append("")


# ######### SimMatrix ########
# debug.info("SimMatrix")
# fig, axs = plt.subplots(figsize=(12, 10))  # Adjust the figsize as needed
# cax = axs.matshow(pearson_mat, interpolation='nearest', cmap="viridis")  # Use a colormap that fits your data
# axs.grid(False)  # It's usually better to disable the grid for matshow
# axs.set_title('Similarity Matrix')
# axs.set_xticks(range(len(selective_richclub_labels)))
# axs.set_xticklabels(selective_richclub_labels, rotation=90, fontsize=16)
# axs.set_yticks(range(len(selective_richclub_labels)))
# axs.set_yticklabels(selective_richclub_labels, fontsize=16)
# # Add a color bar to indicate the strength
# fig.colorbar(cax, ax=axs, fraction=0.046, pad=0.04,label="Metabolite Correlation Stength")  # Adjust fraction and pad to fit your layout
# result_path=join(result_dir_path,'SimmilarityMatrix.pdf')
# # fig.savefig(result_path)  #
# plt.show()





FONTSIZE = 16
outpath = join(output_dir_path,"rich_club_analysis.pdf")
fig, axs = plt.subplots(3,1, figsize=(6, 10))  # Adjust the figsize as needed
# Histogram of Connectivity Strength
# th_simmatrix = copy.deepcopy(simmatrix)
axs[0].hist(simmatrix.flatten(), density=True,color="r",alpha=0.56, label="")
# axs[0].plot(x_con, pareto_fit_con, 'b', label="Pareto Fit")
axs[0].set_xlabel('Connectivity Strength',fontsize=FONTSIZE-2)
axs[0].set_ylabel('Number of nodes',fontsize=FONTSIZE-2)  # Correcting this line
axs[0].set_yscale('log')
axs[0].set_ylim([10e-2,10e2])
axs[0].grid(True)
axs[0].legend()

axs[1].hist(list(degrees.values()), density=False,color="r",alpha=0.56, label="")
axs[1].plot(x_deg, pareto_fit_deg, 'b', label="Pareto Fit")
axs[1].set_xlabel('Node degree',fontsize=FONTSIZE-2)
axs[1].set_ylabel('Number of nodes',fontsize=FONTSIZE-2)  # Correcting this line
axs[1].set_yscale('log')
axs[1].set_ylim([10e-2,10e2])
axs[1].grid(True)
axs[1].legend()

# Observed Rich-Club Coefficient Curve
axs[2].plot(range(observed_rc_arr.shape[0]),observed_rc_arr,"r",label="Observed")
axs[2].plot(range(random_rc_arr.shape[0]),random_rc_arr,"b--",linewidth=0.5,label="Random")
axs[2].plot(range(len(p_values.values())),p_values.values(),"k--",linewidth=0.5)
# axs[1].plot(list(observed_rc.keys()), list(observed_rc.values()), marker='.', linestyle='-', color='b')
# axs[1].plot(list(permuted_max_rc.keys()), list(permuted_max_rc.values()), marker='.', linestyle='-', color='b')
# axs[2].axvline(x=th_degree, color='red', linestyle='--', label='RichClub Threshold')
axs[2].set_xlabel('Degree Threshold',fontsize=FONTSIZE-2)
axs[2].set_ylabel('Rich-Club Coefficient',fontsize=FONTSIZE-2)
# axs[1].set_title('Observed Rich-Club Coefficient')
axs[2].legend()
axs[2].grid(True)
plt.tight_layout()
plt.show()
fig.savefig(outpath)  #



# netplt.plot_weighted_gr§aph(simmatrix,selective_richclub_labels)


adj_matrix          = copy.deepcopy(np.abs(simmatrix))
adj_matrix[adj_matrix < 0.5*np.max(simmatrix)] = 0
np.fill_diagonal(adj_matrix,0)
outpath = join(output_dir_path,"Connectogram.pdf")
fig,_ = circplt.plot_connectivity_circle(con=adj_matrix,node_names=selective_richclub_labels,interactive=True,
                                facecolor="k",textcolor="w",title="Metabolic Connectome - Rich Club Nodes - MFT",
                                fontsize_title=FONTSIZE+2,colorbar=True,colorbar_size=1,show=False)
fig.savefig(outpath)  



def plot_weighted_graph(simmatrix, labels):
    """
    Plots a graph based on a similarity matrix and node labels using the
    Fruchterman-Reingold layout.

    Parameters:
    - simmatrix: A 2D numpy array where each element represents the similarity (weight) between nodes.
    - labels: A list of labels for each node.
    """
    G = nx.Graph()
    # Add nodes with labels directly as identifiers
    for label in labels:
        G.add_node(label)
    # Add edges with weights from the similarity matrix
    for i in range(len(simmatrix)):
        for j in range(i+1, len(simmatrix)):
            # Add an edge only if there's a non-zero similarity
            if simmatrix[i][j] > 0:
                G.add_edge(labels[i], labels[j], weight=simmatrix[i][j])
    # Apply the Fruchterman-Reingold layout with weight consideration
    pos = nx.spring_layout(G, weight='weight')
    # Draw the graph
    fig, axs = plt.subplots(1, figsize=(6, 10))  # Adjust the figsize as needed
    nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', 
            width=1, font_size=10, node_size=500, alpha=0.9)
    axs.set_title("Weighted Network Graph using Fruchterman-Reingold Layout")
    return fig




####### Select RichClub Nodes ########

rich_club_nodes     = [node for node, deg in degrees.items() if deg >= 30]
# rich_club_dict      = nx.algorithms.rich_club_coefficient(G, normalized=False, Q=100)
rich_club_labels    = [label_list_concat[node] for node in rich_club_nodes]
debug.success("rich_club parcels",rich_club_labels)
selective_richclub_labels = list()
for idx, parcel_label in enumerate(label_list_concat):
    if parcel_label in rich_club_labels :
        selective_richclub_labels.append(parcel_label)
    else:
        selective_richclub_labels.append("")
outpath = join(output_dir_path,"Network.pdf")
fig = plot_weighted_graph(simmatrix,selective_richclub_labels)
fig.savefig(outpath)  
