import numpy as np
from tools.datautils import DataUtils
from tools.debug import Debug
from os.path import join, split, exists
import os , math
import matplotlib.pyplot as plt
from tools.filetools import FileTools
from graphplot.simmatrix import SimMatrixPlot
from bids.mridata import MRIData
import nibabel as nib
from nilearn import plotting, image, datasets
from nilearn import datasets
from connectomics.netcluster import NetCluster
from connectomics.parcellate import Parcellate
import networkx as nx
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.pyplot as plt
import copy, sys
from registration.registration import Registration
from connectomics.network import NetBasedAnalysis
from connectomics.parcellate import Parcellate
from connectomics.robustness import NetRobustness
from connectomics.simmilarity import Simmilarity
from graphplot.nodal_simmilarity import NodalSimilarity
import argparse, json
from matplotlib.patches import ConnectionStyle
from nilearn.plotting import plot_stat_map
from nilearn.image import new_img_like
dutils    = DataUtils()
resultdir = join(dutils.ANARESULTSPATH,"connectomes_M_vs_S")
OUTDIR    = join(dutils.ANARESULTSPATH,"PLOS")
os.makedirs(OUTDIR,exist_ok=True)
from graphplot.colorbar import ColorBar

colorbar  = ColorBar()
simm      = Simmilarity()
reg       = Registration()
debug     = Debug()
netclust  = NetCluster()
pltnodal  = NodalSimilarity()
parc      = Parcellate()
nba       = NetBasedAnalysis()


# def main():
# Create the argument parser
# parser = argparse.ArgumentParser(description="Process some input parameters.")

# # Add arguments
# parser.add_argument('--atlas', type=str, required=True, choices=['LFMIHIFIF-2', 'LFMIHIFIF-3', 'LFMIHIFIF-4', 
#                                                                     'geometric_cubeK18mm','geometric_cubeK23mm',
#                                                                     'aal', 'destrieux'], 
#                     help='Atlas choice (must be one of: LFMIHIFIF-2, LFMIHIFIF-3, LFMIHIFIF-4, geometric, aal, destrieux)')
# parser.add_argument('--group', type=str, default='Mindfulness-Project', help='Group name (default: "Mindfulness-Project")')
# parser.add_argument('--include_wm', type=int, default=1, help='Include WM in (default: "Mindfulness-Project")')

# args        = parser.parse_args()
PARC_WM_SCHEME = "wm_cubeK18mm"
ALPHA       = 0.05
PARC_SCHEME =  "LFMIHIFIF-3"
GROUP       = "Mindfulness-Project"
ftools      = FileTools(GROUP)
INCLUDE_WM  = 1
METABOLITES = ["NAANAAG", "Ins", "GPCPCh", "GluGln", "CrPCr"]


################ TITLE ####################
debug.title(f"Compute 4D homotopy for {GROUP} and atlas {PARC_SCHEME}")
debug.info(f"Include white matter {INCLUDE_WM}: {PARC_WM_SCHEME}")


################# Load simmatrices ###################
clust_alg  = "monti"
n_clusters = 5
outsubdir  = join(OUTDIR,"homotypic_clusters",PARC_SCHEME,"simmatrixconcat_4D",f"WM_{INCLUDE_WM}",f"nclusters_{n_clusters}")
filename   = f"homotopy_simmatrixconcat_4D_{PARC_SCHEME}-WM_{INCLUDE_WM}-clust_{clust_alg}-nclust_{n_clusters}.npz"
outpath    = join(outsubdir,filename)
data       = np.load(outpath)
weighted_metab_sim_avg  = data["weighted_metab_sim_avg"]
parcel_concentrations4D = data["parcel_concentrations4D"]
cluster_labels          = data["cluster_labels"]
label_indices           = data["label_indices"]
gm_indices              = np.where(label_indices<3000)[0]
label_indices           = label_indices[gm_indices]
cluster_labels          = cluster_labels[gm_indices]

################ Compute centroids ######################3
parcel_concentrations3D_mu = np.zeros((n_clusters,)+parcel_concentrations4D.shape[1::])
parcel_concentrations3D_sd = np.zeros((n_clusters,)+parcel_concentrations4D.shape[1::])
for i,cluster_label in enumerate(np.unique(cluster_labels)):
    ids = np.where(cluster_labels==cluster_label)[0]
    parcel_concentrations3D_mu[i] = np.mean(parcel_concentrations4D[ids],axis=0)
    parcel_concentrations3D_sd[i] = np.std(parcel_concentrations4D[ids],axis=0)

color_list = ["blue","cyan","gold","orange","darkred"]
fig, axes = plt.subplots(5, figsize=(15, 5))
x_array = np.arange(0,len(METABOLITES))
for i,cluster_label in enumerate(np.unique(cluster_labels)):
    mu = parcel_concentrations3D_mu[i].mean(axis=-1)
    sd = np.sqrt(parcel_concentrations3D_mu[i].std(axis=-1)**2+parcel_concentrations3D_sd[i].std(axis=-1)**2)
    axes[i].fill_between(x_array,mu-sd,mu+sd,alpha=0.23,color=color_list[i])
    axes[i].plot(x_array,mu,"-o",color=color_list[i],label=f"H-cluster {cluster_label}")
    # axes[i].set_ylim(0.8,1.7)
    axes[i].set_ylabel("Z-Score")
    axes[i].legend()
    axes[i].set_xticklabels("", fontsize=16, fontweight='bold')

axes[i].set_xticks(x_array)
axes[i].set_xticklabels(METABOLITES, fontsize=16, fontweight='bold')
plt.show()

################# Load homotopy simmilarities ###################

filename   = f"homotopy_simmatrixconcat_4D_{PARC_SCHEME}-WM_{INCLUDE_WM}-clust_{clust_alg}-nclust_{n_clusters}.json"
outpath    = join(outsubdir,filename)
with open(outpath,"r") as f:
    correlation_dict = json.load(f)
filename   = f"homotopy_simmatrixconcat_4D_{PARC_SCHEME}-WM_{INCLUDE_WM}-clust_{clust_alg}-nclust_{n_clusters}_GM.nii.gz"
outpath    = join(outsubdir,filename)
volume3D   = nib.load(outpath)


################ Binarize #########################
weighted_metab_sim_avg = weighted_metab_sim_avg[gm_indices,:,:]
weighted_metab_sim_avg = weighted_metab_sim_avg[:,gm_indices,:]

binarized_matrix_1 = nba.binarize(weighted_metab_sim_avg[:,:,0], threshold=0.1, mode="abs", threshold_mode="density", binarize=True)
binarized_matrix_2 = nba.binarize(weighted_metab_sim_avg[:,:,1], threshold=0.1, mode="abs", threshold_mode="density", binarize=True)

################### Rich Club ##########################
mridata            = MRIData("S002","V3",group=GROUP)
rc_node_indices    = list()
rc_node_degrees    = list()
numbers,labels_gm,_      = parc.read_tsv_file(mridata.data["parcels"][PARC_SCHEME]["orig"]["labelpath"])
rc_degrees_M,rc_coefficients_M,mean_rc_M, std_rc_M, rc_deg_cutoff_M, pvalues = nba.get_rc_distribution(binarized_matrix_1,threshold_degree=0.6)
rich_club_node_indices_1, rich_club_node_degrees_1 = nba.extract_subnetwork(binarized_matrix_1,rc_deg_cutoff_M,label_indices)
deg_distr_1 = nba.get_degree_per_node(binarized_matrix_1)


rc_degrees_M,rc_coefficients_M,mean_rc_M, std_rc_M, rc_deg_cutoff_M, pvalues = nba.get_rc_distribution(binarized_matrix_2,threshold_degree=0.6)
rich_club_node_indices_2, rich_club_node_degrees_2 = nba.extract_subnetwork(binarized_matrix_2,rc_deg_cutoff_M,label_indices)
deg_distr_2 = nba.get_degree_per_node(binarized_matrix_2)

debug.warning("rich_club_node_indices_2",len(rich_club_node_indices_2))

rc_node_indices.extend(rich_club_node_indices_1)
rc_node_indices.extend(rich_club_node_indices_2)

rc_node_indices = list(set(rc_node_indices))
rc_node_indices = np.sort(np.array(rc_node_indices))

debug.success(len(rc_node_indices))

rc_list = list()
for i,rc_idx in enumerate(rc_node_indices):
    i = np.where(rc_idx==label_indices)[0][0]
    debug.warning(rc_idx,cluster_labels[i],labels_gm[rc_idx],deg_distr_1[i],deg_distr_2[i])
    rc_list.append([rc_idx,cluster_labels[i],labels_gm[rc_idx],deg_distr_1[i],deg_distr_2[i]])



rc_list = np.array(rc_list)
ids = np.argsort(rc_list[:,3].astype(int)+rc_list[:,4].astype(int))
rc_list = rc_list[ids,:]
for entry in rc_list:
    debug.info(entry)


rc_nodes_per_cluster_sorted = list()
for cluster_id in range(1,max(cluster_labels)+1):
    idc = np.where(cluster_id == rc_list[:,1].astype(int))
    selected_cluster_list = rc_list[idc,:]
    # debug.info(selected_cluster_list)
    rc_nodes_per_cluster_sorted.append(selected_cluster_list)

for i,_rc_nodes_per_cluster_sorted in enumerate(rc_nodes_per_cluster_sorted):
    _rc_nodes_per_cluster_sorted = np.array(_rc_nodes_per_cluster_sorted)[0]
    ids = np.argsort(_rc_nodes_per_cluster_sorted[:,3].astype(int))
    rc_nodes_per_cluster_sorted[i] = _rc_nodes_per_cluster_sorted[ids,:]
    debug.info(selected_cluster_list[ids])

selected_cluster_list = selected_cluster_list[0,:,:]
rc_show_indices = [[24,182],
                   [107,245,246,108,110,248,114,252,48,179,41,178],
                   [28,166,34,35,180,42],
                   [70,208,57,195,69,207,58,192,61,199,71,209,73,211,59,197],
                   list(rc_nodes_per_cluster_sorted[4][-20:,0].astype(int))]

labels_to_show = list()
for i,rc_list_per_cluster in enumerate(rc_show_indices):
    list_to_search = rc_list[:,0].astype(int)
    labels_per_rc_cluster = list()
    for rc_index in rc_list_per_cluster:
        ids = np.where(rc_index==list_to_search)[0]
        if len(ids)>0:
            label = rc_list[ids[0],2]
        labels_per_rc_cluster.append(label)
    labels_to_show.append(labels_per_rc_cluster)




# Preprocess the data: replace NaN with 0 and remove self-correlations
for node, edges in correlation_dict.items():
    for target, values in edges.items():
        edges[target] = [0 if np.isnan(v) else v for v in values]
    correlation_dict[node] = {k: v for k, v in edges.items() if k != node}

def preprocess_data(correlation_dict):
    for node, edges in correlation_dict.items():
        for target, values in edges.items():
            edges[target] = [0 if np.isnan(v) else v for v in values]
        correlation_dict[node] = {k: v for k, v in edges.items() if k != node}
    return correlation_dict

def draw_curved_edge(ax, src, dst, rad, width, color, offset):
    midpoint = (src + dst) / 2
    direction = dst - src
    norm_direction = direction / np.linalg.norm(direction)
    orth_direction = np.array([-norm_direction[1], norm_direction[0]])
    
    if rad < 0:
        orth_direction = -orth_direction
    
    midpoint += orth_direction * offset
    control_point = midpoint + orth_direction * abs(rad)
    
    path_data = [
        (Path.MOVETO, src),
        (Path.CURVE3, control_point),
        (Path.CURVE3, dst)
    ]
    
    codes, verts = zip(*path_data)
    path = Path(verts, codes)
    patch = PathPatch(path, facecolor='none', edgecolor=color, lw=width, alpha=0.8)
    ax.add_patch(patch)

def draw_curved_edges(G, pos, ax, offset=0.1):
    for (u, v, data) in G.edges(data=True):
        rad = 0.1
        draw_curved_edge(ax, np.array(pos[u]), np.array(pos[v]), rad, 10 * data['weight1'], data['color1'], offset)
        draw_curved_edge(ax, np.array(pos[u]), np.array(pos[v]), -rad, 10 * data['weight2'], data['color2'], offset)

def plot_graph(correlation_dict, node_colors, rc_size_cluster, fig=None, ax=None, rc_node_size=300):
    """
    Plots the graph with curved edges, node colors, and integer values around nodes.
    Parameters:
    correlation_dict (dict): Dictionary with correlation data.
    node_colors (list): List of colors for the nodes.
    rc_size_cluster (list): List of integer values to plot around nodes.
    fig (matplotlib.figure.Figure): Matplotlib figure object.
    ax (matplotlib.axes.Axes): Matplotlib axes object.
    Returns:
    None
    """
    # Preprocess the data
    correlation_dict = preprocess_data(correlation_dict)
    # Create the graph
    G = nx.Graph()
    # Add nodes with letters
    node_labels = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E'}
    G.add_nodes_from(node_labels.keys())
    # Add edges with correlation strengths
    for node, edges in correlation_dict.items():
        for target, (strength1, strength2) in edges.items():
            G.add_edge(int(node), int(target), weight1=abs(strength1), weight2=abs(strength2),
                       color1='red' if strength1 > 0 else 'blue',
                       color2='mediumvioletred' if strength2 > 0 else 'darkturquoise')
    
    # Define positions for a symmetric layout
    pos = nx.circular_layout(G)
    
    # Create a matplotlib figure and axis
    if fig is None and ax is None:
        fig, ax = plt.subplots()
    
    # Draw the nodes
    nx.draw_networkx_nodes(G, pos, node_size=1000, node_color=node_colors, ax=ax, alpha=0.85)
    
    # Draw curved edges
    draw_curved_edges(G, pos, ax)
    
    # Draw the labels
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=20, font_color='white', ax=ax)
    
    # Add surrounding subnodes and edges
    subnode_id = max(G.nodes) + 1
    subnode_positions = {}
    for i, (node, rc_size) in enumerate(zip(G.nodes(), rc_size_cluster)):
        x, y = pos[node]
        for j in range(rc_size):
            angle = 2 * np.pi * j / rc_size
            dx = 0.2 * np.cos(angle)
            dy = 0.2 * np.sin(angle)
            subnode_positions[subnode_id] = (x + dx, y + dy)
            G.add_node(subnode_id)
            G.add_edge(node, subnode_id, color1='red', color2='red')
            subnode_id += 1
        
        # Add edges between subnodes to form a fully connected subnetwork
        subnodes = list(subnode_positions.keys())[-rc_size:]
        for k in range(len(subnodes)):
            for l in range(k + 1, len(subnodes)):
                G.add_edge(subnodes[k], subnodes[l], color1='red', color2='red')
    
    # Draw all nodes including subnodes
    all_pos = {**pos, **subnode_positions}
    nx.draw_networkx_nodes(G, all_pos, node_size=rc_node_size, node_color='lightgrey', ax=ax, alpha=0.6)
    nx.draw_networkx_edges(G, all_pos, edgelist=[(u, v) for (u, v, d) in G.edges(data=True) if 'weight1' not in d], edge_color='red', alpha=0.6, ax=ax)
    
    # Draw labels for main nodes again to keep them visible
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=20, font_color='white', ax=ax)
    
    # Show the plot
    plt.title("5-Node Network with Double Correlation Edges and Surrounding Subnodes")
    plt.axis('off')
    plt.show()


def plot_brain_slices(image3D_nifti, output_file, fig=None, axes=[], vmin=None, vmax=None, colormap="jet", slices=[-5, -13, 0]):
    """
    """
    colormap = colormap
    mni_template = datasets.load_mni152_template()
    if vmin is None:
        vmin = image3D_nifti.get_fdata().min()
    if vmax is None:
        vmax = image3D_nifti.get_fdata().max()
    # Convert to Nifti1Image
    slice_str_arr = ["x", "y", "z"]
    titles = ["Sagittal", "Coronal", "Axial"]
    bar_flag = False
    if len(axes) == 0:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, slice_str in enumerate(slice_str_arr):
        if i == 2:
            bar_flag = True
        plotting.plot_stat_map(image3D_nifti, cmap=colormap, 
                            vmin=0,
                            vmax=vmax,
                            bg_img=mni_template,
                            cut_coords=[slices[i]],
                            display_mode=slice_str,
                            colorbar=bar_flag,
                            axes=axes[i],
                            annotate=False)
        axes[i].set_title(titles[i])
    if output_file:
        fig.savefig(f"{output_file}")
    # Define the colormap
    cmap = plt.cm.get_cmap('nipy_spectral', 7)
    # Get color codes for each integer value
    color_codes = {i: cmap(i / 6) for i in range(1, 6)}  # Exclude 0
    return fig, axes, color_codes



fig, axes = plt.subplots(2, 3, figsize=(15, 5))
colormap =  colorbar.bars("blueblackred")
_fig,_axes,color_codes = plot_brain_slices(volume3D,output_file=None,fig=fig, axes=axes[0,:],slices=[-2,-2,3])
node_colors = list(color_codes.values())

rc_size_cluster = [len(i)  for i in rc_show_indices]

plot_graph(correlation_dict,node_colors,rc_size_cluster,fig =fig,ax=axes[1,1],
           rc_node_size=23)
plt.show()

fig, axes = plt.subplots(1, figsize=(15, 5))
plot_graph(correlation_dict,node_colors,rc_size_cluster,fig =fig,ax=axes,
           rc_node_size=50)
plt.show()




# Preprocess the data
correlation_dict = preprocess_data(correlation_dict)

# Create the main graph
G = nx.Graph()
for node in correlation_dict:
    G.add_node(node)
# Add edges with weights and colors based on strengths
for node, edges in correlation_dict.items():
    for target, (strength1, strength2) in edges.items():
        G.add_edge(int(node), int(target), weight1=abs(strength1), weight2=abs(strength2),
                    color1='red' if strength1 > 0 else 'blue',
                    color2='mediumvioletred' if strength2 > 0 else 'darkturquoise')

# Generate positions for the main nodes
pos = nx.spring_layout(G, seed=42)

# Plot the main graph
fig, ax = plt.subplots(figsize=(12, 12))
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=12, ax=ax)
draw_curved_edges(G, pos, ax)

# Plot the subnodes around each main node
for idx, (node, size) in enumerate(zip(G.nodes, rc_size_cluster)):
    subG = nx.complete_graph(size)
    sub_pos = nx.circular_layout(subG, scale=0.2, center=pos[node])
    nx.draw_networkx_nodes(subG, sub_pos, node_color='red', node_size=50, ax=ax)
    nx.draw_networkx_edges(subG, sub_pos, edge_color='red', ax=ax, width=0.23)

plt.show()
        



# if __name__ == "__main__":
#     main()
