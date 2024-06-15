import os, sys, copy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import ants
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import powerlaw  # You may need to install this package
from graphplot import circular as circplt
from graphplot.simmatrix import SimMatrix
# from tqdm import tqdm
from registration.registration import Registration
from registration.tools import RegTools
from scipy.stats import percentileofscore
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
from nilearn import plotting as niplt
from scipy.stats import pareto
from scipy.stats import chi2
from scipy.ndimage.measurements import center_of_mass
from nilearn.plotting import find_xyz_cut_coords
from connectomics.nettools import Tools


os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '32'

netba  = NetBasedAnalysis()
dutils = DataUtils()

rtools = RegTools()
ftools = FileTools()
debug  = Debug()
reg    = Registration()
pb     = ProgressBar()
# pltsl  = PlotSlices()
parc   = Parcellate()
pltsim = SimMatrix()
rand   = Randomize

ctools = Tools()


PLOTDEBUG = False


RESULTS_PATH = join(dutils.ANARESULTSPATH,"simmatrix")
os.makedirs(RESULTS_PATH,exist_ok=True)
RESULTS_PLOT_PATH = join(dutils.ANARESULTSPATH,"simmatrix_plots")

###############################################################################
N_RANDOM=10
###############################################################################
ignore_list  = ["stem","wm"]
main_parcels = ["ctx","subc","thal","amygd","hipp","hypo","cer"]
hemispheres  = ["lh","rh"]
################################################################################
METABOLITES  = ["Cr+PCr","Glu+Gln","GPC+PCh","Ins","NAA+NAAG"]
############ List all subjects ##################
subject_list = ftools.list_files_and_extract("MindfullTeen","OrigRes")

subject_id = subject_list[0]
ids        = 0
mrsiData   = MRSIData(subject_id)
mrsirand   = rand(mrsiData,METABOLITES)
######## Load retain&ignor list
retain_list_arr = np.load(join(dutils.ANARESULTSPATH,"Qcheck","retain_list.npz"))
subject_id_arr  = retain_list_arr["subject_id"]
lipid           = retain_list_arr["lipid"]
sim_matrix_list = np.array(ctools.find_sim_matrix_files(RESULTS_PATH))

ids, subject_id = 0,subject_list[0]
for ids, subject_id in enumerate(subject_list):
    debug.separator()

    # ##### Filter Subject IDs following retain list 
    # lipid_match = f"{int(lipid[ids]):02}"
    # match_subject_id = f"{subject_id}_LipRem{lipid_match}"

    # match_id = np.where(sim_matrix_list[:,1]==match_subject_id)[0]
    # if len(match_id)==0:
    #     debug.warning(match_subject_id,"not found")
    #     continue
    
    # match_id=match_id[0]
    # path , subject_id    = sim_matrix_list[match_id]

    debug.title(f"Processing {subject_id}")

    ###### Create result path
    # result_dir_path = join(RESULTS_PLOT_PATH,subject_id)
    result_dir_path = join(RESULTS_PLOT_PATH,f"{subject_id[0]}_LipRem{subject_id[1]}")
    if os.path.exists(join(result_dir_path,'RichClub-Pvalues.pdf')):
        continue
    os.makedirs(result_dir_path,exist_ok=True)
    ############ SETUP PATHS ##################
    mrsi_recording_id = ftools.get_mrsi_recording(subject_id)
    t1_path           = ftools.get_t1_path(subject_id)
    brain_t1_path, brain_t1_mask_path = reg.skull_strip(t1_path,image_type="t1")
    ############# T1 Image  #############
    mrsiData.data["t1signal"] = ants.image_read(f"{brain_t1_path}.gz").numpy()
    mrsiData.data["t1mask"]   = ants.image_read(brain_t1_mask_path).numpy()
    ############ Create MNI template ################## 
    debug.info("Load parcels")
    parcel_image3D, parcel_labels_ids, label_list, color_codes = parc.get_parcel(subject_id,ignore_list)
    n_parcels = len(label_list)
    _,parcel_labels_ignore = parc.find_ignored_parcels(parcel_image3D,mask=mrsiData.data["mask"])
    label_list_concat = ["-".join(sublist) for sublist in label_list]
    ############ Parcellate and SimMatrix   #############
    debug.info("Randomize, Parcellate MRSI volume and compute SimMatrix")
    threshold_arr  = np.linspace(0.10,0.70,7)
    threshold_arr  = np.concatenate([threshold_arr,np.linspace(0.70,0.98,15)])
    simmatrix_list = list()
    for i in tqdm(range(N_RANDOM)):
        met_image4D_data           = mrsirand.sample_noisy_img4D()
        parcel_concentrations      = parc.parcellate_vectorized(met_image4D_data,parcel_image3D,parcel_labels_ids)
        simmatrix, pvalue_mat      = parc.compute_simmatrix_pearson(parcel_concentrations,parcel_labels_ignore,rescale=False)
        simmatrix[pvalue_mat>0.05] = 0
        simmatrix_list.append(simmatrix)


    p_values_per_degree_per_th_list = list()
    with Progress() as progress:
        task1 = progress.add_task("[green]Thresholding SimMatrices...", total=len(threshold_arr))
        for idt,threshold in enumerate(threshold_arr):
            progress.update(task1, advance=1)
            max_degree = 0
            p_value_list   = list()
            # Randomize simmatrix
            # task2 = progress.add_task("[cyan]Compute Rich Club Coeffs...", total=len(simmatrix_list))
            for ids, simmatrix in  enumerate(simmatrix_list):
                # progress.update(task2, advance=1)
                # Binarize
                adj_matrix          = copy.deepcopy(simmatrix)
                adj_matrix[adj_matrix < threshold*np.max(simmatrix)] = 0
                np.fill_diagonal(adj_matrix,0)
                # Create graphs
                adj_G               = nx.from_numpy_array(adj_matrix)
                # Rich club coeffs
                observed_rich_club_coeff = nx.rich_club_coefficient(adj_G, normalized=False)
                if ids==0:
                    rand_G                  = netba.get_random_graphs(adj_matrix,N=1000)
                    random_rich_club_coeffs = [nx.rich_club_coefficient(rg, normalized=False) for rg in rand_G]
                # Statistical Significane
                p_values                 = dict()
                for k in observed_rich_club_coeff:
                    random_coeffs_k  = np.array([rc[k] for rc in random_rich_club_coeffs if k in rc])
                    observed_coeff_k = observed_rich_club_coeff[k]
                    # Calculate the p-value for this degree k
                    p_value     = (100 - percentileofscore(random_coeffs_k, observed_coeff_k, kind='strict')) / 100
                    p_values[k] = p_value
                max_degree=max(max_degree,k)
                p_value_list.append(p_values)
            # Merge pvalues per degree via Chi-2 test
            p_values_per_degree_list = dict()
            for degree in range(max_degree):
                p_values_per_degree = list()
                for p_values in p_value_list:
                    try:
                        p_values_per_degree.append(p_values[degree])
                    except:
                        pass
                chi_square_stat  = -2 * np.sum(np.log(np.array(p_values_per_degree)))
                df               = 2 * len(p_values_per_degree)  # degrees of freedom
                combined_p_value = chi2.sf(chi_square_stat, df)
                p_values_per_degree_list[degree] = combined_p_value
            p_values_per_degree_per_th_list.append(p_values_per_degree_list)


    alpha = 0.005
    max_size = 0
    for idt,threshold in enumerate(threshold_arr):
        p_values_per_degree =  p_values_per_degree_per_th_list[idt]
        degree_arr = list(p_values_per_degree.keys())
        max_size   = max(max_size,len(degree_arr))

    degree_arr = np.arange(max_size+1)[1:]
    pvalues_2D = np.zeros([len(threshold_arr),len(degree_arr)])
    for idt,threshold in enumerate(threshold_arr):
        p_values_per_degree = p_values_per_degree_per_th_list[idt]
        p_value_at_threshold = list()
        for degree in degree_arr:
            try:
                p_value_at_threshold.append(p_values_per_degree[degree])
            except:
                p_value_at_threshold.append(1)
        pvalues_2D[idt] = np.array(p_value_at_threshold)


    debug.success("Done computing pvalues")
    debug.info("Plotting")
    pvalues_2D[pvalues_2D>alpha]=1
    fig, ax = plt.subplots(figsize=(15, 10))
    # im = ax.imshow(pvalues_2D)
    cax = ax.imshow(pvalues_2D, interpolation='nearest', cmap="viridis")  # Use a colormap that fits your data
    # Show all ticks and label them with the respective list entries
    degree_arr_x = np.arange(0,max(degree_arr)+1,5)
    ax.set_xticks(degree_arr_x, labels=degree_arr_x.astype(str))
    degree_arr_y = np.linspace(0,max(threshold_arr),10)
    ax.set_yticks(degree_arr_y, labels=degree_arr_y.round(2).astype(str))
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04,label="RC P-value")  # Adjust fraction and pad to fit your layout
    fig.tight_layout()
    outpath=join(result_dir_path,'RichClub-Pvalues')
    fig.savefig(f"{outpath}.pdf")  #
    debug.success("Plot saved to",outpath)
    # plt.show()
    np.savez(f"{outpath}.npz",
             pvalues_2D=pvalues_2D,
             degree_arr=degree_arr,
             threshold_arr=threshold_arr)


debug.success("Done")

sys.exit()


# plt.plot([0,max(degree_arr)],[alpha,alpha],"r--")
# plt.legend()
# plt.grid()
# plt.xlabel("Node Degree")
# plt.ylabel("RC P-value")
# plt.yscale("log")
# plt.show()





    
#     # norm_met_image4D_data     = parc.normalizeMetabolites4D(met_image4D_data)
#     # # parc.plot_img_slices([met_image4D_data_1[0],met_image4D_data_2[0]],np.linspace(5,95,8),mask=mrsiData.data["mask"],PLOTSHOW=1)
#     # parcel_concentrations     = parc.parcellate_vectorized(met_image4D_data,parcel_image3D,parcel_labels_ids)
#     # # parcel_concentrations     = parc.parcellate(norm_met_image4D_data,parcel_image3D,parcel_labels_ids)
#     # # correlation_matrix[:,:,i] = parc.compute_simmatrix(parcel_concentrations,parcel_labels_ignore,rescale=False)
#     # simmatrix, pvalue_mat   = parc.compute_simmatrix_pearson(parcel_concentrations,parcel_labels_ignore,rescale=False)
#     # correlation_matrix[:,:,i] = simmatrix
#     # pvalue_matrix[:,:,i]      = pvalue_mat


# simmatrix[pvalue_mat>0.05] = 0
# # Average SimMatrices 
# parcel_label_ids_ignore = parcel_labels_ignore[:,0]
# n_labels                = len(label_list_concat)
# np.fill_diagonal(simmatrix,0)


# ####### Rich Club ########
# adj_matrix, _ = netba.activate_matrix_and_count_degrees(simmatrix, threshold=0.8*np.max(simmatrix))
# np.fill_diagonal(adj_matrix,0)

# G = nx.from_numpy_array(adj_matrix)
# G.remove_edges_from(nx.selfloop_edges(G))
# degrees             = dict(G.degree())
# observed_rc         = netba.get_rich_club_coefficient(adj_matrix)

# plt.hist(pvalue_mat.flatten())
# plt.show()

# plt.hist(simmatrix.flatten())
# plt.show()

# plt.hist(list(degrees.values()))
# plt.show()

# G = nx.from_numpy_array(adj_matrix)
# permuted_max_rc = []
# for _ in range(1000):
#     # Randomize the network while preserving the degree distribution
#     G_random      = nx.expected_degree_graph([d for n, d in nx.degree(G)], selfloops=False)
#     random_matrix = nx.to_numpy_array(G_random)
#     random_rc     = netba.get_rich_club_coefficient(random_matrix)
#     sim_rc        = np.array(list(random_rc.values()))
#     permuted_max_rc.append(sim_rc[0:35])

# permuted_max_rc     = np.array(permuted_max_rc)
# permuted_max_rc_avg = np.mean(permuted_max_rc,axis=0)
# permuted_max_rc_std = np.std(permuted_max_rc,axis=0)
# observed_rc_arr     = np.array(list((observed_rc.values())))[0:len(permuted_max_rc_avg)]






# ####### Fit Pareto Connectivity########
# b, loc, scale = pareto.fit(simmatrix.flatten(), loc=0)
# x_con = np.linspace(-1, np.max(simmatrix), 100)  # Adjust as necessary for your data range
# pareto_fit_con  = pareto.pdf(x_con, b, loc, scale)
# ####### Fit Pareto Degree Distribution ########
# b, loc, scale = pareto.fit(list(degrees.values()))
# x_deg = np.linspace(0.0, max(list(degrees.values())), 100)  # Adjust as necessary for your data range
# pareto_fit_deg  = pareto.pdf(x_deg, b, loc, scale)


# ####### Plot 
# FONTSIZE=16
# debug.info("Rich Club Analysis")
# ############# Rich Club Analysis #############
# fig, axs = plt.subplots(3,1, figsize=(6, 10))  # Adjust the figsize as needed
# # Histogram of Connectivity Strength
# # th_simmatrix = copy.deepcopy(simmatrix)
# axs[0].hist(simmatrix.flatten(), density=True,color="r",alpha=0.56, label="")
# # axs[0].plot(x_con, pareto_fit_con, 'b', label="Pareto Fit")
# axs[0].set_xlabel('Connectivity Strength',fontsize=FONTSIZE-2)
# axs[0].set_ylabel('Number of nodes',fontsize=FONTSIZE-2)  # Correcting this line
# axs[0].set_yscale('log')
# axs[0].set_ylim([10e-2,10e2])
# axs[0].grid(True)
# axs[0].legend()

# axs[1].hist(list(degrees.values()), density=False,color="r",alpha=0.56, label="")
# axs[1].plot(x_deg, pareto_fit_deg, 'b', label="Pareto Fit")
# axs[1].set_xlabel('Node degree',fontsize=FONTSIZE-2)
# axs[1].set_ylabel('Number of nodes',fontsize=FONTSIZE-2)  # Correcting this line
# axs[1].set_yscale('log')
# axs[1].set_ylim([10e-2,10e2])
# axs[1].grid(True)
# axs[1].legend()

# # Observed Rich-Club Coefficient Curve
# axs[2].plot(range(observed_rc_arr.shape[0]),observed_rc_arr,"r",label="Observed")
# axs[2].plot(range(permuted_max_rc_avg.shape[0]),permuted_max_rc_avg,"b--",linewidth=0.5)

# axs[2].fill_between(range(observed_rc_arr.shape[0]),
#                       permuted_max_rc_avg-3*permuted_max_rc_std,
#                       permuted_max_rc_avg+3*permuted_max_rc_std,
#                       alpha=0.23,
#                       color='skyblue',
#                       label="Random Graph")
# # axs[1].plot(list(observed_rc.keys()), list(observed_rc.values()), marker='.', linestyle='-', color='b')
# # axs[1].plot(list(permuted_max_rc.keys()), list(permuted_max_rc.values()), marker='.', linestyle='-', color='b')
# # axs[2].axvline(x=th_degree, color='red', linestyle='--', label='RichClub Threshold')
# axs[2].set_xlabel('Degree Threshold',fontsize=FONTSIZE-2)
# axs[2].set_ylabel('Rich-Club Coefficient',fontsize=FONTSIZE-2)
# # axs[1].set_title('Observed Rich-Club Coefficient')
# axs[2].legend()
# axs[2].grid(True)
# plt.tight_layout()
# plt.show()
# outpath=join('rich_club_analysis.pdf')
# fig.savefig(outpath)  #


# ##########3

# rich_club_nodes     = [node for node, deg in degrees.items() if deg >= 45]
# # observed_rc      = stats["observed_rc"]
# # permuted_max_rc  = stats["permuted_max_rc"]
# # p_value          = stats["p_value"]
# # rich_club_dict = nx.algorithms.rich_club_coefficient(G, normalized=False, Q=100)
# rich_club_labels = [label_list_concat[node] for node in rich_club_nodes]
# debug.success("rich_club parcels",rich_club_labels)



# # parcel_image3D_MNI = reg.transform(mni152_brain_path,ants.from_numpy(parcel_image3D),transform_MNI)
# # parcel_image3D_MNI = parcel_image3D_MNI.numpy().astype(np.int16)
# MNI_template = ants.image_read(mni152_brain_path)
# label_image_MNI = ants.apply_transforms(fixed=MNI_template, 
#                                         moving=ants.from_numpy(parcel_image3D),
#                                         transformlist=transform_MNI,
#                                         interpolator='nearestNeighbor')



# affineMNI152 = nib.load(mni152_brain_path).affine
# # Create a deep copy of the original image to preserve its data.
# rich_club_image = copy.deepcopy(parcel_image3D)
# # Initialize the rich_club_image to zeros
# rich_club_image[:] = 0
# # Iterate over each rich_club_node and selectively retain their values
# for rich_club_node in rich_club_nodes:
#     mask = parcel_image3D == rich_club_node
#     rich_club_image[mask] = rich_club_node

# label_image_RC_MNI = ants.apply_transforms(fixed=MNI_template, 
#                                         moving=ants.from_numpy(rich_club_image),
#                                         transformlist=transform_MNI,
#                                         interpolator='nearestNeighbor')


# # labels_T1 = np.unique(parcel_image3D)[1:]  # This skips the first label assuming it's 0 (background)

# # Calculate the center of mass for each label
# label_coords = []
# for label in rich_club_nodes:
#     # Find the center of mass for the current label
#     com_voxel = center_of_mass(label_image_RC_MNI.numpy() == label)
#     # Convert voxel coordinates to MNI space
#     # com_mni = nib.affines.apply_affine(affineMNI152, com_voxel)
#     label_coords.append(com_voxel)


# # parcel_coords    = ants.label_image_centroids(label_image_MNI)
# # centroids        = parc.compute_centroids(parcel_image3D_MNI,rich_club_nodes)
# rich_club_nodes  = np.array(rich_club_nodes)
# rich_club_matrix = adj_matrix[rich_club_nodes[:, None], rich_club_nodes]
# niplt.plot_connectome(rich_club_matrix, label_coords)
# niplt.show()



# ########################################
# debug.info("Plotting and saving results")
# outDir = join(RESULTS_PATH,f"{subject_id[0]}_LipRem{subject_id[1]}")
# os.makedirs(outDir,exist_ok=True)
# outpath = join(outDir,f"SimMatrix")
# np.savez(f"{outpath}.npz",simmatrix=correlation_matrix,labels=label_list_concat,parcel_labels_ignore=parcel_labels_ignore)
# debug.success(f"Saved to {outpath}")
# debug.separator()
    



