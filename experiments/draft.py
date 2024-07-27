import os, sys, copy,shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np

from graphplot.simmatrix import SimMatrixPlot
# from tqdm import tqdm
from rich.progress import Progress,track
from tools.progress_bar import ProgressBar
from tools.datautils import DataUtils
from os.path import split, join, exists
from tools.filetools import FileTools
from tools.debug import Debug
from connectomics.parcellate import Parcellate
from randomize.randomize import Randomize
import networkx as nx
from connectomics.network import NetBasedAnalysis
from registration.registration import Registration
from randomize.randomize import Randomize
from bids.mridata import MRIData
import matplotlib.pyplot as plt
from scipy.stats import linregress
from collections import Counter
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.linear_model import RANSACRegressor

GROUP    = "Mindfulness-Project"
CORR_METHOD = "mi"

dutils   = DataUtils()
ftools   = FileTools(GROUP)
debug    = Debug()
reg      = Registration()
pb       = ProgressBar()
parc     = Parcellate()
simplt   = SimMatrixPlot()
nba      = NetBasedAnalysis()
###############################################################################
PLOTDEBUG = False
N_PERT  = 50
FONTSIZE = 16
###############################################################################

sel_parcel_list = ["ctx-rh","subc-rh","thal-rh","amygd-rh","hipp-rh",
                   "ctx-lh","subc-lh","thal-lh","amygd-lh","hipp-lh","stem"]

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
############ List all subjects ##################
retain_list_arr   = np.load(join(dutils.ANARESULTSPATH,"Qcheck","retain_list.npz"))
subject_id_arr    = retain_list_arr["subject_id"]
############ List all subjects ##################
recording_list    = np.array(ftools.list_recordings())
recording         = recording_list[2]
debug.separator()
# Init data
subject_id,session=recording
prefix = f"sub-{subject_id}_ses-{session}"
mridata  = MRIData(subject_id,session,group=GROUP)
connectome_dir_path = join(mridata.ROOT_PATH,"derivatives","connectomes",
                        f"sub-{subject_id}",f"ses-{session}","spectroscopy")
# MRSI Data
mrsi_ref_img_path = mridata.data["mrsi"]["Ins"]["orig"]["path"]
mrsi_ref_img_np   = mridata.data["mrsi"]["Ins"]["orig"]["nifti"].get_fdata().squeeze()
header_mrsi       = mridata.data["mrsi"]["Ins"]["orig"]["nifti"].header
##############################################################################
############## Transform Parcel image from Anat -> MRSI #############
##############################################################################
mrsi_orig_mask_np                                 = np.zeros(mrsi_ref_img_np.shape)
mrsi_orig_mask_np[mrsi_ref_img_np>0]              = 1
mridata.data["mrsi"]["mask"]["orig"]["nifti"]     = ftools.numpy_to_nifti(mrsi_orig_mask_np,header_mrsi)
mridata.data["mrsi"]["mask"]["origfilt"]["nifti"] = ftools.numpy_to_nifti(mrsi_orig_mask_np,header_mrsi)
#
mrsi_rootname          = mrsi_ref_img_path.replace(".nii.gz","")
brain_t1_path          = mridata.data["t1w"]["brain"]["orig"]["path"]
#
anat_parcel_nifti      = mridata.data["parcels"]["LFMIHIFIF-3"]["orig"]["path"]
anat_parcel_mrsi_nifti = anat_parcel_nifti.replace("space-orig","space-mrsi")
transform_list         = mridata.get_transform("forward","spectroscopy")
parcel_mrsi_np = reg.transform(mrsi_ref_img_path,anat_parcel_nifti,transform_list,
                                interpolator_mode="genericLabel").numpy()
############ Get parcels and mask outside MRSI region   #############
parcel_header_dict = parc.get_parcel_header(mridata.data["parcels"]["LFMIHIFIF-3"]["orig"]["labelpath"])
# parcel_mrsi_np ,parcel_header_dict = parc.filter_parcel(parcel_mrsi_np,parcel_header_dict ,ignore_list=ignore_list)
parcel_mrsi_np ,parcel_header_dict = parc.merge_parcels(parcel_mrsi_np,parcel_header_dict, merge_parcels_dict)
# mridata.data["parcel"] = parcel_mrsi_np
t1mask_orig_path = mridata.data["t1w"]["mask"]["orig"]["path"]
transform_list = mridata.get_transform("inverse","spectroscopy")
t1mask_mrsi_img = reg.transform(mrsi_ref_img_path,t1mask_orig_path,transform_list).numpy()
parcel_header_dict = parc.count_voxels_per_parcel(parcel_mrsi_np,mrsi_orig_mask_np,
                                                                t1mask_mrsi_img,parcel_header_dict)
# Extracting all label values without filtering on 'mask'
all_labels_list         = [sub_dict['label'] for sub_dict in parcel_header_dict.values()]
voxels_outside_mrsi     = {k: v for k, v in parcel_header_dict.items() if v['count'][-1] <= 5}
# Extracting all 'label' values into a single list
parcel_labels_ignore    = [sub_dict['label'] for sub_dict in voxels_outside_mrsi.values()]
parcel_label_ids_ignore = [keys for keys in voxels_outside_mrsi.keys()]
# parcel_labels_ignore.append("BND")
# parcel_label_ids_ignore.append(0)
label_list_concat       = ["-".join(sublist) for sublist in all_labels_list]
parcel_labels_ignore_concat = ["-".join(sublist) for sublist in parcel_labels_ignore]
n_parcels               = len(parcel_header_dict)
######### PATHS ########
outfilename  = split(anat_parcel_mrsi_nifti)[1].replace("space-mrsi_","")
outfilename  = outfilename.replace(".nii.gz","_simmatrix.npz")
outfilepath  = join(connectome_dir_path,outfilename)
os.makedirs(connectome_dir_path,exist_ok=True)
############ Parcellate and SimMatrix   #############
######### get parcel positions for 2d plot #########
parcel_ids_positions, label_list_concat = parc.get_main_parcel_plot_positions(sel_parcel_list,label_list_concat)
# if compute_flag:
debug.info("Randomize, Parcellate MRSI volume and compute SimMatrix")
mrsirand       = Randomize(mridata,"origfilt")

# simmatrix_mi, pvalue_mi   = parc.compute_simmatrix(mrsirand,parcel_mrsi_np,parcel_header_dict,parcel_label_ids_ignore,N_PERT,corr_mode = "mi",rescale="mean")
simmatrix_spz, pvalue_spz,parcel_concentrations = parc.compute_simmatrix(mrsirand,parcel_mrsi_np,parcel_header_dict,parcel_label_ids_ignore,N_PERT,corr_mode = "spearman",rescale="zscore")
simmatrix_ref             = copy.deepcopy(simmatrix_spz)
simmatrix_ref[pvalue_spz>0.005] = 0


for i, delta in enumerate(delta_arr):
    debug.success(mets_remove_list[i],delta)


































    labels_indices = np.array(list(parcel_header_dict.keys()))
    np.trim_zeros(labels_indices)
    simmatrix_ids_to_delete=list()
    for idx_to_del in parcel_label_ids_ignore:
        simmatrix_ids_to_delete.append(np.where(labels_indices==idx_to_del)[0][0])
            
    debug.success("simmatrix_sp shape",simmatrix_sp.shape)
    ######### Save Results ########
    os.makedirs(connectome_dir_path,exist_ok=True)
    np.savez(f"{outfilepath}",
            simmatrix_sp            = simmatrix_sp,
            pvalue_sp               = pvalue_sp,
            simmatrix_mi            = simmatrix_mi,
            pvalue_mi               = pvalue_mi,
            simmatrix_spz           = simmatrix_spz,
            pvalue_spz              = pvalue_spz,
            labels                  = label_list_concat,
            labels_indices          = labels_indices,
            parcel_labels_ignore    = parcel_labels_ignore_concat,
            simmatrix_ids_to_delete = simmatrix_ids_to_delete)
    debug.success(f"Results Saved to {outfilepath}")
    debug.separator()
    try:    
        fig, axs = plt.subplots(1,3, figsize=(16, 12))  # Adjust size as necessary
        plot_outpath = outfilepath.replace(".npz","_simmatrix")
        simplt.plot_simmmatrix(simmatrix_sp,ax=axs[0],titles=f"{prefix} Spearman mu-Norm",
                            scale_factor=0.4,
                            parcel_ids_positions=parcel_ids_positions,colormap="magma") 
        simplt.plot_simmmatrix(simmatrix_spz,ax=axs[1],titles=f"{prefix} Spearman Z-Norm",
                            scale_factor=0.4,
                            parcel_ids_positions=parcel_ids_positions,
                            colormap="magma",show_parcels="H") 
        simplt.plot_simmmatrix(simmatrix_mi,ax=axs[2],titles=f"{prefix} MI",
                            scale_factor=0.4,
                            parcel_ids_positions=parcel_ids_positions,
                            colormap="magma",show_parcels="H",result_path = plot_outpath) 
        ######### Adjacency Matrix ########

        # Adj Matrix
        simmatrix_adjusted = copy.deepcopy(simmatrix_sp)
        simmatrix_adjusted[pvalue_sp>=0.05]  = 0
        # Delete specified rows & columns

        array_after_row_deletion = np.delete(simmatrix_adjusted, simmatrix_ids_to_delete, axis=0)
        simmatrix_adjusted       = np.delete(array_after_row_deletion, simmatrix_ids_to_delete, axis=1)

        non_zero_indices   = np.where(simmatrix_adjusted.sum(axis=0) != 0)[0]
        simmatrix_adjusted = simmatrix_adjusted[non_zero_indices[:, None], non_zero_indices]
        th = 0.75
        simmatrix_binarized = copy.deepcopy(simmatrix_adjusted)
        simmatrix_binarized[np.abs(simmatrix_adjusted)<th]   = 0
        simmatrix_binarized[np.abs(simmatrix_adjusted)>=th]  = np.sign(simmatrix_adjusted[np.abs(simmatrix_adjusted)>=th])
        # Create the positive and negative edges subnetwork
        positive_edges = np.where(simmatrix_binarized == 1, 1, 0)
        negative_edges = np.where(simmatrix_binarized == -1, 1, 0)
        ######### Degree Distribution ########
        degree_distribution = nba.degree_distribution(simmatrix_binarized)
        degrees             = np.array(list(degree_distribution.keys()))
        ids                 = np.argsort(degrees)
        degree_counts       = np.array(list(degree_distribution.values()))
        degrees,degree_counts = degrees[ids[1::]],degree_counts[ids[1::]]
        ransac = RANSACRegressor()
        ransac.fit(degrees.reshape(-1, 1), np.log(degree_counts))
        X_fit        = np.linspace(degrees.min(), degrees.max(), 100).reshape(-1, 1)
        y_pred_huber = ransac.predict(X_fit)



        ######### RichClub ########
        adjacency_matrix = np.where(simmatrix_binarized == -1, 1, simmatrix_binarized)
        np.fill_diagonal(adjacency_matrix,0)
        G = nx.from_numpy_array(adjacency_matrix)
        reference_degrees = np.array(sorted(set(d for n, d in G.degree())))
        # Compute rich-club coefficient distribution for Metabolic network
        rc_coefficients = nba.rich_club_coefficient_curve(G, reference_degrees)
        # Compute rich-club coefficient distribution for random network
        mean_rc, std_rc = nba.rich_club_distribution(G, reference_degrees)


        ######## Restore OG simmatrix_sp ##########
        simmatrix_adjusted = copy.deepcopy(simmatrix_spz)
        simmatrix_adjusted[pvalue_sp>=0.05]  = 0
        simmatrix_binarized = copy.deepcopy(simmatrix_adjusted)
        simmatrix_binarized[np.abs(simmatrix_adjusted)<th]   = 0
        simmatrix_binarized[np.abs(simmatrix_adjusted)>=th]  = np.sign(simmatrix_adjusted[np.abs(simmatrix_adjusted)>=th])
        ######### SimMatrix PLots ########
        plot_outpath = outfilepath.replace(".npz","_plot_adjacency")
        fig, axs = plt.subplots(2,2, figsize=(16, 12))  # Adjust size as necessary
        simplt.plot_simmmatrix(simmatrix_adjusted,ax=axs[0,0],titles=f"Metabolic Correlation",
                            scale_factor=0.4,
                            parcel_ids_positions=parcel_ids_positions,colormap="magma") 

        axs[1,0].plot(degrees, degree_counts,".", color='r', alpha=0.7)
        axs[1,0].plot(X_fit, np.exp(y_pred_huber), color='red', label='Huber Regression')
        axs[1,0].set_xlabel('Degree',fontsize=FONTSIZE)
        axs[1,0].set_ylabel('Counts',fontsize=FONTSIZE)
        axs[1,0].set_yscale('log')
        axs[1,0].legend()
        axs[1,0].grid()


        axs[1,1].plot(reference_degrees, rc_coefficients, label='Metabolic Network', color='blue')
        axs[1,1].fill_between(reference_degrees, mean_rc - std_rc, mean_rc + std_rc, color='gray', alpha=0.5, label='Random Network ±1σ')
        axs[1,1].set_xlabel('Degree',fontsize=FONTSIZE)
        axs[1,1].set_ylabel('Rich-Club Coefficient',fontsize=FONTSIZE)
        axs[1,1].legend()
        axs[1,1].grid()


        simplt.plot_simmmatrix(simmatrix_binarized,ax=axs[0,1],titles=f"Binarized ",
                            parcel_ids_positions=parcel_ids_positions,colormap="jet",
                            scale_factor=0.6,
                            result_path = plot_outpath)
        plt.tight_layout() 
        # plt.show()
        os.makedirs(join(dutils.ANARESULTSPATH,"simmatrix_sp",GROUP),exist_ok=True)
        shutil.copyfile(f"{plot_outpath}.pdf",join(dutils.ANARESULTSPATH,"simmatrix_sp",GROUP,f"sub-{subject_id}_ses-{session}")) 


    except Exception as e:
        debug.error("Failed creating results",e)
        continue

    


