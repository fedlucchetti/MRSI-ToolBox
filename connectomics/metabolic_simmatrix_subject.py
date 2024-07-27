import os, sys, copy,shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from graphplot.simmatrix import SimMatrixPlot
# from tqdm import tqdm
import json
from tools.progress_bar import ProgressBar
from tools.datautils import DataUtils
from os.path import split, join, exists
from tools.filetools import FileTools
from tools.debug import Debug
from connectomics.parcellate import Parcellate
from randomize.randomize import Randomize
from connectomics.network import NetBasedAnalysis
from registration.registration import Registration
from randomize.randomize import Randomize
from bids.mridata import MRIData
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
import argparse

METABOLITES           = ["NAANAAG", "Ins", "GPCPCh", "GluGln", "CrPCr"]
PARC_CEREB_SCHEME     = "cerebellum" 
dutils   = DataUtils()
debug    = Debug()
reg      = Registration()
pb       = ProgressBar()
parc     = Parcellate()
simplt   = SimMatrixPlot()
nba      = NetBasedAnalysis()
###############################################################################



def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process some input parameters.")

    # Add arguments
    parser.add_argument('--atlas', type=str, required=True, choices=['LFMIHIFIF-2', 'LFMIHIFIF-3', 'LFMIHIFIF-4', 
                                                                     'geometric_cubeK18mm','geometric_cubeK23mm',
                                                                     'aal', 'destrieux'], 
                        help='Atlas choice (must be one of: LFMIHIFIF-2, LFMIHIFIF-3, LFMIHIFIF-4, geometric, aal, destrieux)')
    parser.add_argument('--group', type=str, default='Mindfulness-Project', help='Group name (default: "Mindfulness-Project")')
    parser.add_argument('--atlas_wm', type=str, default='wm_cubeK18mm', help='White Matter Atlas(default: "wm_cubeK18mm")')
    parser.add_argument('--include_cer', type=int, default=1, help='Include cerebellum parcellation')
    parser.add_argument('--n_pert', type=int, default=50, help='Number of perturbations (default: 50)')
    parser.add_argument('--subject_id', type=str, help='subject id', default="S002")
    parser.add_argument('--session', type=str, help='recording session',choices=['V1', 'V2', 'V3'], default="V2")


    # Parse the arguments
    args = parser.parse_args()

    # Print the values for demonstration purposes
    debug.info(f"Atlas GM: {args.atlas}")
    debug.info(f"Atlas WM: {args.atlas_wm}")
    debug.info(f"Group: {args.group}")
    debug.info(f"Number of perturbations: {args.n_pert}")
    debug.info(f"subject id: {args.subject_id}")
    debug.info(f"session: {args.session}")

    subject_id  = args.subject_id
    session     = args.session
    GROUP       = args.group
    ftools      = FileTools(GROUP)
    N_PERT      = args.n_pert
    inc_cereb   = args.n_pert

    FONTSIZE       = 16
    PARC_SCHEME    = args.atlas
    PARC_WM_SCHEME = args.atlas_wm
    MERGE_PARCEL_PATH = join(dutils.DEVANALYSEPATH,"connectomics","data",f"merge_parcels_{PARC_SCHEME}.json")
    ###############################################################################
    ############ Parcel List + Merge ##################
    if PARC_SCHEME=="aal":
        sel_parcel_list = ["Frontal","Cingulum","Hippocampus","Occipital","Parietal",
                        "Thalamus","Temporal","Cerebelum"]
    else:
        sel_parcel_list = ["ctx-rh","subc-rh","thal-rh","hipp-rh",
                           "ctx-lh","subc-lh","thal-lh","hipp-lh","stem","cer"]
    merge_parcels_dict = {}
    if exists(MERGE_PARCEL_PATH):
        with open(MERGE_PARCEL_PATH, 'r') as file:
            merge_parcels_dict = json.load(file)



    ################################################################################
    ############ List all subjects ##################
    retain_list_arr   = np.load(join(dutils.ANARESULTSPATH,"Qcheck","retain_list.npz"))
    subject_id_arr    = retain_list_arr["subject_id"]
    ############ List all subjects ##################
    recording_list = np.array(ftools.list_recordings())
    debug.separator()
    # Init data
    
    prefix = f"sub-{subject_id}_ses-{session}"
    connectome_dir_path = join(dutils.DATAPATH,GROUP,"derivatives","connectomes",
                            f"sub-{subject_id}",f"ses-{session}","spectroscopy")
    if "LFMIHIFIF" in PARC_SCHEME:
        scale = PARC_SCHEME[-1]
        outfilename = f"{prefix}_run-01_acq-memprage_atlas-chimeraLFMIHIFIF_desc-scale{scale}grow2mm_dseg_simmatrix.npz"
    else:
        outfilename = f"{prefix}_run-01_acq-memprage_atlas-{PARC_SCHEME}_dseg_simmatrix.npz"
    outfilepath  = join(connectome_dir_path,outfilename)
    if exists(outfilepath):
        debug.success(prefix,"Already processed")
        # return
    mridata  = MRIData(subject_id,session,group=GROUP)
    # MRSI Data
    mrsi_ref_img_path = mridata.data["mrsi"]["Ins"]["orig"]["path"]
    mrsi_ref_img_np   = mridata.data["mrsi"]["Ins"]["orig"]["nifti"].get_fdata().squeeze()
    header_mrsi       = mridata.data["mrsi"]["Ins"]["orig"]["nifti"].header
    if mrsi_ref_img_path ==0:
        debug.error("No MRSI data found")
        return
    ##############################################################################
    ############## Transform Parcel image from Anat -> MRSI #############
    ##############################################################################
    debug.title(f"Compute Metabolic Simmilarity {prefix}")
    mrsi_orig_mask_np                                 = np.zeros(mrsi_ref_img_np.shape)
    mrsi_orig_mask_np[mrsi_ref_img_np>0]              = 1
    mridata.data["mrsi"]["mask"]["orig"]["nifti"]     = ftools.numpy_to_nifti(mrsi_orig_mask_np,header_mrsi)
    mridata.data["mrsi"]["mask"]["origfilt"]["nifti"] = ftools.numpy_to_nifti(mrsi_orig_mask_np,header_mrsi)

    anat_gm_parcel_orig_path = mridata.data["parcels"][PARC_SCHEME]["orig"]["path"]
    anat_wm_parcel_orig_path = mridata.data["parcels"][PARC_WM_SCHEME]["orig"]["path"]
    parcel_header_dict_gm    = parc.get_parcel_header(mridata.data["parcels"][PARC_SCHEME]["orig"]["labelpath"],cutoff=3001)
    parcel_header_dict_wm    = parc.get_parcel_header(mridata.data["parcels"][PARC_WM_SCHEME]["orig"]["labelpath"],cutoff=None)
    

    transform_list    = mridata.get_transform("forward","spectroscopy")
    parcel_gm_mrsi_np = reg.transform(mrsi_ref_img_path,anat_gm_parcel_orig_path,transform_list,
                                    interpolator_mode="genericLabel").numpy().astype(int)
    parcel_wm_mrsi_np = reg.transform(mrsi_ref_img_path,anat_wm_parcel_orig_path,transform_list,
                                    interpolator_mode="genericLabel").numpy().astype(int)
    parcel_gm_mrsi_np[parcel_gm_mrsi_np>=3000] = 0 # Mask tiny GM WM parcels
    
    if inc_cereb:
        mask_big_cer_parcel_indices = list()
        for k in parcel_header_dict_gm.keys():
             if "cer" == parcel_header_dict_gm[k]["label"][0]:
                mask_big_cer_parcel_indices.append(k)
                parcel_gm_mrsi_np[parcel_gm_mrsi_np==k] = 0
        for mask_big_cer_parcel_idx in mask_big_cer_parcel_indices:
            del parcel_header_dict_gm[mask_big_cer_parcel_idx]
            
        anat_cer_parcel_orig_path = mridata.data["parcels"][PARC_CEREB_SCHEME]["orig"]["path"]
        parcel_cereb_mrsi_np      = reg.transform(mrsi_ref_img_path,anat_cer_parcel_orig_path,transform_list,
                                        interpolator_mode="genericLabel").numpy().astype(int)
        parcel_gm_mrsi_np         = parc.merge_gm_wm_parcel(parcel_gm_mrsi_np, parcel_cereb_mrsi_np).astype(int)
        parcel_header_dict_cer    = parc.get_parcel_header(mridata.data["parcels"][PARC_CEREB_SCHEME]["orig"]["labelpath"],cutoff=None)
        parcel_header_dict_gm     = parc.merge_gm_wm_dict(parcel_header_dict_gm,parcel_header_dict_cer)

    parcel_mrsi_np      = parc.merge_gm_wm_parcel(parcel_gm_mrsi_np, parcel_wm_mrsi_np).astype(int)
    parcel_header_dict  = parc.merge_gm_wm_dict(parcel_header_dict_gm,parcel_header_dict_wm)

    ############ Get parcels and mask outside MRSI region   #############

    
    
    # parcel_mrsi_np ,parcel_header_dict = parc.filter_parcel(parcel_mrsi_np,parcel_header_dict ,ignore_list=ignore_list)
    parcel_mrsi_np ,parcel_header_dict = parc.merge_parcels(parcel_mrsi_np,parcel_header_dict, merge_parcels_dict)
    t1mask_orig_path   = mridata.data["t1w"]["mask"]["orig"]["path"]
    transform_list     = mridata.get_transform("inverse","spectroscopy")
    t1mask_mrsi_img    = reg.transform(mrsi_ref_img_path,t1mask_orig_path,transform_list).numpy()
    parcel_header_dict = parc.count_voxels_per_parcel(parcel_mrsi_np,mrsi_orig_mask_np,
                                                                    t1mask_mrsi_img,parcel_header_dict)
    unique_parcel_ids = np.unique(parcel_mrsi_np).astype(int)
    # debug.info("unique_parcel_ids",unique_parcel_ids)
    
    # for k,v in parcel_header_dict.items(): debug.info(k,v)
    # Extracting all label values without filtering on 'mask'
    all_labels_list         = [sub_dict['label'] for sub_dict in parcel_header_dict.values()]
    voxels_outside_mrsi     = {k: v for k, v in parcel_header_dict.items() if v['count'][-1] <= 5}
    # Extracting all 'label' values into a single list
    parcel_labels_ignore    = [sub_dict['label'] for sub_dict in voxels_outside_mrsi.values()]
    parcel_label_ids_ignore = [keys for keys in voxels_outside_mrsi.keys()]
    label_list_concat       = ["-".join(sublist) for sublist in all_labels_list]
    parcel_labels_ignore_concat = ["-".join(sublist) for sublist in parcel_labels_ignore]
    n_parcels               = len(parcel_header_dict)
    # for k in parcel_header_dict.keys():debug.info(k,parcel_header_dict[k])
    # sys.exit()
    os.makedirs(connectome_dir_path,exist_ok=True)
    ############ Parcellate and SimMatrix   #############
    ######### get parcel positions for 2d plot #########
    parcel_ids_positions, label_list_concat = parc.get_main_parcel_plot_positions(sel_parcel_list,label_list_concat)
    # if compute_flag:
    mrsirand       = Randomize(mridata,"origfilt")

    simmatrix_sp_1, pvalue_sp_1,parcel_concentrations   = parc.compute_simmatrix(mrsirand,parcel_mrsi_np,parcel_header_dict,parcel_label_ids_ignore,N_PERT,corr_mode = "spearman",rescale="zscore")
    simmatrix_sp_2, pvalue_sp_2,parcel_concentrations   = parc.compute_simmatrix(mrsirand,parcel_mrsi_np,parcel_header_dict,parcel_label_ids_ignore,N_PERT,corr_mode = "spearman2",rescale="zscore")

    # simmatrix_mi, pvalue_mi,_   = parc.compute_simmatrix(mrsirand,parcel_mrsi_np,parcel_header_dict,parcel_label_ids_ignore,N_PERT,corr_mode = "mi",rescale="mean")
    simmatrix_sp_leave_out      = parc.leave_one_out(simmatrix_sp_1,mrsirand,parcel_mrsi_np,parcel_header_dict,parcel_label_ids_ignore,N_PERT,corr_mode = "spearman",rescale="zscore")

    del parcel_header_dict[0]
    labels_indices = np.array(list(parcel_header_dict.keys()))
    np.trim_zeros(labels_indices)
    simmatrix_ids_to_delete=list()
    for idx_to_del in parcel_label_ids_ignore:
        simmatrix_ids_to_delete.append(np.where(labels_indices==idx_to_del)[0][0])
    
    ######### Save Results ########
    os.makedirs(connectome_dir_path,exist_ok=True)
    np.savez(f"{outfilepath}",
            parcel_concentrations   = parcel_concentrations,
            simmatrix_sp            = simmatrix_sp_1,
            pvalue_sp               = pvalue_sp_1,
            simmatrix_s2            = simmatrix_sp_2,
            pvalue_sp2              = pvalue_sp_2,
            simmatrix_sp_leave_out  = simmatrix_sp_leave_out,
            simmatrix_mi            = None,
            pvalue_mi               = None,
            labels                  = label_list_concat,
            labels_indices          = labels_indices,
            parcel_labels_ignore    = parcel_labels_ignore_concat,
            simmatrix_ids_to_delete = simmatrix_ids_to_delete,
            metabolites_leaveout    = METABOLITES)
    debug.success(f"Results Saved to {outfilepath}")
    debug.separator()
    try:    
    # dont show WM
        mask_ids = np.where(labels_indices>=9000)[0]
        simmatrix_sp_plot_1 = np.delete(simmatrix_sp_1,mask_ids,axis=0)
        simmatrix_sp_plot_1 = np.delete(simmatrix_sp_plot_1,mask_ids,axis=1)
        simmatrix_sp_plot_2 = np.delete(simmatrix_sp_2,mask_ids,axis=0)
        simmatrix_sp_plot_2 = np.delete(simmatrix_sp_plot_2,mask_ids,axis=1)
        # pvalue_sp_1    = np.delete(pvalue_sp_1,mask_ids,axis=0)
        # pvalue_sp_1    = np.delete(pvalue_sp_1,mask_ids,axis=1)
        fig, axs = plt.subplots(1,2, figsize=(16, 12))  # Adjust size as necessary
        plot_outpath = outfilepath.replace(".npz","_simmatrix")
        simplt.plot_simmatrix(simmatrix_sp_plot_1,ax=axs[0],titles=f"{prefix} Spearman",
                            scale_factor=0.4,
                            parcel_ids_positions=parcel_ids_positions,
                            colormap="blueblackred",show_parcels="H") 
        simplt.plot_simmatrix(simmatrix_sp_plot_2,ax=axs[1],titles=f"{prefix} Spearman 2",
                            scale_factor=0.4,
                            parcel_ids_positions=parcel_ids_positions,
                            colormap="blueblackred",show_parcels="H",result_path = plot_outpath) 
        # plt.show()
        ######### Adjacency Matrix ########

        # Adj Matrix
        simmatrix_sp       = copy.deepcopy(simmatrix_sp_1)
        simmatrix_adjusted = copy.deepcopy(simmatrix_sp)
        simmatrix_adjusted[pvalue_sp_1>=0.05]  = 0
        # Delete specified rows & columns

        array_after_row_deletion = np.delete(simmatrix_adjusted, simmatrix_ids_to_delete, axis=0)
        simmatrix_adjusted       = np.delete(array_after_row_deletion, simmatrix_ids_to_delete, axis=1)

        # simmatrix_adjusted = np.delete(simmatrix_adjusted,mask_ids,axis=0)
        # simmatrix_adjusted = np.delete(simmatrix_adjusted,mask_ids,axis=1)

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
        reference_degrees, rc_coefficients, mean_rc, std_rc, _, _ = nba.get_rc_distribution(simmatrix_binarized)
        ######## Restore OG simmatrix_sp ##########
        simmatrix_adjusted = copy.deepcopy(simmatrix_sp)
        simmatrix_adjusted[pvalue_sp_1>=0.05]  = 0
        simmatrix_binarized = copy.deepcopy(simmatrix_adjusted)
        simmatrix_binarized[np.abs(simmatrix_adjusted)<th]   = 0
        simmatrix_binarized[np.abs(simmatrix_adjusted)>=th]  = np.sign(simmatrix_adjusted[np.abs(simmatrix_adjusted)>=th])
        ######### SimMatrix PLots ########
        plot_outpath = outfilepath.replace(".npz","_plot_adjacency")
        fig, axs = plt.subplots(2,2, figsize=(16, 12))  # Adjust size as necessary
        simplt.plot_simmatrix(simmatrix_adjusted,ax=axs[0,0],titles=f"Metabolic Correlation",
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


        simplt.plot_simmatrix(simmatrix_binarized,ax=axs[0,1],titles=f"Binarized ",
                            parcel_ids_positions=parcel_ids_positions,colormap="jet",
                            scale_factor=0.6,
                            result_path = plot_outpath)
        plt.tight_layout() 
        # plt.show()
        os.makedirs(join(dutils.ANARESULTSPATH,"simmatrix_sp",GROUP),exist_ok=True)
        shutil.copyfile(f"{plot_outpath}.pdf",join(dutils.ANARESULTSPATH,"simmatrix_sp",GROUP,f"sub-{subject_id}_ses-{session}")) 


    except Exception as e:
        debug.error("Failed creating results",e)
        return



if __name__ == "__main__":
    main()





    


