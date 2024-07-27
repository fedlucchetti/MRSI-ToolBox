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
from connectomics.netcluster import NetCluster
from nilearn import datasets
import copy, sys
from registration.registration import Registration
from connectomics.network import NetBasedAnalysis
from connectomics.parcellate import Parcellate
from connectomics.robustness import NetRobustness
from connectomics.simmilarity import Simmilarity
from graphplot.nodal_simmilarity import NodalSimilarity
import argparse


dutils    = DataUtils()
resultdir = join(dutils.ANARESULTSPATH,"connectomes_M_vs_S")
OUTDIR    = join(dutils.ANARESULTSPATH,"PLOS")
os.makedirs(OUTDIR,exist_ok=True)

simm      = Simmilarity()
reg       = Registration()
debug     = Debug()
netclust  = NetCluster()
pltnodal  = NodalSimilarity()
parc      = Parcellate()
simplt    = SimMatrixPlot()
nba       = NetBasedAnalysis()
FONTSIZE = 16

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process some input parameters.")

    # Add arguments
    parser.add_argument('--atlas', type=str, required=True, choices=['LFMIHIFIF-2', 'LFMIHIFIF-3', 'LFMIHIFIF-4', 
                                                                     'geometric_cubeK18mm','geometric_cubeK23mm',
                                                                     'aal', 'destrieux'], 
                        help='Atlas choice (must be one of: LFMIHIFIF-2, LFMIHIFIF-3, LFMIHIFIF-4, geometric, aal, destrieux)')
    parser.add_argument('--group', type=str, default='Mindfulness-Project', help='Group name (default: "Mindfulness-Project")')
    parser.add_argument('--include_wm', type=int, default=1, help='Include WM in (default: "Mindfulness-Project")')
    parser.add_argument('--subject_id', type=str, help='subject id', default="S002")
    parser.add_argument('--session', type=str, help='recording session',choices=['V1', 'V2', 'V3'], default="V2")

    args        = parser.parse_args()
    subject_id  = args.subject_id
    session     = args.session
    ALPHA       = 0.05
    PARC_SCHEME = args.atlas 
    GROUP       = args.group
    ftools      = FileTools(GROUP)
    debug.title(f"Compute 4D homotopy for {GROUP} and atlas {PARC_SCHEME}")
    INCLUDE_WM  = bool(args.include_wm)
    debug.info(f"Include white matter {INCLUDE_WM}")
    ####################################


    mridata  = MRIData(subject_id,session,group=GROUP)
    prefix   = f"sub-{subject_id}_ses-{session}"
    con_path = mridata.data["connectivity"]["spectroscopy"][PARC_SCHEME]["path"]
    con_data = np.load(con_path)
    sim_matrix = con_data["simmatrix_sp"]
    p_values   = con_data["pvalue_sp"]
    sim_matrix[p_values>0.001] = 0
    sim_matrix2 = con_data["simmatrix_s2"]

    p_values   = con_data["pvalue_sp2"]
    sim_matrix2[p_values>0.001] = 0
    labels_indices = con_data["labels_indices"]


    ########## Clean simmilarity matrices ##########
    
    # Only plot GM parcels
    gm_indices = np.where(labels_indices<3000)[0]
    weighted_metab_sim = sim_matrix[gm_indices,:]
    weighted_metab_sim = weighted_metab_sim[:,gm_indices]

    weighted_metab_sim2 = sim_matrix2[gm_indices,:]
    weighted_metab_sim2 = weighted_metab_sim2[:,gm_indices]

    # Filter empy edges matrices 
    zero_diag_indices              = np.where(np.diag(weighted_metab_sim) == 0)[0]
    weighted_metab_sim            = np.delete(weighted_metab_sim,zero_diag_indices,axis=0)
    weighted_metab_sim            = np.delete(weighted_metab_sim,zero_diag_indices,axis=1)
    weighted_metab_sim2            = np.delete(weighted_metab_sim2,zero_diag_indices,axis=0)
    weighted_metab_sim2            = np.delete(weighted_metab_sim2,zero_diag_indices,axis=1)
    labels_indices                 = np.delete(sim_matrix2,zero_diag_indices,axis=0)

    # Binarize
    binarized_matrix_1 = nba.binarize(weighted_metab_sim, threshold=0.1, mode="posneg", threshold_mode="density", binarize=True)
    binarized_matrix_2 = nba.binarize(weighted_metab_sim2, threshold=0.1, mode="posneg", threshold_mode="density", binarize=True)

    fig, axs = plt.subplots(2,2, figsize=(16, 12))  # Adjust size as necessary
    OUTDIR    = join(dutils.ANARESULTSPATH,"PLOS")

    OUTDIR    = join(dutils.ANARESULTSPATH,"PLOS")
    plot_outpath = join(OUTDIR,"Quadratic_vs_Linear_simmatrix")
    simplt.plot_simmatrix(weighted_metab_sim,ax=axs[0,0],titles=f"{prefix} Linear Corr",
                        scale_factor=0.4,
                        parcel_ids_positions=None,
                        colormap="blueblackred") 
    simplt.plot_simmatrix(weighted_metab_sim2,ax=axs[0,1],titles=f"{prefix} Quadratic Corr",
                        scale_factor=0.4,
                        parcel_ids_positions=None,
                        colormap="blueblackred",result_path = None) 
    simplt.plot_simmatrix(binarized_matrix_1,ax=axs[1,0],titles=f"Linear binarized: con density 0.1",
                        scale_factor=0.4,
                        parcel_ids_positions=None,
                        colormap="blueblackred") 
    simplt.plot_simmatrix(binarized_matrix_2,ax=axs[1,1],titles=f"Quadratic binarized: con density 0.1",
                        scale_factor=0.4,
                        parcel_ids_positions=None,
                        colormap="blueblackred",result_path = plot_outpath) 
    # plt.show()

    fig, axs = plt.subplots(1,3, figsize=(16, 12))  # Adjust size as necessary
    for i,threshold in enumerate([0.15,0.1,0.05]):
        binarized_matrix_1 = nba.binarize(weighted_metab_sim, threshold=threshold, mode="posneg", threshold_mode="density", binarize=True)
        rc_degrees_M,rc_coefficients_M,mean_rc_M, std_rc_M, rc_deg_cutoff_M, pvalues = nba.get_rc_distribution(binarized_matrix_1,threshold_degree=0.6)
        axs[i].plot(rc_degrees_M, rc_coefficients_M, label=f'edge density {threshold}', color='blue')
        axs[i].fill_between(rc_degrees_M, mean_rc_M - std_rc_M, mean_rc_M + std_rc_M, color='gray', alpha=0.5, label='Random Network ±1σ')
        axs[i].set_xlabel('Degree',fontsize=FONTSIZE)
        axs[i].set_ylabel('Rich-Club Coefficient',fontsize=FONTSIZE)
        axs[i].legend()
        axs[i].grid()
    plt.show()


if __name__ == "__main__":
    main()
