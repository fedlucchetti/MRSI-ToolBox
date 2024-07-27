import os, sys, copy, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np

from graphplot.simmatrix import SimMatrixPlot
# from tqdm import tqdm
from rich.progress import Progress,track
from tools.progress_bar import ProgressBar
from tools.datautils import DataUtils
from os.path import split, join
from tools.filetools import FileTools
from tools.debug import Debug
from connectomics.parcellate import Parcellate
import networkx as nx
# from connectomics.network import NetBasedAnalysis
from registration.registration import Registration
from bids.mridata import MRIData
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import nibabel as nib

GROUP    = "Mindfulness-Project"

dutils   = DataUtils()
ftools   = FileTools(GROUP)
debug    = Debug()
reg      = Registration()
simplt   = SimMatrixPlot()

pb     = ProgressBar()
parc   = Parcellate()

PLOTDEBUG = False
N_RANDOM  = 20
PARC_SCHEME = "LFMIHIFIF-3"
################################################################################
METABOLITES       = ["Cr+PCr","Glu+Gln","GPC+PCh","Ins","NAA+NAAG"]
PARC_SCHEME       = "LFMIHIFIF"
PARC_SCALE        = "3"
N_PARCELS         = 250
PARC_STRING       = f"{PARC_SCHEME}-{PARC_SCALE}"
###############################################################################
###############################################################################
###############################################################################
# ignore_list  = ["stem","wm","cer"]
ignore_list  = ["wm","hypo","medulla","scp"]
main_parcels = ["ctx","subc","thal","amygd","hipp","hypo"]
fuse_parcels = ["amygd","hipp","hypo"]
sel_parcel_list = ["ctx-lh","subc-lh","thal-lh","amygd-lh","hipp-lh","stem",
                   "ctx-rh","subc-rh","thal-rh","amygd-rh","hipp-rh"]

############ Parcel List + Merge ##################
sel_parcel_list = ["ctx-rh","subc-rh","thal-rh","amygd-rh","hipp-rh",
                   "ctx-lh","subc-lh","thal-lh","amygd-lh","hipp-lh","stem"]
with open(join(dutils.DEVANALYSEPATH,"connectomics","data",f"merge_parcels_{PARC_STRING}.json"), 'r') as file:
    merge_parcels_dict = json.load(file)


################################################3

    

def merge_matrix(simmatrix_density, merge_parcels_dict):
    final_matrix = copy.deepcopy(simmatrix_density)
    if len(merge_parcels_dict)==0:
        return final_matrix
    n = simmatrix_density.shape[0]
    merged_matrix = simmatrix_density.copy()
    rows_to_remove = set()
    cols_to_remove = set()
    for target_row in merge_parcels_dict:
        _,last_idx = merge_parcels_dict[target_row]['merge']
        merge_indices = np.arange(target_row+1,last_idx+1)
        for idx in merge_indices:
            if idx != target_row:
                # Sum the rows
                merged_matrix[target_row, :] += simmatrix_density[idx, :]
                # Sum the columns
                merged_matrix[:, target_row] += simmatrix_density[:, idx]
        merged_matrix[:, target_row]/=len(merge_indices)
        merged_matrix[target_row, :]/=len(merge_indices)
        # Mark the merged rows and columns for removal
        for idx in merge_indices:
            if idx != target_row:
                rows_to_remove.add(idx)
                cols_to_remove.add(idx)
    # Remove the zeroed rows and columns (if they should be completely removed)
    keep_indices = sorted(set(range(n)) - rows_to_remove)
    final_matrix = merged_matrix[np.ix_(keep_indices, keep_indices)]
    return final_matrix

def compute_average_coordinate(label_image, label_value):
    # Find the indices where the label image equals the given label value
    indices = np.argwhere(label_image == label_value)
    
    # Compute the mean of these indices along the first axis
    average_coordinate = np.mean(indices, axis=0)
    
    return average_coordinate

def compute_centroids(label_image, label_indices):
    # Compute the average coordinate (centroid) for each label in label_indices
    centroids = np.array([compute_average_coordinate(label_image, label) for label in label_indices])
    
    return centroids


def compute_distance_matrix(centroids):
    # Compute the pairwise Euclidean distance matrix
    distance_matrix = cdist(centroids, centroids, metric='euclidean')
    
    return distance_matrix




############ List all subjects ##################
retain_list_arr   = np.load(join(dutils.ANARESULTSPATH,"Qcheck","retain_list.npz"))
subject_id_arr    = retain_list_arr["subject_id"]
############### List
density_con_arr = list()
length_con_arr  = list()
distance_matrix_arr = list()
metab_con_arr,metab_pvalues = list(),list()
metab_con_mi_arr,metab_pvalues_mi  = list(),list()
outdirpath = join(dutils.ANARESULTSPATH,"connectomes_M_vs_S")

############ List all subjects ##################
recording_list = np.array(ftools.list_recordings())
recording      = recording_list[42]
subject_id_arr = list()
session_arr    = list()
simmatrix_sp_leave_out_arr = list()
simmatrix_ids_to_delete_arr = list()
for ids, recording in enumerate(recording_list):
    debug.separator()
    # Init data
    debug.title(f"Processing {recording} - {ids}/{len(recording_list)}")
    subject_id,session = recording

    # subject_id,session="S045","V3"
    mridata  = MRIData(subject_id,session,group=GROUP)
    parcel_imgage = nib.load(mridata.data["parcels"][PARC_STRING]["orig"]["path"]).get_fdata()
    m_connectome_dir_path = join(mridata.ROOT_PATH,"derivatives","connectomes",
                            f"sub-{subject_id}",f"ses-{session}","spectroscopy")
    s_connectome_dir_path = join(mridata.ROOT_PATH,"derivatives","connectomes",
                            f"sub-{subject_id}",f"ses-{session}","dwi")
    
    simmatrix_metab         = np.zeros([N_PARCELS,N_PARCELS])
    pvalues                 = np.zeros([N_PARCELS,N_PARCELS])
    simmatrix_metab_mi      = np.zeros([N_PARCELS,N_PARCELS])
    pvalues_mi              = np.zeros([N_PARCELS,N_PARCELS])
    labels                  = N_PARCELS*[""]
    label_indices           = -1*np.zeros(N_PARCELS)
    label_indices_ignore    = [""]
    simmatrix_ids_to_delete = list()
    simmatrix_sp_leave_out  = np.zeros([5,N_PARCELS,N_PARCELS])
    distance_matrix         = np.zeros([N_PARCELS,N_PARCELS])
    simmatrix_density       = np.zeros([N_PARCELS,N_PARCELS])
    simmatrix_length        = np.zeros([N_PARCELS,N_PARCELS])
    try:
        if os.path.exists(m_connectome_dir_path):
            for filename in os.listdir(m_connectome_dir_path):
                if "npz" in filename and f"{PARC_SCHEME}_desc-scale{PARC_SCALE}" in filename:
                    data_metab = np.load(join(m_connectome_dir_path,filename))
                    simmatrix_metab         = data_metab["simmatrix_sp"]
                    pvalues                 = data_metab["pvalue_sp"]
                    simmatrix_metab_mi      = data_metab["simmatrix_mi"]
                    pvalues_mi              = data_metab["pvalue_mi"]
                    labels                  = data_metab["labels"]
                    label_indices           = data_metab["labels_indices"]
                    label_indices_ignore    = data_metab["parcel_labels_ignore"]
                    simmatrix_ids_to_delete = data_metab["simmatrix_ids_to_delete"]
                    simmatrix_sp_leave_out  = data_metab["simmatrix_sp_leave_out"]
                    centroids               = compute_centroids(parcel_imgage, label_indices)
                    distance_matrix         = compute_distance_matrix(centroids)
                    debug.success("distance_matrix",distance_matrix.shape)
                    simmatrix_metab[pvalues>0.005] = 0
                    n_labels = simmatrix_metab.shape[1]
                    break
        if os.path.exists(s_connectome_dir_path):
            for filename in os.listdir(s_connectome_dir_path):
                if "npz" in filename and f"{PARC_SCHEME}_desc-scale{PARC_SCALE}" in filename:
                    data_struct= np.load(join(s_connectome_dir_path,filename))
                    simmatrix_density = data_struct["connectome_density"][0:max(label_indices),0:max(label_indices)]
                    simmatrix_length  = data_struct["connectome_length"][0:max(label_indices),0:max(label_indices)]
                    simmatrix_density = merge_matrix(simmatrix_density,merge_parcels_dict)
                    simmatrix_length  = merge_matrix(simmatrix_length,merge_parcels_dict)
                    break
        density_con_arr.append(simmatrix_density)
        length_con_arr.append(simmatrix_length)
        metab_pvalues.append(pvalues)
        metab_con_arr.append(simmatrix_metab)
        metab_con_mi_arr.append(simmatrix_metab_mi)
        metab_pvalues_mi.append(pvalues_mi)
        subject_id_arr.append(subject_id)
        session_arr.append(session)
        simmatrix_sp_leave_out_arr.append(simmatrix_sp_leave_out)
        debug.info("simmatrix_sp_leave_out",simmatrix_sp_leave_out.shape)
        distance_matrix_arr.append(distance_matrix)
        simmatrix_ids_to_delete_arr.append(simmatrix_ids_to_delete)
    except Exception as e:
        debug.warning("",e)
    


density_con_arr     = np.array(density_con_arr)
length_con_arr      = np.array(length_con_arr)
metab_con_arr       = np.array(metab_con_arr)
metab_pvalues       = np.array(metab_pvalues)
simmatrix_sp_leave_out_arr = np.array(simmatrix_sp_leave_out_arr)
metab_con_mi_arr    = np.array(metab_con_mi_arr)
metab_pvalues_mi    = np.array(metab_pvalues_mi)
subject_id_arr      = np.array(subject_id_arr)
session_arr         = np.array(session_arr)
# simmatrix_ids_to_delete_arr = np.array(simmatrix_ids_to_delete_arr)


qmask_arr = np.ones(density_con_arr.shape[0:2])
for idd,_ids_to_del_list in enumerate(simmatrix_ids_to_delete_arr):
    qmask_arr[idd][_ids_to_del_list] = 0






np.savez(join(outdirpath,f"simM_metab_struct_desc_desc-scale{PARC_SCALE}.npz"),
         density_con_arr=density_con_arr,
         length_con_arr=length_con_arr,
         metab_con_sp_arr=metab_con_arr,
         metab_con_mi_arr=metab_con_mi_arr,
         metab_pvalues_sp=metab_pvalues,
         metab_pvalues_mi=metab_pvalues_mi,
         subject_id_arr=subject_id_arr,
         session_arr=session_arr,
         distance_matrix_arr=distance_matrix_arr,
         qmask_arr=qmask_arr,
         label_indices=label_indices[0:-1],
         simmatrix_sp_leave_out_arr=simmatrix_sp_leave_out_arr)

#############################################################
n_zeros_arr = list()
for i,sim in enumerate(metab_con_arr):
    n_zeros = len(np.where(sim==0)[0])
    debug.info(subject_id_arr[i],session_arr[i],n_zeros)
    n_zeros_arr.append(n_zeros)
n_zeros_arr = np.array(n_zeros_arr)
debug.info("mu 0:",n_zeros_arr.mean(),"std 0:",n_zeros_arr.std())
metab_con_arr_refined = list()
session_arr_ref = list()
subject_id_arr_ref = list()
for i,sim in enumerate(metab_con_arr):
    n_zeros = len(np.where(sim==0)[0])
    if n_zeros<n_zeros_arr.mean()+n_zeros_arr.std():
        metab_con_arr_refined.append(sim)
        session_arr_ref.append(session_arr[i])
        subject_id_arr_ref.append(subject_id_arr[i])
    else:
        debug.error(subject_id_arr[i],session_arr[i],n_zeros)
#############################################################

    
