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

    args        = parser.parse_args()
    ALPHA       = 0.05
    PARC_SCHEME = args.atlas 
    GROUP       = args.group
    ftools      = FileTools(GROUP)
    debug.title(f"Compute 4D homotopy for {GROUP} and atlas {PARC_SCHEME}")
    INCLUDE_WM  = bool(args.include_wm)
    debug.info(f"Include white matter {INCLUDE_WM}")
    ####################################
    weighted_metab_sim_list   = list()
    weighted_metab_sim_list2  = list()
    parcel_concentrations5D   = list()
    recording_list = np.array(ftools.list_recordings())
    recording      = recording_list[42]
    subject_id_arr,session_arr=list(),list()
    for idm,recording in enumerate(recording_list):
        subject_id,session=recording
        prefix = f"sub-{subject_id}_ses-{session}"
        debug.info(prefix)
        mridata  = MRIData(subject_id,session,group=GROUP)
        con_path = mridata.data["connectivity"]["spectroscopy"][PARC_SCHEME]["path"]
        if not exists(con_path):continue
        try:
            con_data = np.load(con_path)
            sim_matrix = con_data["simmatrix_sp"]
            p_values   = con_data["pvalue_sp"]
            sim_matrix[p_values>0.001] = 0
            sim_matrix2 = con_data["simmatrix_s2"]
            p_values   = con_data["pvalue_sp2"]
            sim_matrix2[p_values>0.001] = 0
            debug.info("parcel_concentrations",con_data["parcel_concentrations"].shape)
            parcel_concentrations5D.append(con_data["parcel_concentrations"])
            weighted_metab_sim_list.append(sim_matrix)
            weighted_metab_sim_list2.append(sim_matrix2)
            subject_id_arr.append(subject_id)
            session_arr.append(session)
        except Exception as e:
            debug.warning(prefix,e)

    label_indices_group     = copy.deepcopy(con_data["labels_indices"])
    _str_arr                 = copy.deepcopy(con_data["labels"])
    parcel_labels_group     = np.array([s for s in _str_arr if s != "BND"])


    parcel_concentrations5D = np.array(parcel_concentrations5D)

    mridata  = MRIData("S002","V3",group=GROUP)
    con_path = mridata.data["connectivity"]["spectroscopy"][PARC_SCHEME]["path"]
    label_indices_subj = copy.deepcopy(con_data["labels_indices"])
    _str_arr           = copy.deepcopy(con_data["labels"])
    parcel_labels_subj = np.array([s for s in _str_arr if s != "BND"])
    _simmatrix_subj    = con_data["simmatrix_sp"]

    ########## Clean simmilarity matrices ##########

    # Filter empy connectivty matrices 
    weighted_metab_sim_list   = np.array(weighted_metab_sim_list)
    weighted_metab_sim_list2  = np.array(weighted_metab_sim_list2)

    weighted_metab_sim,i,e    = simm.filter_sparse_matrices(weighted_metab_sim_list)
    weighted_metab_sim2       = np.delete(weighted_metab_sim_list2,e,axis=0)
    parcel_concentrations5D   = np.delete(parcel_concentrations5D,e,axis=0)
    session_arr               = np.delete(session_arr,e,axis=0)
    subject_id_arr            = np.delete(subject_id_arr,e,axis=0)

    weighted_metab_sim        = np.array(weighted_metab_sim)
    weighted_metab_sim2        = np.array(weighted_metab_sim2)



    ############# Detect empty correlations from pop AVG  #############
    simmatrix_pop             = weighted_metab_sim.mean(axis=0)
    simmatrix_pop2            = weighted_metab_sim2.mean(axis=0)
    n_recordings              = weighted_metab_sim.shape[0]
    debug.success("Aggregated n",n_recordings,"recordings")
    # 
    zero_diag_indices         = np.where(np.diag(simmatrix_pop) == 0)[0]

    wm_exclude                = np.where((label_indices_group >= 3000)&(label_indices_group<=4000))[0]
    # if INCLUDE_WM:
    #     # Inlcude WM parcels and exclude total WM
    #     wm_exclude            = np.where(label_indices_group == 3000)[0]
    # else:
    #     # Exclude all WM 
    #     wm_exclude           = np.where(label_indices_group >=3000)[0]
    mask_parcel_indices      = np.concatenate([zero_diag_indices,wm_exclude])


    # delete rowd/cols of empty correlations 
    simmatrix_pop_clean           = np.delete(simmatrix_pop, mask_parcel_indices, axis=0)
    simmatrix_pop_clean           = np.delete(simmatrix_pop_clean, mask_parcel_indices, axis=1)
    simmatrix_pop_clean2          = np.delete(simmatrix_pop2, mask_parcel_indices, axis=0)
    simmatrix_pop_clean2          = np.delete(simmatrix_pop_clean2, mask_parcel_indices, axis=1)
    parcel_concentrations5D       = np.delete(parcel_concentrations5D, mask_parcel_indices, axis=1)

    weighted_metab_sim_list       = np.delete(weighted_metab_sim_list, mask_parcel_indices, axis=1)
    weighted_metab_sim_list       = np.delete(weighted_metab_sim_list, mask_parcel_indices, axis=2)

    weighted_metab_sim_list2      = np.delete(weighted_metab_sim_list2, mask_parcel_indices, axis=1)
    weighted_metab_sim_list2      = np.delete(weighted_metab_sim_list2, mask_parcel_indices, axis=2)

    weighted_metab_sim_4D          = np.zeros((weighted_metab_sim_list.shape)+(2,))
    weighted_metab_sim_4D[:,:,:,0] = weighted_metab_sim_list
    weighted_metab_sim_4D[:,:,:,1] = weighted_metab_sim_list2




    # Regroup into 2D arr
    weighted_metab_sim_4D_avg     = np.zeros((simmatrix_pop_clean.shape)+(2,))
    weighted_metab_sim_4D_avg[:,:,0] = simmatrix_pop_clean
    weighted_metab_sim_4D_avg[:,:,1] = simmatrix_pop_clean2

    label_indices_group  = np.delete(label_indices_group, mask_parcel_indices)
    parcel_labels_group  = np.delete(parcel_labels_group, mask_parcel_indices)
    n_parcels_group      = len(parcel_labels_group)

    _simmatrix_subj   = np.delete(_simmatrix_subj, mask_parcel_indices, axis=0)
    simmatrix_subj    = np.delete(_simmatrix_subj, mask_parcel_indices, axis=1)

    # debug.info("mask_parcel_indices    ",mask_parcel_indices)
    # debug.info("label_indices_group    ",label_indices_group)
    # debug.info("parcel_labels_group    ",parcel_labels_group)

    label_indices_subj = np.delete(label_indices_subj, mask_parcel_indices)
    parcel_labels_subj = np.delete(parcel_labels_subj, mask_parcel_indices)
    n_parcels_subj     = len(parcel_labels_subj)

    # debug.info("label_indices_group             ",label_indices_group)
    # debug.info("parcel_labels_group             ",parcel_labels_group)
    # debug.info("weighted_metab_sim_4D     shape",weighted_metab_sim_4D.shape)
    # debug.info("weighted_metab_sim_4D_avg shape",weighted_metab_sim_4D_avg.shape)

    ############## Save intermdiate simmatrices and parcel conc
    resultssubdir = join(OUTDIR,"simmatrix",GROUP)
    os.makedirs(resultssubdir,exist_ok=True)
    filename = f"group-{GROUP}_atlas-{PARC_SCHEME}_desc-simmatrix_WM_{int(INCLUDE_WM)}.npz"
    outpath  = join(resultssubdir,filename)
    np.savez(outpath,parcel_concentrations5D   = parcel_concentrations5D,
                     weighted_metab_sim        = weighted_metab_sim_4D,
                     weighted_metab_sim_avg    = weighted_metab_sim_4D_avg,
                     label_indices_group       = label_indices_group,
                     parcel_labels_group       = parcel_labels_group,
                     subject_id_arr            = subject_id_arr,
                     session_arr               = session_arr)
    debug.success("Saved simmatrices, parcel conc to file",outpath)

    ############## GET MNI Parcellation ###############
    mni_template    = datasets.load_mni152_template()
    parcel_gm_t1w_path = mridata.data["parcels"][PARC_SCHEME]["orig"]["path"]
    gmParcel    = nib.load(parcel_gm_t1w_path).get_fdata()
    anat_header = nib.load(parcel_gm_t1w_path).header
    gmParcel[gmParcel>=3000] = 0
    if INCLUDE_WM:
        parcel_wm_t1w_path = mridata.data["parcels"][PARC_SCHEME]["orig"]["path"]
        wmParcel = nib.load(parcel_wm_t1w_path).get_fdata()
        parcel_t1w_np   = parc.merge_gm_wm_parcel(gmParcel, wmParcel)
        parcel_t1w_ni   = ftools.numpy_to_nifti(parcel_t1w_np,anat_header)
    else:
        parcel_t1w_ni   = ftools.numpy_to_nifti(gmParcel,anat_header)
    transform_list  = mridata.get_transform("forward","anat")
    parcel_mni_img  = reg.transform(fixed_image=mni_template,moving_image=parcel_t1w_ni,
                                    interpolator_mode="genericLabel",transform=transform_list)
    parcel_mni_img_nii   = nib.Nifti1Image(parcel_mni_img.numpy(), mni_template.affine)
    parcellation_data_np = parcel_mni_img_nii.get_fdata()
    ftools.save_nii_file(parcellation_data_np,mni_template.header,join(OUTDIR,f"parcellation_mi152_{PARC_SCHEME}.nii.gz"))


    ########## Compute individual Nodal Similarity Feature Vectors and then average ##########
    debug.info("Compute Group Nodal Similarity Feature Vectors")

    features2D_2  = simm.get_feature_nodal_similarity(weighted_metab_sim_4D_avg[:,:,1])
    features4D    = simm.get_4D_feature_nodal_similarity(weighted_metab_sim_4D_avg)
    projected_data_4D = list()
    resultssubdir = join(OUTDIR,"homotypic_clusters",PARC_SCHEME)
    os.makedirs(resultssubdir,exist_ok=True)
    for j in range(features4D.shape[1]):
        projected_data_3D = simm.nodal_strength_map(features4D[:,j],parcellation_data_np,label_indices_group)
        projected_data_4D.append(projected_data_3D)
    projected_data_4D = np.array(projected_data_4D)
    projected_data_4D = np.transpose(projected_data_4D, (1, 2, 3, 0))
    filename = f"group-{GROUP}_space-mni_atlas-{PARC_SCHEME}_desc-homotopy_WM_{int(INCLUDE_WM)}.nii.gz"
    outpath  = join(resultssubdir,filename)
    ftools.save_nii_file(projected_data_4D,mni_template.header,outpath)

    # sys.exit()
    debug.info("Compute individual Nodal Similarity Feature Vectors and then average")
    features4D_subj_list = np.zeros((weighted_metab_sim.shape[0],)+(features2D_2.shape[0],)+(4,))
    for i,_weighted_metab_sim_1 in enumerate(weighted_metab_sim):
        subject_id,session = subject_id_arr[i],session_arr[i]
        mridata            = MRIData(subject_id,session,group=GROUP)
        _weighted_metab_sim_2  = weighted_metab_sim2[i]
        _weighted_metab_sim_1  = np.delete(_weighted_metab_sim_1, mask_parcel_indices, axis=0)
        _weighted_metab_sim_1  = np.delete(_weighted_metab_sim_1, mask_parcel_indices, axis=1)
        _weighted_metab_sim_2  = np.delete(_weighted_metab_sim_2, mask_parcel_indices, axis=0)
        _weighted_metab_sim_2  = np.delete(_weighted_metab_sim_2, mask_parcel_indices, axis=1)
        features_1             = simm.get_feature_nodal_similarity(_weighted_metab_sim_1)
        features_2             = simm.get_feature_nodal_similarity(_weighted_metab_sim_2)
        features4D_subj_list[i,:,0:2]    = features_1
        features4D_subj_list[i,:,2:4]    = features_2
        weighted_metab_sim_4D_subj       = np.zeros((_weighted_metab_sim_1.shape)+(2,))
        weighted_metab_sim_4D_subj[:,:,0] = _weighted_metab_sim_1
        weighted_metab_sim_4D_subj[:,:,1] = _weighted_metab_sim_2
        features4D_subj = simm.get_4D_feature_nodal_similarity(weighted_metab_sim_4D_avg)
        projected_data_4D = list()
        for j in range(features4D_subj.shape[1]):
            projected_data_3D = simm.nodal_strength_map(features4D_subj[:,j],parcellation_data_np,label_indices_group)
            projected_data_4D.append(projected_data_3D)
        projected_data_4D = np.array(projected_data_4D)
        projected_data_4D = np.transpose(projected_data_4D, (1, 2, 3, 0))
        dir_path = split(mridata.data["connectivity"]["spectroscopy"][PARC_SCHEME]["path"])[0]
        dir_path = dir_path.replace("connectomes","homotopy")
        os.makedirs(dir_path,exist_ok=True)
        filename = f"sub-{subject_id}_ses-{session}_space-mni_atlas-{PARC_SCHEME}_desc-homotopy_WM_{int(INCLUDE_WM)}.nii.gz"
        outpath  = join(dir_path,filename)
        ftools.save_nii_file(projected_data_4D,mni_template.header,outpath)
        debug.info()
        filename = filename.replace("homotopy","featured4D")
        filename = filename.replace("nii.gz","npz")
        np.savez(join(dir_path,filename),features4D=features4D_subj,label_indices=label_indices_group)
        debug.info(i+1,weighted_metab_sim.shape[0])
    ######################################################################
    # features4D_subj_avg = features4D_subj_list.mean(axis=0)
    # filename = f"group-{GROUP}_space-mni_atlas-{PARC_SCHEME}_desc-groupavg_homotopy_WM_{int(INCLUDE_WM)}.nii.gz"
    # outpath  = join(resultssubdir,filename)
    # ftools.save_nii_file(projected_data_4D,mni_template.header,outpath)

if __name__ == "__main__":
    main()
