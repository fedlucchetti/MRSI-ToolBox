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
from nilearn import datasets
from connectomics.netcluster import NetCluster
from connectomics.parcellate import Parcellate

import copy, sys
from registration.registration import Registration
from connectomics.network import NetBasedAnalysis
from connectomics.parcellate import Parcellate
from connectomics.robustness import NetRobustness
from connectomics.simmilarity import Simmilarity
from graphplot.nodal_simmilarity import NodalSimilarity
import argparse, json



PARC_CEREB_SCHEME = "cerebellum" 
PARC_WM_SCHEME    = "wm_cubeK18mm"

dutils    = DataUtils()
resultdir = join(dutils.ANARESULTSPATH,"connectomes_M_vs_S")
OUTDIR    = join(dutils.ANARESULTSPATH,"PLOS")
os.makedirs(OUTDIR,exist_ok=True)


nba       = NetBasedAnalysis()
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
    parser.add_argument('--include_wm', type=int, default=1, help='Include WM in (default: 1)')
    parser.add_argument('--include_cer', type=int, default=1, help='Include cerebellum in (default: 1)')
    args        = parser.parse_args()
    ALPHA       = 0.05
    PARC_SCHEME = args.atlas 
    GROUP       = args.group
    INCLUDE_CER = args.include_cer
    ftools      = FileTools(GROUP)
    INCLUDE_WM  = args.include_wm
    ################ TITLE ####################
    debug.title(f"Compute 4D homotopy for {GROUP} and atlas {PARC_SCHEME}")
    debug.info(f"Include white matter {INCLUDE_WM}: {PARC_WM_SCHEME}")
    ############## GET MNI Parcellation ###############
    mni_template       = datasets.load_mni152_template()
    mridata            = MRIData("S002","V3",group=GROUP)
    # GM
    labels_all = list()
    parcel_gm_t1w_path = mridata.data["parcels"][PARC_SCHEME]["orig"]["path"]
    parcel_t1w_gm_ni   = nib.load(parcel_gm_t1w_path)
    gmParcel           = parcel_t1w_gm_ni.get_fdata()
    gmParcel[gmParcel>=3000] = 0
    anat_header        = nib.load(parcel_gm_t1w_path).header
    indices_gm,labels_gm,_ = parc.read_tsv_file(mridata.data["parcels"][PARC_SCHEME]["orig"]["labelpath"])
    labels_gm              = np.delete(labels_gm,np.where(indices_gm>=3000)[0])
    # Cerebellum
    if INCLUDE_CER:
        parcel_cer_t1w_path = mridata.data["parcels"][PARC_CEREB_SCHEME]["orig"]["path"]
        parcel_t1w_cer_ni   = nib.load(parcel_cer_t1w_path)
        cerParcel           = parcel_t1w_cer_ni.get_fdata()
        _,labels_cer,_      = parc.read_tsv_file(mridata.data["parcels"][PARC_CEREB_SCHEME]["orig"]["labelpath"])
        cer_indices_list = list()
        for i,_label in enumerate(labels_gm):
            if "cer" in _label:
                cer_indices_list.append(i)
                debug.warning("Remove",i,labels_gm[i])
        
        debug.warning("cerebellum indices",cer_indices_list)
        gmParcel[gmParcel==cer_indices_list] = 0
        labels_gm = np.delete(labels_gm,cer_indices_list)
        labels_all.extend(labels_gm)
        labels_all.extend(labels_cer)
        gmParcel  = parc.merge_gm_wm_parcel(gmParcel, cerParcel)
        parcel_t1w_gm_ni = ftools.numpy_to_nifti(gmParcel,anat_header)
        # debug.info("labels_cer",labels_cer)
        # debug.info("labels_all",labels_all)
        

    # WM
    parcel_wm_t1w_path = mridata.data["parcels"][PARC_WM_SCHEME]["orig"]["path"]
    parcel_t1w_wm_ni   = nib.load(parcel_wm_t1w_path)
    wmParcel           = parcel_t1w_wm_ni.get_fdata()
    
    if INCLUDE_WM:
        parcel_t1w_np      = parc.merge_gm_wm_parcel(gmParcel, wmParcel)
        parcel_t1w_ni      = ftools.numpy_to_nifti(parcel_t1w_np,anat_header)
        _,labels_wm,_      = parc.read_tsv_file(mridata.data["parcels"][PARC_WM_SCHEME]["orig"]["labelpath"])
        labels_wm          = np.array(labels_wm).astype(int)+9001
        labels_wm          = list(labels_wm.astype(str))
        labels_all.extend(labels_wm)
    else:
        parcel_t1w_ni   = ftools.numpy_to_nifti(gmParcel,anat_header)

    transform_list  = mridata.get_transform("forward","anat")
    parcel_wm_mni_img  = reg.transform(fixed_image=mni_template,moving_image=parcel_t1w_wm_ni,
                                    interpolator_mode="genericLabel",transform=transform_list)
    parcel_gm_mni_img  = reg.transform(fixed_image=mni_template,moving_image=parcel_t1w_gm_ni,
                                    interpolator_mode="genericLabel",transform=transform_list)
    parcel_mni_img     = reg.transform(fixed_image=mni_template,moving_image=parcel_t1w_ni,
                                    interpolator_mode="genericLabel",transform=transform_list)
    parcel_mni_img_nii      = nib.Nifti1Image(parcel_mni_img.numpy(), mni_template.affine)
    parcel_mni_gm_img_nii   = nib.Nifti1Image(parcel_gm_mni_img.numpy(), mni_template.affine)
    parcel_mni_wm_img_nii   = nib.Nifti1Image(parcel_wm_mni_img.numpy(), mni_template.affine)

    parcellation_data_np    = parcel_mni_img_nii.get_fdata()
    parcellation_data_gm_np = parcel_mni_gm_img_nii.get_fdata()
    parcellation_data_wm_np = parcel_mni_wm_img_nii.get_fdata()
    ftools.save_nii_file(parcellation_data_np,mni_template.header,join(OUTDIR,f"parcellation_mi152_{PARC_SCHEME}-WM_{INCLUDE_WM}.nii.gz"))
    ############## GET indices ###############
    feature4d_path = mridata.data["featured4d"][PARC_SCHEME]["path"]
    debug.info("feature4d_path",feature4d_path)
    label_indices  = np.load(feature4d_path)["label_indices"]
    ################ QMASK ####################
    BIDS_ROOT_PATH = join(dutils.DATAPATH,GROUP)
    qmask_dir      = join(BIDS_ROOT_PATH,"derivatives","group","qmask")
    qmask_path     = join(qmask_dir,f"group_space-mni_acq-qmask_desc-CrPCr_spectroscopy.nii.gz")
    qmask_pop      = nib.load(qmask_path)
    n_voxel_counts_dict = parc.count_voxels_inside_parcel(qmask_pop.get_fdata(), parcellation_data_np, label_indices)
    ignore_rows = list()
    accept_rows = list()
    debug.success("parcellation_data_np",np.unique(parcellation_data_np).astype(int).shape)
    debug.success("labels_all",len(labels_all))
    debug.success("label_indices",label_indices.shape)
    debug.success("n_voxel_counts_dict keys",len(n_voxel_counts_dict.keys()))
    sys.exit()

    for i,index in enumerate(n_voxel_counts_dict):
        # if index <3000:
        #     index = index
        # else:
        #     continue
        cov = n_voxel_counts_dict[index]
        if cov < 0.69:
            ignore_rows.append(i)
            debug.warning(index,labels_all[index],cov)
        else:
            debug.success(index,labels_all[index],cov)
            accept_rows.append(i)
    sys.exit()


    ################ Select quality recordings
    quality_list_path  = join(dutils.DATAPATH,GROUP,"mrsi_quality_check.json")
    with open(quality_list_path,"r") as f:
        quality_list = json.load(f)
    ################# Load simmatrices ###################
    simatrix_dir_path = join(OUTDIR,"simmatrix",GROUP)
    filename = f"group-{GROUP}_atlas-{PARC_SCHEME}_desc-simmatrix_WM_{int(INCLUDE_WM)}.npz"
    data     = np.load(join(simatrix_dir_path,filename))
    weighted_metab_sim    = data["weighted_metab_sim"]
    simmatrix_sessions    = data["session_arr"]
    simmatrix_subject_ids = data["subject_id_arr"]
    #######################################################


    homotopy_list   = list()
    parcel_concentrations5D   = list()

    recording_list = np.array(ftools.list_recordings())
    debug.info(recording_list.shape)
    subject_id_arr = recording_list[:,0]
    debug.info(subject_id_arr)
    subject_id_arr = np.unique(subject_id_arr)
    recording      = recording_list[42]
    count_hom      = 0
    count_simm     = 0
    weighted_metab_sim_avg = np.zeros(weighted_metab_sim[0].shape)
    # for idm,recording in enumerate(recording_list):
    for idm,recording in enumerate(subject_id_arr):
        subject_id = recording
        # subject_id,session=recording
        # if quality_list[subject_id][session]["spectroscopy"]<=0.5:
        #     continue
        if quality_list[subject_id]["V1"]["spectroscopy"]==-1:
            if quality_list[subject_id]["V2"]["spectroscopy"]==-1:
                continue
            else:
                session = "V2"
        else:
            session = "V1"
        # subject_id,session = "S048","V1"
        prefix = f"sub-{subject_id}_ses-{session}"
        mridata  = MRIData(subject_id,session,group=GROUP)
        mridata.load_homotopy(include_wm=INCLUDE_WM)
        feature4d_path = mridata.data["featured4d"][PARC_SCHEME]["path"]
        if feature4d_path==0:continue
        debug.info("load homotopy for ",prefix)
        data = np.load(feature4d_path)
        feature4d = data["features4D"]
        if count_hom==0:features4D_popavg = feature4d
        else:features4D_popavg       += feature4d
        count_hom+=1
        ######### Create average simmatrix #########
        sel_idx = np.where((simmatrix_sessions==session) & (simmatrix_subject_ids==subject_id))[0]
        if sel_idx==0:continue
        weighted_metab_sim_avg+=weighted_metab_sim[sel_idx[0]]
        count_simm+=1
        ######## Parcel Concentrations ############
        con_path = mridata.data["connectivity"]["spectroscopy"][PARC_SCHEME]["path"]
        if con_path==0:continue
        parcel_concentration = np.load(con_path)["parcel_concentrations"]
        parcel_concentrations5D.append(parcel_concentration)

    debug.info("Averaged",count_hom,"homotopy maps")
    debug.info("Averaged",count_simm,"simmatrices")
    parcel_concentrations5D = np.array(parcel_concentrations5D)
    debug.info("parcel_concentrations5D",parcel_concentrations5D.shape)

    features4D_popavg/=count_hom
    weighted_metab_sim_avg/=count_simm
    label_indices = data["label_indices"]

    label_indices           = np.delete(label_indices,ignore_rows)
    features4D_popavg       = np.delete(features4D_popavg,ignore_rows,axis=0)
    weighted_metab_sim_avg  = np.delete(weighted_metab_sim_avg,ignore_rows,axis=0)
    weighted_metab_sim_avg  = np.delete(weighted_metab_sim_avg,ignore_rows,axis=1)

    parcel_concentrations5D = np.delete(parcel_concentrations5D, ignore_rows, axis=1)
    debug.info("parcel_concentrations5D",parcel_concentrations5D.shape)
    parcel_concentrations4D = parcel_concentrations5D.mean(axis=0)



    outsubdir = join(OUTDIR,"homotypic_clusters",PARC_SCHEME)
    # n_clusters_list= [3,4]
    # for n_clusters in n_clusters_list:
    #     cluster_labels = netclust.cluster_all_algorithms(features4D_popavg,n_clusters=n_clusters)
    #     for clust_alg in cluster_labels.keys():
    #         projected_data_3D = simm.nodal_strength_map(cluster_labels[clust_alg]+1,parcellation_data_np,label_indices)
    #         outpath = join(outsubdir,f"homotopy_groupavg_4D_{PARC_SCHEME}-WM_{INCLUDE_WM}-clust_{clust_alg}-nclust_{n_clusters}.nii.gz")
    #         ftools.save_nii_file(projected_data_3D,mni_template.header,outpath)
    #         debug.success("saved to",outpath)


    # ######## Compute group homotopy from average simmatrix
    # features4D_group = simm.get_4D_feature_nodal_similarity(weighted_metab_sim_avg)
    # n_clusters_list= [3,4]
    # for n_clusters in n_clusters_list:
    #     cluster_labels = netclust.cluster_all_algorithms(features4D_group,n_clusters=n_clusters)
    #     for clust_alg in cluster_labels.keys():
    #         projected_data_3D = simm.nodal_strength_map(cluster_labels[clust_alg]+1,parcellation_data_np,label_indices)
    #         outpath = join(outsubdir,f"homotopy_simmatrixavg_4D_{PARC_SCHEME}-WM_{INCLUDE_WM}-clust_{clust_alg}-nclust_{n_clusters}.nii.gz")
    #         ftools.save_nii_file(projected_data_3D,mni_template.header,outpath)
    #         debug.success("saved to",outpath)

    ######## Compute group homotopy from concatenated simmatrices linear + quadratic
    outsubdir = join(OUTDIR,"homotypic_clusters",PARC_SCHEME,"simmatrixconcat_4D",f"WM_{INCLUDE_WM}")
    os.makedirs(outsubdir,exist_ok=True)
    features_ND_group = simm.get_feature_similarity(weighted_metab_sim_avg)
    binarized_matrix_1 = nba.binarize(weighted_metab_sim_avg[:,:,0], threshold=0.1, mode="posneg", threshold_mode="density", binarize=True)
    binarized_matrix_2 = nba.binarize(weighted_metab_sim_avg[:,:,1], threshold=0.1, mode="posneg", threshold_mode="density", binarize=True)
    binarized_matrix = np.zeros(weighted_metab_sim_avg.shape)
    binarized_matrix[:,:,0] = binarized_matrix_1
    binarized_matrix[:,:,1] = binarized_matrix_2
    features_ND_group_binarized = simm.get_feature_similarity(binarized_matrix)
    n_clusters_list= [2,3,4,5,6,7,8,9,10]
    for n_clusters in n_clusters_list:
        outsubdir_clust = join(outsubdir,f"nclusters_{n_clusters}")
        os.makedirs(outsubdir_clust,exist_ok=True)
        #### Weighted
        cluster_labels = netclust.cluster_all_algorithms(features_ND_group,n_clusters=n_clusters)
        for clust_alg in cluster_labels.keys():
            projected_data_3D = simm.nodal_strength_map(cluster_labels[clust_alg]+1,parcellation_data_np,label_indices)
            outpath = join(outsubdir_clust,f"homotopy_simmatrixconcat_4D_{PARC_SCHEME}-WM_{INCLUDE_WM}-clust_{clust_alg}-nclust_{n_clusters}.nii.gz")
            ftools.save_nii_file(projected_data_3D,mni_template.header,outpath)
            debug.success("saved to",outpath)
            # GM
            projected_data_3D = simm.nodal_strength_map(cluster_labels[clust_alg]+1,parcellation_data_gm_np,label_indices)
            outpath = join(outsubdir_clust,f"homotopy_simmatrixconcat_4D_{PARC_SCHEME}-WM_{INCLUDE_WM}-clust_{clust_alg}-nclust_{n_clusters}_GM.nii.gz")
            ftools.save_nii_file(projected_data_3D,mni_template.header,outpath)
            debug.success("saved to",outpath)
            # WM
            if INCLUDE_WM:
                projected_data_3D = simm.nodal_strength_map(cluster_labels[clust_alg]+1,parcellation_data_wm_np,label_indices)
                outpath = join(outsubdir_clust,f"homotopy_simmatrixconcat_4D_{PARC_SCHEME}-WM_{INCLUDE_WM}-clust_{clust_alg}-nclust_{n_clusters}_WM.nii.gz")
                ftools.save_nii_file(projected_data_3D,mni_template.header,outpath)
                debug.success("saved to",outpath)

        clust_alg = "monti"
        cluster_labels = netclust.cluster_all_algorithms(features_ND_group,n_clusters=n_clusters)[clust_alg]
        # Save cluster labels_all:
        outpath = join(outsubdir_clust,f"homotopy_simmatrixconcat_4D_{PARC_SCHEME}-WM_{INCLUDE_WM}-clust_{clust_alg}-nclust_{n_clusters}.npz")
        np.savez(outpath,label_indices=label_indices,
                 cluster_labels=cluster_labels+1,
                 weighted_metab_sim_avg=weighted_metab_sim_avg,
                 parcel_concentrations4D=parcel_concentrations4D)
         # get 3d map metabolite concentrations and cluster 
        n_clusters = max(cluster_labels)+1
        
        parcel_concentrations3D = np.zeros((n_clusters,)+parcel_concentrations4D.shape[1::])
        for i,cluster_label in enumerate(np.unique(cluster_labels)):
            ids = np.where(cluster_labels==cluster_label)[0]
            parcel_concentrations3D[i] = np.mean(parcel_concentrations4D[ids],axis=0)
        # cluster_correlations
        outpath = join(outsubdir_clust,f"homotopy_simmatrixconcat_4D_{PARC_SCHEME}-WM_{INCLUDE_WM}-clust_{clust_alg}-nclust_{n_clusters}.json")
        simm_dict = dict()
        for i in range(n_clusters):
            simm_dict[i+1] = {}
            # debug.info(parcel_concentrations3D[i])
            for j in range(i, n_clusters):
                _simm = netclust.cluster_correlations(parcel_concentrations3D, i,j)
                simm_dict[i+1][j+1] = _simm
        # Write dictionary to a file
        with open(outpath, 'w') as f:
            json.dump(simm_dict, f, indent=4)

        



if __name__ == "__main__":
    main()
