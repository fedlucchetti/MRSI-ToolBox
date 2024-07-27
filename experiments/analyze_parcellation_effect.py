import numpy as np
from tools.datautils import DataUtils
from tools.debug import Debug
from os.path import join, split 
import os , math
import matplotlib.pyplot as plt
from tools.filetools import FileTools
from graphplot.simmatrix import SimMatrixPlot
from bids.mridata import MRIData
import nibabel as nib
from connectomics.nettools import NetTools
from connectomics.netcluster import NetCluster
from nilearn import plotting, image, datasets
import ants
import networkx as nx
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import copy, sys
from registration.registration import Registration
from connectomics.network import NetBasedAnalysis
from connectomics.parcellate import Parcellate
from connectomics.robustness import NetRobustness
from connectomics.simmilarity import Simmilarity
from graphplot.nodal_simmilarity import NodalSimilarity
from rich.console import Console
from rich.table import Table

dutils    = DataUtils()
OUTDIR    = join(dutils.ANARESULTSPATH,"PLOS")

GROUP     = "Mindfulness-Project"
simm      = Simmilarity()
simplt    = SimMatrixPlot()
reg       = Registration()
ftools    = FileTools(GROUP)
debug     = Debug()
nettools  = NetTools()
nba       = NetBasedAnalysis()
parc      = Parcellate()
netrobust = NetRobustness()
netclust  = NetCluster()
pltnodal  = NodalSimilarity()

############ Meta stat sign #########3
ALPHA            = 0.05
INCLUDE_WM       = True
n_recordings     = 102
CLUST_ALG        = "monti"
############ Labels ############

####################################


FONTSIZE        = 16


############ Load data ############
PARC_SCHEME     = "LFMIHIFIF-3"
n_clusters      = 5
INCLUDE_WM      = 1
outsubdir       = join(OUTDIR,"homotypic_clusters",PARC_SCHEME,"simmatrixconcat_4D",f"WM_{INCLUDE_WM}")
outpath         = join(outsubdir,f"homotopy_simmatrixconcat_4D_{PARC_SCHEME}-WM_{INCLUDE_WM}-clust_{CLUST_ALG}-nclust_{n_clusters}_GM.nii.gz")
data            = nib.load(outpath)
homotopy_scale3 = data.get_fdata()


PARC_SCHEME     = "LFMIHIFIF-2"
n_clusters      = 6
INCLUDE_WM      = 0
outsubdir       = join(OUTDIR,"homotypic_clusters",PARC_SCHEME,"simmatrixconcat_4D",f"WM_{INCLUDE_WM}")
outpath         = join(outsubdir,f"homotopy_simmatrixconcat_4D_{PARC_SCHEME}-WM_{INCLUDE_WM}-clust_{CLUST_ALG}-nclust_{n_clusters}_GM.nii.gz")
data            = nib.load(outpath)
homotopy_scale2 = data.get_fdata()

PARC_SCHEME     = "LFMIHIFIF-4"
n_clusters      = 5
INCLUDE_WM      = 0
outsubdir       = join(OUTDIR,"homotypic_clusters",PARC_SCHEME,"simmatrixconcat_4D",f"WM_{INCLUDE_WM}")
outpath         = join(outsubdir,f"homotopy_simmatrixconcat_4D_{PARC_SCHEME}-WM_{INCLUDE_WM}-clust_{CLUST_ALG}-nclust_{n_clusters}_GM.nii.gz")
data            = nib.load(outpath)
homotopy_scale4 = data.get_fdata()

PARC_SCHEME     = "aal"
n_clusters      = 6
INCLUDE_WM      = 1
CLUST_ALG       = "SpectralClustering"
outsubdir       = join(OUTDIR,"homotypic_clusters",PARC_SCHEME,"simmatrixconcat_4D",f"WM_{INCLUDE_WM}")
outpath         = join(outsubdir,f"homotopy_simmatrixconcat_4D_{PARC_SCHEME}-WM_{INCLUDE_WM}-clust_{CLUST_ALG}-nclust_{n_clusters}_GM.nii.gz")
data            = nib.load(outpath)
homotopy_aal    = data.get_fdata()

gm_mask         = datasets.load_mni152_gm_mask().get_fdata()
mask            = homotopy_scale2+homotopy_scale3+homotopy_scale4
mask[mask>0] = 1
clusters_sclale3 = homotopy_scale3[gm_mask==1].flatten()
clusters_sclale2 = homotopy_scale2[gm_mask==1].flatten()
clusters_sclale4 = homotopy_scale4[gm_mask==1].flatten()
clusters_aal     = homotopy_aal[gm_mask==1].flatten()


results = netrobust.evaluate_clustering_agreement_with_permutation_test(clusters_sclale3, clusters_sclale2, 
                                                                        metric='ARI',n_permutations=100)
score_scale2, pvalue_scale2 = results["score"],results["p-value"]

results = netrobust.evaluate_clustering_agreement_with_permutation_test(clusters_sclale3, clusters_sclale4, 
                                                                        metric='ARI',n_permutations=100)
score_scale4, pvalue_scale4 = results["score"],results["p-value"]

results = netrobust.evaluate_clustering_agreement_with_permutation_test(clusters_sclale3, clusters_aal, 
                                                                        metric='ARI',n_permutations=100)
score_aal, pvalue_aal = results["score"],results["p-value"]






# ################ Individual Homotopy difference
# for i,subject_id in enumerate(subject_id_arr):
#     session = session_arr[i]
#     mridata  = MRIData(subject_id,session,group=GROUP)
#     dir_path = split(mridata.data["connectivity"]["spectroscopy"][PARC_SCHEME]["path"])[0]
#     dir_path = dir_path.replace("connectomes","homotopy")
#     filename = f"sub-{subject_id}_ses-{session}_space-mni_atlas-{PARC_SCHEME}_desc-homotopy_WM_{int(INCLUDE_WM)}.nii.gz"
#     outpath  = join(dir_path,filename)




# Create the table
console = Console()
########################################################################
table = Table(title="Parcellation effect on Homotopy")
# Add columns
table.add_column("Parc Scheme", justify="center", style="cyan", no_wrap=True)
table.add_column("score", justify="center", style="magenta")
table.add_column("pvalue", justify="center", style="green")

table.add_row("LFMIHIFIF-2",f"{round(score_scale2,2)}", f"{round(pvalue_scale2,2)}")
table.add_row("LFMIHIFIF-4",f"{round(score_scale4,2)}", f"{round(pvalue_scale4,2)}")
table.add_row("AAL        ",f"{round(score_aal,2)}", f"{round(pvalue_aal,2)}")
# table.add_row("Geometric  ",f"{round(delta_geom,2)}", f"{round(sigma_geom,2)}")
console.print(table)

####################################
sys.exit()
weighted_metab_sim_list            = list()

recording_list = np.array(ftools.list_recordings())






# recording = recording_list[0]
PARC_SCHEMES = ["LFMIHIFIF-2","LFMIHIFIF-3","LFMIHIFIF-4"]
for idm,recording in enumerate(recording_list):
    subject_id,session=recording
    prefix = f"sub-{subject_id}_ses-{session}"
    debug.info("Processing",prefix)
    mridata       = MRIData(subject_id,session,group=GROUP)
    ################################
    for i,PARC_SCHEME in enumerate(PARC_SCHEMES):
        con_path      = mridata.data["connectivity"]["spectroscopy"][PARC_SCHEME]["path"]
        con_data      = np.load(con_path)
        homotopy_path = con_path.replace("connectomes","homotopy")
        homotopy_path = homotopy_path.replace("run-01_acq-memprage_","")
        homotopy_path = homotopy_path.replace("_simmatrix.npz","_homotopy.nii.gz")
        os.makedirs(split(homotopy_path)[0],exist_ok=True)
        sim_matrix    = con_data["simmatrix_sp"]
        p_values      = con_data["pvalue_sp"]
        label_indices = con_data["labels_indices"]
        sim_matrix[p_values>0.001] = 0
        homotopy_np = simm.get_homotopy(sim_matrix,parcellation_data_np_list[i],label_indices)
        ftools.save_nii_file(homotopy_np,mni_template.header,homotopy_path)

sys.exit()
################################
corr_32, _ , pvalue_23 = netrobust.compare_two_sets(nodal_map_3, nodal_map_2)
corr_34, _ , pvalue_34 = netrobust.compare_two_sets(nodal_map_3, nodal_map_4)
debug.success(recording,"3 vs 2",corr_32)
debug.success(recording,"3 vs 4",corr_34)
corr_32_arr.append(corr_32)
pvalue_32_arr.append(pvalue_23)
corr_34_arr.append(corr_34)
pvalue_34_arr.append(pvalue_34)


corr_32_arr,pvalue_32_arr,corr_34_arr,pvalue_34_arr = get_homotopy_pop()


compare_two_sets(matrices1, matrices2)

homotopy_list_2 = get_homotopy_pop("LFMIHIFIF-2")
homotopy_list_3 = get_homotopy_pop("LFMIHIFIF-3")
homotopy_list_4 = get_homotopy_pop("LFMIHIFIF-4")



