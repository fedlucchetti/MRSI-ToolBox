import numpy as np
from tools.datautils import DataUtils
from tools.debug import Debug
from os.path import join, split ,exists
import os , math
import matplotlib.pyplot as plt
from tools.filetools import FileTools
from graphplot.simmatrix import SimMatrixPlot
from connectomics.nettools import NetTools
import copy
from connectomics.network import NetBasedAnalysis
import numpy as np
import copy
import itertools
import networkx as nx
from scipy.linalg import eigh
import time, sys

NITERATIONS = 10000


dutils = DataUtils()
resultdir = join(dutils.ANARESULTSPATH,"connectomes_M_vs_S")
GROUP    = "Mindfulness-Project"
simplt   = SimMatrixPlot()
ftools   = FileTools(GROUP)
debug    = Debug()
nettools = NetTools()
nba      = NetBasedAnalysis()



data = np.load(join(resultdir,"simM_metab_struct.npz"))
for k in data.keys():print(k)
struct_con_arr  = data["density_con_arr"]
metab_con_arr   = data["metab_con_sp_arr"]
metab_pvalues   = data["metab_pvalues_sp"]
subject_id_arr  = data["subject_id_arr"]
session_arr     = data["session_arr"]

metab_con_arr[metab_pvalues>0.001] = 0


ids             = np.argsort(subject_id_arr)
subject_id_arr  = subject_id_arr[ids]
session_arr     = session_arr[ids]
metab_con_arr   = metab_con_arr[ids]

n_zeros_arr = list()
for i,sim in enumerate(metab_con_arr):
    n_zeros = len(np.where(sim==0)[0])
    debug.info(subject_id_arr[i],session_arr[i],n_zeros)
    n_zeros_arr.append(n_zeros)

n_zeros_arr = np.array(n_zeros_arr)
debug.info(n_zeros_arr.mean(),n_zeros_arr.std())
# plt.hist(n_zeros_arr)
# plt.show()
# sys.exit()

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
        debug.error(subject_id_arr[i],session_arr[i])


sys.exit()
metab_con_arr_refined = np.array(metab_con_arr_refined)
session_arr_ref       = np.array(session_arr_ref)
subject_id_arr_ref    = np.array(subject_id_arr_ref)

debug.success(f"retained {len(subject_id_arr_ref)} of {len(subject_id_arr)}")




def compute_laplacian_spectrum(matrix):
    G = nx.from_numpy_array(matrix)
    L = nx.laplacian_matrix(G).toarray()
    eigenvalues = np.linalg.eigvalsh(L)
    return eigenvalues

def binarize_matrix(matrix, threshold):
    binarized_matrix = copy.deepcopy(matrix)
    binarized_matrix[np.abs(matrix) < threshold] = 0
    binarized_matrix[np.abs(matrix) >= threshold] = np.sign(matrix[np.abs(matrix) >= threshold])
    return binarized_matrix

def monte_carlo_simulation(sim_matrices, iterations=1000):
    best_thresholds = None
    min_std = float('inf')
    threshold_range = np.linspace(0.60, 0.85, 1000)

    # for it in range(iterations):
    it = 0
    while True:
        try:
            thresholds = [np.random.choice(threshold_range) for _ in range(len(sim_matrices))]
            spectra = []
            
            for sim_matrix, threshold in zip(sim_matrices, thresholds):
                binarized_matrix = binarize_matrix(sim_matrix, threshold)
                spectrum = compute_laplacian_spectrum(binarized_matrix)
                spectra.append(spectrum)
            
            combined_spectra = np.vstack(spectra)
            std_deviation = np.std(combined_spectra, axis=0).mean()

            if std_deviation < min_std:
                min_std = std_deviation
                best_thresholds = thresholds
                debug.success(it,min_std)
            it+=1
            # debug.info("Elapsed",time.time()-start)
        except KeyboardInterrupt:
            break
    
    return best_thresholds, min_std

# Example usage
if __name__ == "__main__":
    # Assuming sim_matrices is a list of N 2D numpy arrays representing the similarity matrices
    sim_matrices = metab_con_arr_refined
    best_thresholds, min_std = monte_carlo_simulation(sim_matrices,NITERATIONS)
    best_thresholds = np.array(best_thresholds)
    debug.success("Best Thresholds: ", best_thresholds)
    debug.success("Minimum Standard Deviation: ", min_std)



    outpath = join(dutils.ANARESULTSPATH,f"{GROUP}_threshold_list.npz")
    if exists(outpath):
        old_min_std = np.load(outpath)["min_std"]
        if min_std < old_min_std:
             debug.info("Saving to file")
             np.savez(outpath,
             subject_id_arr  = subject_id_arr_ref,
             session_arr     = session_arr_ref,
             best_thresholds = best_thresholds,
             min_std         = min_std)
             debug.success("Done")

