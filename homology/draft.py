import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt
import scipy.sparse.csgraph as csgraph
from os.path import join, split
from tools.datautils import DataUtils
import sys


dutils    = DataUtils()

resultdir = join(dutils.ANARESULTSPATH,"connectomes_M_vs_S")
####################################

data = np.load(join(resultdir,"simM_metab_struct.npz"))

# length_con_arr      = data["distance_matrix_arr"]
metab_con_arr          = data["metab_con_sp_arr"]
session_arr = data["session_arr"]
subject_id_arr = data["subject_id_arr"]

# Generate a random similarity matrix
np.random.seed(0)
similarity_matrix = metab_con_arr[2]
print(similarity_matrix.shape)
# similarity_matrix = np.random.rand(250, 250)
# similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2  # Make it symmetric

# Thresholds to consider
thresholds = np.linspace(0.5, 0.95, 100)

# Initialize lists to store Betti numbers
n_components_arr = []
n_cycles_arr = []
betti_0_intervals = []
betti_1_intervals = []
# Initialize lists to store Betti numbers
n_components_arr = []
n_cycles_arr = []
n_voids_arr = []

n_cycles_arr = np.zeros(thresholds.shape)
birth_death_pairs = []


# for idt, t in enumerate(thresholds):
#     # Binarize the similarity matrix based on the current threshold
#     adjacency_matrix = np.zeros(similarity_matrix.shape)
#     adjacency_matrix[np.abs(similarity_matrix) > t] = 1
#     adjacency_matrix = similarity_matrix
#     similarity_matrix = np.abs(similarity_matrix)+0.01

    




# Create a Rips complex from the adjacency matrix
print(np.nanmax(np.log(similarity_matrix+0.1)))
rips_complex = gd.RipsComplex(distance_matrix=np.sqrt(1-similarity_matrix), max_edge_length=0.95)
simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
# Compute the persistence
simplex_tree.persistence()

# Get the birth and death times of 1-dimensional features (cycles)
pairs = simplex_tree.persistence_intervals_in_dimension(1)
n_cycles = len([p for p in pairs if p[1] < np.inf])  # Only count cycles with finite death time
n_cycles_arr[0] = n_cycles

# Store birth and death times of cycles
for pair in pairs:
    if pair[1] < np.inf:  # Ensure it's a proper interval with finite death
        birth_death_pairs.append((pair[0], pair[1]))

# print(f"Threshold: {t}, Number of cycles: {n_cycles}")
# print(birth_death_pairs)

# Plot the number of cycles as a function of the threshold
fig, axs = plt.subplots(1, 2, figsize=(16, 8))  # Adjust size as necessary
axs[0].plot(thresholds, n_cycles_arr, label='Number of Cycles (Betti-1)')
axs[0].set_xlabel('Threshold')
axs[0].set_ylabel('Count')
axs[0].legend()

# Plot the birth and death intervals of cycles as horizontal bars
for i, (birth, death) in enumerate(birth_death_pairs):
    axs[1].hlines(y=i, xmin=birth, xmax=death, colors='b', linestyles='-', lw=2)
    print(birth,death)

axs[1].set_xlabel('Threshold')
axs[1].set_ylabel('Cycle index')
axs[1].set_title('Birth and Death Intervals of Cycles (Betti-1)')
axs[1].invert_yaxis()  # Invert y-axis to have cycles sorted from bottom to top

plt.show()