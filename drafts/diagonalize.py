import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cosine
import networkx as nx

# Example adjacency matrices for two networks
A1 = np.array([[0, 1, 0, 0],
               [1, 0, 1, 1],
               [0, 1, 0, 1],
               [0, 1, 1, 0]])

A2 = np.array([[0, 1, 1, 0],
               [1, 0, 0, 1],
               [1, 0, 0, 1],
               [0, 1, 1, 0]])

def laplacian_matrix(A):
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    return L

# Compute Laplacian matrices
L1 = laplacian_matrix(A1)
L2 = laplacian_matrix(A2)

# Compute eigenvalues and eigenvectors of L1
eigvals1, eigvecs1 = eigh(L1)

# Diagonalize L1 to verify
L1_diag = eigvecs1.T @ L1 @ eigvecs1
print("Diagonalized L1 (V1^T L1 V1):\n", L1_diag)

# Transform L2 using the eigenvector matrix of L1
L2_transformed = eigvecs1.T @ L2 @ eigvecs1
print("Transformed L2 (V1^T L2 V1):\n", L2_transformed)

# Compute eigenvalues of the transformed L2
eigvals_transformed = np.linalg.eigvals(L2_transformed)
eigvals_transformed = np.sort(eigvals_transformed)
print("Eigenvalues of transformed L2:", eigvals_transformed)

# Compute eigenvalues of the original L2
eigvals2 = np.linalg.eigvals(L2)
eigvals2 = np.sort(eigvals2)
print("Eigenvalues of original L2:", eigvals2)

# Compute the difference between the eigenvalues
eigenvalue_difference = np.abs(eigvals_transformed - eigvals2)
print("Difference between eigenvalues of transformed and original L2:", eigenvalue_difference)

# Optional: compute a summary statistic, e.g., the sum of absolute differences
sum_of_differences = np.sum(eigenvalue_difference)
print("Sum of absolute differences:", sum_of_differences)
