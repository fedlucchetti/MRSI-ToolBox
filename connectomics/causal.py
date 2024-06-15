
from rich.progress import Progress,track
import numpy as np
from scipy.stats import fisher_exact





class CausalNN:
    def __init__(self):
        pass


    def get_neighbors(self,matrix, i, j, order):
        neighbors = []
        indices_list = []
        n1, n2 = matrix.shape
        if order>0:
            for di in range(-order, order + 1):
                for dj in range(-order, order + 1):
                    if di == 0 and dj == 0  or np.abs(di)+np.abs(dj) != order:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n1 and 0 <= nj < n2:
                        neighbors.append(matrix[ni, nj])
                        indices_list.append([ni, nj])
        else:
            ni, nj = i, j 
            neighbors.append(matrix[ni, nj])
            indices_list.append([ni, nj])
        return neighbors, indices_list

    # get_neighbors(matrix, 10,10,1)

    def calculate_p_value_matrix(self,S_matrices, M_matrices, order):
        N = len(S_matrices)
        n1, n2 = S_matrices[0].shape
        p_value_matrix = np.zeros((n1, n2))
        for i in track(range(n1), description="n1..."):
            for j in range(n2):
                S0M0, S0M1, S1M1, S1M0 = 0, 0, 0, 0
                neighbors,indices_list = self.get_neighbors(S_matrices[0], i, j, order)
                for k in range(N):
                    S = S_matrices[k]
                    M = M_matrices[k]
                    for i,j in indices_list:
                        if S[i, j] == 0 and M[i, j] == 0:
                            S0M0 += 1
                        elif S[i, j] == 0 and M[i, j] == 1:
                            S0M1 += 1
                        elif S[i, j] == 1 and M[i, j] == 1:
                            S1M1 += 1
                        elif S[i, j] == 1 and M[i, j] == 0:
                            S1M0 += 1
                # Construct contingency table
                contingency_table = np.array([[S0M0, S0M1], [S1M0, S1M1]])
                # Perform Fisher's exact test
                _, p_value = fisher_exact(contingency_table)
                p_value_matrix[i, j] = p_value
        return p_value_matrix