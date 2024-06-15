
from tools.filetools import FileTools
from tools.debug import Debug
import os , glob
import shutil
from os.path import split, join
import numpy as np
import nibabel as nib
from scipy.sparse import csgraph

ft     = FileTools()
debug  = Debug()


class NetTools:
    def __init__(self) -> None:
        pass


    def laplacian_spectrum(self,bin_matrix):
        """
        Computes the Laplacian spectrum of a binarized similarity matrix.

        Parameters:
            sim_matrix (np.ndarray): The input similarity matrix.
            binarization_threshold (float): Threshold to binarize the similarity matrix.
            plot_spectrum (bool): If True, plots the Laplacian spectrum.

        Returns:
            np.ndarray: The Laplacian spectrum (sorted eigenvalues).
        """

        # Compute the Laplacian matrix
        laplacian_matrix = csgraph.laplacian(bin_matrix, normed=False)

        # Compute the eigenvalues of the Laplacian matrix (Laplacian spectrum)
        eigenvalues = np.linalg.eigvalsh(laplacian_matrix)
        return eigenvalues

    def merge_matrix_elements(self,simmatrix, merge_dict):
        # Determine new matrix size
        n = simmatrix.shape[0]
        indices_to_merge = sorted({idx for details in merge_dict.values() for idx in details['merge']})
        new_indices_count = n - len(indices_to_merge) + len(merge_dict)
        
        # Create a new matrix of reduced size
        new_matrix = np.zeros((new_indices_count, new_indices_count))
        
        # Map old indices to new indices
        new_index_map = {}
        reduced_index = 0
        for i in range(n):
            if i in indices_to_merge:
                continue
            new_index_map[i] = reduced_index
            reduced_index += 1

        # Update new indices for merged groups
        for key, value in merge_dict.items():
            merge_indices = value['merge']
            new_index_map[min(merge_indices)] = reduced_index
            for idx in merge_indices:
                if idx != min(merge_indices):
                    new_index_map[idx] = reduced_index
            reduced_index += 1

        # Fill the new matrix
        for i in range(n):
            new_i = new_index_map[i]
            for j in range(n):
                new_j = new_index_map[j]
                new_matrix[new_i, new_j] += simmatrix[i, j]
        return new_matrix

    def construct_metabolic_simmilarity(self,metab_con_arr,metab_pvalues,ALPHA=0.05,threshold=0.69):
        """
        Constructs a metabolic similarity matrix from correlation arrays and p-values associated
        with the metabolic connections. It generates both a merged continuous correlation matrix
        and a binary correlation matrix based on significance and a threshold value.
        Returns:
        - tuple (numpy.ndarray, numpy.ndarray):
        - The first array is the binary metabolic correlation matrix.
        - The second array is the continuous (mean) metabolic correlation matrix.
        """
        n_subjects = metab_con_arr.shape[0]
        n_labels   = metab_con_arr.shape[-1]
        metab_con_merged     = np.zeros([n_subjects,n_labels,n_labels])
        metab_con_bin_merged = np.zeros([n_subjects,n_labels,n_labels])
        for ids,metab_con_subj in enumerate(metab_con_arr):
            
            metab_con_subj[metab_pvalues[ids]< ALPHA] = np.sign(metab_con_subj[metab_pvalues[ids]< ALPHA])
            metab_con_subj[metab_pvalues[ids]> ALPHA] = 0
            if len(metab_con_subj.shape)==3:
                metab_con_merged[ids]   = np.tanh(np.mean(np.arctanh(metab_con_subj),axis=0))
                _metab_con_bin_merged   = np.mean(metab_con_subj,axis=0)
            else:
                metab_con_merged        = metab_con_subj
                _metab_con_bin_merged   = metab_con_subj
            corr_sign = np.sign(_metab_con_bin_merged[np.abs(_metab_con_bin_merged)>=threshold])
            _metab_con_bin_merged[np.abs(_metab_con_bin_merged)<threshold] = 0
            _metab_con_bin_merged[np.abs(_metab_con_bin_merged)>=threshold] = corr_sign
            metab_con_bin_merged[ids] = _metab_con_bin_merged
        return metab_con_bin_merged, metab_con_merged


    def compute_joint_probability_distributions(self,metab_bin, struc_bin):
        """
        Compute the four joint probability distributions (PDFs) between metabolic and structural binarized matrices.

        Parameters:
            metab_bin (np.ndarray): Array of shape (K, 251, 251) containing K binarized metabolic matrices.
            struc_bin (np.ndarray): Array of shape (K, 251, 251) containing K binarized structural matrices.

        Returns:
            tuple of np.ndarray: Four matrices of shape (251, 251) containing the probabilities:
                                - prob(metab_bin=1, struc_bin=1)
                                - prob(metab_bin=1, struc_bin=0)
                                - prob(metab_bin=0, struc_bin=1)
                                - prob(metab_bin=0, struc_bin=0)
        """
        # Check if the input arrays have the same shape
        if metab_bin.shape != struc_bin.shape:
            raise ValueError("Both input arrays must have the same shape (K, 251, 251).")
        # Check if the input arrays have the correct shape
        if len(metab_bin.shape) != 3 or metab_bin.shape[1] != metab_bin.shape[2]:
            raise ValueError("Input arrays must have shape (K, 251, 251).")
        # Extract dimensions
        K, N, _ = metab_bin.shape
        # Initialize the probability matrices
        prob_11 = np.zeros((N, N))
        prob_10 = np.zeros((N, N))
        prob_01 = np.zeros((N, N))
        prob_00 = np.zeros((N, N))
        # Iterate through all pairs of matrices
        for k in range(K):
            mask_11 = (metab_bin[k] == 1) & (struc_bin[k] == 1)
            mask_10 = (metab_bin[k] == 1) & (struc_bin[k] == 0)
            mask_01 = (metab_bin[k] == 0) & (struc_bin[k] == 1)
            mask_00 = (metab_bin[k] == 0) & (struc_bin[k] == 0)
            prob_11 += mask_11
            prob_10 += mask_10
            prob_01 += mask_01
            prob_00 += mask_00
        # Calculate the joint probabilities
        prob_11 /= K
        prob_10 /= K
        prob_01 /= K
        prob_00 /= K
        return prob_11, prob_10, prob_01, prob_00



