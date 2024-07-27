import numpy as np
from scipy.stats import pearsonr
# from scipy.stats import spearmanr as pearsonr

from statsmodels.stats.multitest import multipletests
from scipy import stats
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score, 
                             adjusted_mutual_info_score, homogeneity_score, 
                             completeness_score, v_measure_score)
from scipy.stats import permutation_test

metric_functions = {
    'ARI': adjusted_rand_score,
    'NMI': normalized_mutual_info_score,
    'AMI': adjusted_mutual_info_score,
    'Homogeneity': homogeneity_score,
    'Completeness': completeness_score,
    'V-Measure': v_measure_score,
}

class NetRobustness(object):
    def __init__(self):
        pass
    # Function to calculate the average of N KxK matrices
    def average_matrices(self,matrices):
        return np.mean(matrices, axis=0)

    # Function to calculate the average of N KxK matrices
    def average_matrices(self,matrices):
        return np.mean(matrices, axis=0)

    # Function to apply Fisher's Z-transformation
    def fisher_z_transform(self,r):
        return 0.5 * np.log((1 + r) / (1 - r))

    # Function to apply inverse Fisher's Z-transformation
    def inverse_fisher_z_transform(self,z):
        return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)

    # Function to calculate edge weight correlations
    def edge_weight_correlations(self,matrices, avg_matrix):
        correlations = []
        p_values = []
        
        flat_avg_matrix = avg_matrix.flatten()
        
        for matrix in matrices:
            # Flatten the matrices to get edge weights
            flat_matrix = matrix.flatten()
            
            # Calculate the Pearson correlation coefficient and p-value
            r, p = pearsonr(flat_matrix, flat_avg_matrix)
            correlations.append(self.fisher_z_transform(r))
            p_values.append(p)
        
        # Adjust p-values for multiple comparisons using Benjamini-Hochberg FDR
        _, corrected_p_values, _, _ = multipletests(p_values, method='fdr_bh')
        
        avg_z = np.mean(correlations)
        avg_correlation = self.inverse_fisher_z_transform(avg_z)
        std_z =  np.std(correlations)
        std_correlation = self.inverse_fisher_z_transform(std_z)
        overall_p_value = np.mean(corrected_p_values)
        
        return avg_correlation, std_correlation, overall_p_value

    # Function to calculate nodal similarities
    def nodal_similarity(self,matrix):
        K = matrix.shape[0]
        similarities = np.zeros(K)
        for i in range(K):
            # Sum the weights of connections from node i to all other nodes
            sum_of_weights = np.sum(matrix[i, :]) - matrix[i, i]  # Exclude self-connection
            # Compute the average weight
            similarities[i] = sum_of_weights / (K - 1)
        return similarities
        


    # Function to calculate nodal similarity correlations
    def nodal_similarity_correlations(self,matrices, avg_matrix):
        avg_similarity = self.nodal_similarity(avg_matrix)
        flat_avg_similarity = avg_similarity.flatten()
        
        correlations = []
        p_values = []
        
        for matrix in matrices:
            similarity = self.nodal_similarity(matrix)
            # Flatten the similarities matrices to get the nodal similarities
            flat_similarity = similarity.flatten()
            
            # Skip if there are NaNs or infinite values in the similarities
            if np.isnan(flat_similarity).sum() > 0 or np.isnan(flat_avg_similarity).sum() > 0:
                continue
            if np.isinf(flat_similarity).sum() > 0 or np.isinf(flat_avg_similarity).sum() > 0:
                continue
            
            # Calculate the Pearson correlation coefficient and p-value
            r, p = pearsonr(flat_similarity, flat_avg_similarity)
            correlations.append(self.fisher_z_transform(r))
            p_values.append(p)
        
        if len(p_values)==1:
            return correlations[0],p_values[0]
        else:
            avg_correlation, std_correlation, overall_p_value = self.aggregate_stats(correlations,p_values)
            return avg_correlation, std_correlation, overall_p_value

    def compare_two_sets(self,matrices1, matrices2):
        if len(matrices1) != len(matrices2):
            raise ValueError("The two sets of matrices must have the same number of matrices.")
        
        correlations = []
        p_values = []
        
        for mat1, mat2 in zip(matrices1, matrices2):
            flat_mat1 = mat1.flatten()
            flat_mat2 = mat2.flatten()
            r, p = pearsonr(flat_mat1, flat_mat2)
            correlations.append(self.fisher_z_transform(r))
            p_values.append(p)
        
        if len(p_values)==1:
            return correlations[0],p_values[0]
        else:
            avg_correlation, std_correlation, overall_p_value = self.aggregate_stats(correlations,p_values)
            return avg_correlation, std_correlation, overall_p_value
    
    def aggregate_stats(self,correlations,p_values):
        if len(correlations)==1:
            return correlations,None,p_values
        # Adjust p-values for multiple comparisons using Benjamini-Hochberg FDR
        _, corrected_p_values, _, _ = multipletests(p_values, method='fdr_bh')
        avg_z = np.mean(correlations)
        avg_correlation = self.inverse_fisher_z_transform(avg_z)
        std_z =  np.std(correlations)
        std_correlation = self.inverse_fisher_z_transform(std_z)
        overall_p_value = np.mean(corrected_p_values)
        return avg_correlation, std_correlation, overall_p_value



    def compare_nodal_similarities(self,matrices1, matrices2):
        if len(matrices1) != len(matrices2):
            raise ValueError("The two sets of matrices must have the same number of matrices.")
        
        correlations = []
        p_values = []
        
        for mat1, mat2 in zip(matrices1, matrices2):
            similarity1 = self.nodal_similarity(mat1)
            similarity2 = self.nodal_similarity(mat2)
            
            # Flatten the similarities to get the nodal similarities
            flat_similarity1 = similarity1.flatten()
            flat_similarity2 = similarity2.flatten()
            
            # Check for NaN or infinite values
            if np.isnan(flat_similarity1).sum() > 0 or np.isnan(flat_similarity2).sum() > 0:
                continue
            if np.isinf(flat_similarity1).sum() > 0 or np.isinf(flat_similarity2).sum() > 0:
                continue
            
            # Calculate the Pearson correlation coefficient and p-value
            r, p = pearsonr(flat_similarity1, flat_similarity2)
            correlations.append(self.fisher_z_transform(r))
            p_values.append(p)
        
        # Adjust p-values for multiple comparisons using Benjamini-Hochberg FDR
        _, corrected_p_values, _, _ = multipletests(p_values, method='fdr_bh')
        
        avg_z = np.mean(correlations)
        avg_correlation = self.inverse_fisher_z_transform(avg_z)
        std_z =  np.std(correlations)
        std_correlation = self.inverse_fisher_z_transform(std_z)
        overall_p_value = np.mean(corrected_p_values)
        
        return avg_correlation, std_correlation, overall_p_value

    def compute_voxel_wise_difference_and_ttest(self,array1, array2):
        """
        Computes the voxel-wise difference between two 3D arrays, performs a t-test, and returns the statistics.
        
        Parameters:
        array1 (np.ndarray): First 3D array.
        array2 (np.ndarray): Second 3D array.
        
        Returns:
        t_stat (np.ndarray): t-statistics for each voxel.
        p_values (np.ndarray): p-values for each voxel.
        """
        # Ensure the arrays have the same shape
        if array1.shape != array2.shape:
            raise ValueError("Input arrays must have the same shape")

        # Compute voxel-wise difference
        difference = array1 - array2
        
        # Compute mean and standard deviation of differences
        mean_diff = np.mean(difference, axis=-1)
        std_diff = np.std(difference, axis=-1, ddof=1)

        # Compute t-statistics
        t_stat = mean_diff / (std_diff / np.sqrt(array1.shape[-1]))

        # Compute p-values
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=array1.shape[-1] - 1))
        
        return t_stat, p_values

    def evaluate_clustering_agreement_with_permutation_test(self,labels1, labels2, metric='ARI', n_permutations=1000):
        if metric not in metric_functions:
            raise ValueError(f"Invalid metric '{metric}'. Choose from {list(metric_functions.keys())}.")
        
        # Calculate the observed score
        observed_score = metric_functions[metric](labels1, labels2)
        
        # Function to calculate the score for a permutation of labels2
        def permutation_score(labels1, labels2):
            return metric_functions[metric](labels1, np.random.permutation(labels2))

        # Perform the permutation test
        perm_test_result = permutation_test(
            data=[labels1, labels2],
            statistic=lambda x, y: metric_functions[metric](x, y),
            n_resamples=n_permutations,
            alternative='greater'
        )
        
        p_value = perm_test_result.pvalue

        return {
            'score': observed_score,
            'p-value': p_value
        }


    def fisher_method(self,p_values):
        """
        Applies Fisher's method to combine p-values.
        
        Parameters:
        p_values (np.ndarray): Array of p-values.
        
        Returns:
        combined_p (float): Combined p-value.
        """
        chi_square_stat = -2 * np.sum(np.log(p_values))
        combined_p = 1 - stats.chi2.cdf(chi_square_stat, 2 * len(p_values))
        return combined_p

    def aggregate_voxel_wise_comparisons(self,arrays1, arrays2):
        """
        Repeats voxel-wise difference computation and t-test over a set of comparisons,
        and aggregates the p-values using Fisher's method.
        
        Parameters:
        arrays1 (list of np.ndarray): List of first 3D arrays for comparisons.
        arrays2 (list of np.ndarray): List of second 3D arrays for comparisons.
        
        Returns:
        aggregated_p_values (np.ndarray): Aggregated p-values for each voxel.
        """
        if len(arrays1) != len(arrays2):
            raise ValueError("The lists of arrays must have the same length")
        
        all_p_values = []

        for array1, array2 in zip(arrays1, arrays2):
            _, p_values = self.compute_voxel_wise_difference_and_ttest(array1, array2)
            all_p_values.append(p_values)
        
        all_p_values = np.array(all_p_values)

        # Aggregate p-values using Fisher's method for each voxel
        aggregated_p_values = np.apply_along_axis(self.fisher_method, 0, all_p_values)
        
        return aggregated_p_values



# Example usage
if __name__ == "__main__":
    N = 10  # Number of matrices
    K = 5   # Size of each KxK matrix
    netrobust = NetRobustness()
    # Generate random MSN matrices for demonstration
    matrices1 = [np.random.rand(K, K) for _ in range(N)]
    matrices2 = [np.random.rand(K, K) for _ in range(N)]

    # Calculate the average matrix
    avg_matrix = netrobust.average_matrices(matrices1)

    # Calculate edge weight correlations
    avg_edge_corr, std_edge_corr, p_value_edge_corr = netrobust.edge_weight_correlations(matrices1, avg_matrix)
    print(f"Average Edge Weight Correlation:      {avg_edge_corr:.2f}, \t SD: {std_edge_corr:.2f},\t p-value: {p_value_edge_corr:.3f}")

    # Calculate nodal similarity correlations
    avg_nodal_corr, std_nodal_corr, p_value_nodal_corr = netrobust.nodal_similarity_correlations(matrices1, avg_matrix)
    print(f"Average Nodal Similarity Correlation: {avg_nodal_corr:.2f}, \t SD: {std_nodal_corr:.2f},\t p-value: {p_value_nodal_corr:.3f}")

    # Compare two sets of matrices
    avg_set_corr, std_set_corr, p_value_set_corr = netrobust.compare_two_sets(matrices1, matrices2)
    print(f"Average Set Correlation:              {avg_set_corr:.2f}, \t SD: {std_set_corr:.2f},\t p-value: {p_value_set_corr:.3f}")

    # Compare nodal similarities between two sets of matrices
    avg_nodal_set_corr, std_nodal_set_corr, p_value_nodal_set_corr = netrobust.compare_nodal_similarities(matrices1, matrices2)
    print(f"Average Nodal Set Correlation:        {avg_nodal_set_corr:.2f}, \t SD: {std_nodal_set_corr:.2f},\t p-value: {p_value_nodal_set_corr:.3f}")

