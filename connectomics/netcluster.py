import numpy as np
import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from scipy.spatial.distance import squareform
import community as community_louvain
import networkx as nx
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, Birch
from sklearn.mixture import GaussianMixture
from collections import Counter
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap, LocallyLinearEmbedding, TSNE
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.stats import mode
from scipy.stats import linregress, spearmanr
from connectomics.parcellate import Parcellate
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

METABOLITES     = ["NAANAAG", "Ins", "GPCPCh", "GluGln", "CrPCr"]
FONTSIZE        = 16

parc      = Parcellate()

class NetCluster(object):
    def __init__(self):
        pass

    def consensus_clustering(self,similarity_matrix, n_clusters, n_iterations=10):
        """
        Perform consensus clustering on a similarity matrix using multiple clustering algorithms.
        
        Parameters:
        - similarity_matrix: 2D numpy array representing the similarity matrix.
        - n_clusters: Number of clusters.
        - n_iterations: Number of clustering runs to perform for each algorithm.
        
        Returns:
        - consensus_labels: Consensus cluster labels for each data point.
        """
        n_samples = similarity_matrix.shape[0]
        all_labels = np.zeros((n_iterations * 4, n_samples), dtype=int)

        # Convert similarity matrix to distance matrix
        distance_matrix = 1 - similarity_matrix

        clustering_algorithms = [
            KMeans(n_clusters=n_clusters, n_init=1),
            AgglomerativeClustering(n_clusters=n_clusters),
            SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='discretize')
        ]

        idx = 0
        for alg in clustering_algorithms:
            for _ in range(n_iterations):
                if isinstance(alg, SpectralClustering):
                    labels = alg.fit_predict(similarity_matrix)
                else:
                    labels = alg.fit_predict(distance_matrix)
                all_labels[idx, :] = labels
                idx += 1

        # Add Louvain algorithm results
        for _ in range(n_iterations):
            # Convert similarity matrix to a weighted graph
            graph = nx.from_numpy_array(similarity_matrix)
            partition = community_louvain.best_partition(graph)
            labels = np.array([partition[i] for i in range(n_samples)])
            all_labels[idx, :] = labels
            idx += 1

        # Build the co-association matrix
        co_association_matrix = np.zeros((n_samples, n_samples), dtype=float)

        for i in range(n_samples):
            for j in range(n_samples):
                co_association_matrix[i, j] = np.sum(all_labels[:, i] == all_labels[:, j])

        co_association_matrix /= (n_iterations * (len(clustering_algorithms) + 1))

        # Perform hierarchical clustering on the co-association matrix
        condensed_dist_matrix = squareform(1 - co_association_matrix)
        linkage_matrix = sch.linkage(condensed_dist_matrix, method='average')
        consensus_labels = sch.fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')

        return consensus_labels



    def project_4d_to_1d(self,data, method='isomap'):
        """
        Project a 4D array onto a 1D array using specified manifold learning method.
        
        Parameters:
        data (np.ndarray): Input array of shape (N, 4).
        method (str): Method to use for projection. Options are:
                    'isomap', 'lle', 'hessian_lle', 'laplacian_eigenmaps',
                    'tsne', 'umap', 'pca', 'lda'
                    
        Returns:
        np.ndarray: Output array of shape (N, 1) after projection.
        """
        if method == 'isomap':
            model = Isomap(n_components=1)
        elif method == 'lle':
            model = LocallyLinearEmbedding(n_components=1)
        elif method == 'hessian_lle':
            model = LocallyLinearEmbedding(n_components=1, method='hessian')
        elif method == 'laplacian_eigenmaps':
            from sklearn.manifold import SpectralEmbedding
            model = SpectralEmbedding(n_components=1)
        elif method == 'tsne':
            model = TSNE(n_components=1)
        elif method == 'umap':
            model = UMAP(n_components=1)
        elif method == 'pca':
            model = PCA(n_components=1)
        elif method == 'lda':
            model = LDA(n_components=1)
        else:
            raise ValueError("Invalid method specified. Choose from 'isomap', 'lle', 'hessian_lle', 'laplacian_eigenmaps', 'tsne', 'umap', 'pca', 'lda'.")
        
        # Fit the model and transform the data
        transformed_data = model.fit_transform(data)
        
        return transformed_data
    

    def cluster_all_algorithms(self,X,n_clusters=3, random_state=42):
        """
        Clusters the given Nx4 dataset using all available clustering algorithms in scikit-learn.
        
        Parameters:
            X (numpy.ndarray or pandas.DataFrame): The Nx4 dataset to cluster.
            
        Returns:
            dict: A dictionary where keys are the names of clustering algorithms and values are the cluster labels.
        """
      
        # Initialize clustering algorithms
        clustering_algorithms = {
            # 'KMeans': KMeans(n_clusters=n_clusters, random_state=random_state),
            # 'AgglomerativeClustering': AgglomerativeClustering(n_clusters=n_clusters),
            'SpectralClustering': SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', 
                                                     random_state=random_state,n_init=100),
            # 'Birch': Birch(n_clusters=n_clusters),
            'GaussianMixture': GaussianMixture(n_components=n_clusters, random_state=random_state)
        }
        
        # Dictionary to store the cluster labels
        cluster_labels = {}
        
        # Fit each algorithm and get the cluster labels
        for name, algorithm in clustering_algorithms.items():
            try:
                if name == 'AgglomerativeClustering':
                    algorithm.fit(X)
                    labels = algorithm.fit_predict(X)
                else:
                    labels = algorithm.fit_predict(X)
                cluster_labels[name] = labels
            except Exception as e:
                cluster_labels[name] = f"Error: {e}"
    
        # cluster_labels["majority"] = self.majority_labels(cluster_labels)
        # cluster_labels["consensus"] = self.consensus_clustering(cluster_labels, n_clusters)
        cluster_labels["monti"]     = self.monti_consensus_clustering(cluster_labels, n_clusters)
        return cluster_labels


    def relabel_clusters(self,reference_labels, target_labels):
        """
        Relabels the target_labels to match the reference_labels as closely as possible.
        
        Parameters:
            reference_labels (numpy.ndarray): The reference cluster labels.
            target_labels (numpy.ndarray): The target cluster labels to relabel.
            
        Returns:
            numpy.ndarray: The relabeled target cluster labels.
        """
        new_labels = np.zeros_like(target_labels)
        unique_labels = np.unique(reference_labels)
        
        for label in unique_labels:
            mask = reference_labels == label
            if np.any(mask):  # Check if mask is not empty
                modal_result = mode(target_labels[mask])
                if modal_result.mode.size > 0 and modal_result.count.size > 0:
                    new_label = modal_result.mode[0]
                    new_labels[target_labels == new_label] = label
        return new_labels

    def majority_labels(self,cluster_results):
        """
        Determines the majority label for each data point across multiple clustering algorithms.
        
        Parameters:
            cluster_results (dict): A dictionary where keys are the names of clustering algorithms and
                                    values are the cluster labels.
                                    
        Returns:
            numpy.ndarray: An array of majority labels for each data point.
        """
        # Get the number of data points
        num_points = len(next(iter(cluster_results.values())))
        
        # Initialize a list to store majority labels
        majority_labels = []
        
        for i in range(num_points):
            # Collect labels for the ith data point from each algorithm
            labels = [labels[i] for labels in cluster_results.values() if not isinstance(labels, str)]
            
            # Determine the majority label
            most_common_label, _ = Counter(labels).most_common(1)[0]
            
            majority_labels.append(most_common_label)
        
        return np.array(majority_labels)


    def monti_consensus_clustering(self, cluster_labels, n_clusters):
        """
        Determine the consensus clustering using the Monti Consensur algorithm.
        
        Parameters:
            cluster_labels (dict): A dictionary where keys are algorithm names and values are the cluster labels.
            n_clusters (int): The number of clusters to find in the consensus.
            
        Returns:
            np.ndarray: The consensus clustering labels.
        """
        # Extract cluster labels from the dictionary
        clusterings = [labels for labels in cluster_labels.values() if isinstance(labels, np.ndarray)]
        n_samples = clusterings[0].shape[0]
        n_clusterings = len(clusterings)
        
        # Step 1: Create co-association matrix
        co_association_matrix = np.zeros((n_samples, n_samples))
        
        for clustering in clusterings:
            for i in range(n_samples):
                for j in range(n_samples):
                    if clustering[i] == clustering[j]:
                        co_association_matrix[i, j] += 1
        
        co_association_matrix /= n_clusterings
        
        # Step 2: Apply spectral clustering on the co-association matrix
        spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
        consensus_labels = spectral.fit_predict(co_association_matrix)
        
        return consensus_labels        

    def cluster_correlations(self,parcel_concentrations3D,cluster_x_id,cluster_y_id):
        x_array = parcel_concentrations3D[cluster_x_id].flatten()
        y_array = parcel_concentrations3D[cluster_y_id].flatten()
        res = spearmanr(x_array, y_array)
        res2 = parc.speanman_corr_quadratic(x_array, y_array)
        corr1 = res.statistic
        corr2 = round(res2["corr"],2)
        return corr1,corr2
    

    def optimal_gmm_clusters(self,X, max_clusters=10):
        """
        Determines the optimal number of clusters for a Gaussian Mixture Model using BIC and AIC.
        
        Parameters:
            X (numpy.ndarray or pandas.DataFrame): The dataset to cluster.
            max_clusters (int): The maximum number of clusters to consider.
            
        Returns:
            dict: A dictionary with keys 'n_clusters', 'bic', 'aic', and 'models', containing the optimal number of clusters,
                the BIC and AIC values for each number of clusters, and the fitted GMM models.
        """
       
        # Initialize lists to store BIC and AIC values
        bics = []
        aics = []
        models = []
        
        # Fit GMM for different numbers of clusters
        for n_clusters in range(1, max_clusters + 1):
            gmm = GaussianMixture(n_components=n_clusters, random_state=42)
            gmm.fit(X)
            bics.append(gmm.bic(X))
            aics.append(gmm.aic(X))
            models.append(gmm)
        
        # Determine the optimal number of clusters
        optimal_n_clusters_bic = np.argmin(bics) + 1
        optimal_n_clusters_aic = np.argmin(aics) + 1
        
        # Plot BIC and AIC values
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, max_clusters + 1), bics, label='BIC', marker='o')
        plt.plot(range(1, max_clusters + 1), aics, label='AIC', marker='o')
        plt.axvline(optimal_n_clusters_bic, color='r', linestyle='--', label=f'Optimal BIC: {optimal_n_clusters_bic}')
        plt.axvline(optimal_n_clusters_aic, color='g', linestyle='--', label=f'Optimal AIC: {optimal_n_clusters_aic}')
        plt.xlabel('Number of clusters')
        plt.ylabel('Criterion Value')
        plt.title('BIC and AIC for Gaussian Mixture Model')
        plt.legend()
        plt.show()
        
        return {
            'n_clusters_bic': optimal_n_clusters_bic,
            'n_clusters_aic': optimal_n_clusters_aic,
            'bic': bics,
            'aic': aics,
            'models': models
        }

# Example usage
if __name__ == "__main__":
    netclust = NetCluster()
    # Example similarity matrix (symmetric with 1s on the diagonal)
    similarity_matrix = np.array([
        [1.0, 0.8, 0.2, 0.3],
        [0.8, 1.0, 0.4, 0.5],
        [0.2, 0.4, 1.0, 0.6],
        [0.3, 0.5, 0.6, 1.0]
    ])

    n_clusters = 2
    consensus_labels = netclust.consensus_clustering(similarity_matrix, n_clusters)
    print("Consensus Cluster Labels:", consensus_labels)
