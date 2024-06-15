import numpy as np
import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
import community as community_louvain
import networkx as nx



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
