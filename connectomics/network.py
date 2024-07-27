import numpy as np
from tools.debug import Debug
import networkx as nx
import scipy.stats as stats
from scipy.stats import percentileofscore
import copy 
from collections import defaultdict, Counter
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse import csr_matrix
from sklearn.linear_model import RANSACRegressor
from scipy.stats import ttest_ind
from scipy.stats import norm
debug  = Debug()

class NetBasedAnalysis:
    def __init__(self) -> None:
        pass


    def get_rich_club_coefficient(self,correlation_matrix):
        """
        Compute the rich-club coefficient for a given correlation matrix.
        """
        # Convert the correlation matrix to a NetworkX graph
        G = nx.from_numpy_array(correlation_matrix)
        
        # Compute the rich-club coefficient
        rc = nx.rich_club_coefficient(G, normalized=False)
        return rc


    def modularize(self,matrix):
        permuted_ids = reverse_cuthill_mckee(csr_matrix(matrix))
        permuted_matrix = copy.deepcopy(matrix)
        permuted_matrix = permuted_matrix[permuted_ids,:]
        permuted_matrix = permuted_matrix[:,permuted_ids]
        return permuted_matrix, permuted_ids

    def binarize(self, simmatrix, threshold, mode="abs", threshold_mode="value", binarize=True):
        binarized = np.zeros(simmatrix.shape)
        
        if threshold_mode == "density":
            threshold = self.threshold_density(simmatrix, threshold)
        elif threshold_mode == "value":
            pass
        if mode == "posneg":
            if binarize:
                binarized[np.abs(simmatrix) < threshold] = 0
                binarized[np.abs(simmatrix) >= threshold] = np.sign(simmatrix[np.abs(simmatrix) >= threshold])
            else:
                binarized[np.abs(simmatrix) < threshold] = 0
                binarized[np.abs(simmatrix) >= threshold] = simmatrix[np.abs(simmatrix) >= threshold]
        elif mode == "abs":
            if binarize:
                binarized[np.abs(simmatrix) <= threshold] = 0
                binarized[np.abs(simmatrix) >= threshold] = 1
            else:
                binarized[np.abs(simmatrix) <= threshold] = 0
                binarized[np.abs(simmatrix) >= threshold] = simmatrix[np.abs(simmatrix) >= threshold]
        elif mode == "pos":
            binarized[simmatrix < threshold] = 0
            if threshold_mode == "density":
                threshold = self.threshold_density(simmatrix, threshold)
            if binarize:
                binarized[simmatrix >= threshold] = 1
            else:
                binarized[simmatrix >= threshold] = simmatrix[simmatrix >= threshold]
        elif mode == "neg":
            binarized[simmatrix > threshold] = 0
            if threshold_mode == "density":
                threshold = self.threshold_density(simmatrix, threshold)
            if binarize:
                binarized[simmatrix <= threshold] = 1
            else:
                binarized[simmatrix <= threshold] = simmatrix[simmatrix <= threshold]
        # debug.info(threshold)
        return binarized

    def threshold_density(self, matrix, density):
        """
        Computes the threshold value based on the top X% density weights.

        Parameters:
            matrix (np.ndarray): The input similarity matrix.
            density (float): The percentage density for thresholding (e.g., 0.02 for 2%).

        Returns:
            float: The computed threshold value.
        """
        if density < 0 or density > 1:
            raise ValueError("Density must be a value between 0 and 1.")
        
        # Flatten the matrix to sort and find the threshold value
        flattened = matrix.flatten()
        
        # Number of elements to include based on density
        num_elements = int(np.ceil(density * len(flattened)))
        
        # Find the threshold value
        threshold_value = np.partition(flattened, -num_elements)[-num_elements]
        
        return threshold_value

    
    def get_betweeness(self,adj_matrix):
        adj_G               = nx.from_numpy_array(adj_matrix)
        betweenness = nx.betweenness_centrality(adj_G)
        return betweenness
    
    def get_global_clustering_coef(self,adj_matrix,num_simulations=100):
        if isinstance(adj_matrix, np.ndarray):
            adj_G               = nx.from_numpy_array(adj_matrix)
        elif isinstance(adj_matrix, nx.Graph):
            pass
        global_clustering_coef_meas = nx.transitivity(adj_G)
        clustering_coefs = list()
        for _ in range(num_simulations):
            rand_G = self.generate_random_graph(adj_G)
            clustering_coefs.append(nx.transitivity(rand_G))
        
        clustering_coefs = np.array(clustering_coefs)
        p_value = ttest_ind([global_clustering_coef_meas], clustering_coefs, equal_var=False).pvalue[0]
        return global_clustering_coef_meas,np.mean(np.array(clustering_coefs)), p_value


    def get_average_path_length(self,adj_matrix,num_simulations=100):
        avg_path_length_meas = self.__get_average_path_length(adj_matrix)
        adj_G               = nx.from_numpy_array(adj_matrix)
        avg_path_coeffs = list()
        for _ in range(num_simulations):
            rand_G = self.generate_random_graph(adj_G)
            avg_path_coeffs.append(self.__get_average_path_length(rand_G))
        avg_path_coeffs = np.array(avg_path_coeffs)
        p_value = ttest_ind([avg_path_length_meas], avg_path_coeffs, equal_var=False).pvalue[0]
        return avg_path_length_meas,np.mean(np.array(avg_path_coeffs)),p_value


    def get_small_world(self,adj_matrix,mode="sigma"):
        if isinstance(adj_matrix, np.ndarray):
            adj_G               = nx.from_numpy_array(adj_matrix)
        elif isinstance(adj_matrix, nx.Graph):
            pass
        if mode=="omega":
            return nx.omega(adj_G)
        elif mode=="sigma":
            return nx.sigma(adj_G)
    def __get_average_path_length(self,adj_matrix):
        """
        Computes the global clustering coefficient and the weighted average shortest path length for a graph.
        
        Parameters:
        G (networkx.Graph): The input graph.
        
        Returns:
        tuple: A tuple containing the global clustering coefficient and the weighted average shortest path length.
        """
        if isinstance(adj_matrix, np.ndarray):
            G               = nx.from_numpy_array(adj_matrix)
        elif isinstance(adj_matrix, nx.Graph):
            G = adj_matrix
        # Identify the connected components
        components = [G.subgraph(c).copy() for c in nx.connected_components(G)]

        # Compute the average shortest path length for each component
        component_lengths = []
        component_sizes = []
        for component in components:
            if len(component) > 1:  # Path length is not defined for single-node components
                component_length = nx.average_shortest_path_length(component)
                component_lengths.append(component_length)
                component_sizes.append(len(component))

        # Compute the weighted average of the shortest path lengths
        if component_sizes:
            total_nodes = sum(component_sizes)
            weighted_avg_path_length = sum(l * s for l, s in zip(component_lengths, component_sizes)) / total_nodes
        else:
            weighted_avg_path_length = None

        return weighted_avg_path_length
            

    def get_rich_club_stats(self,adj_matrix):
        
        adj_G               = nx.from_numpy_array(adj_matrix)
        # Rich club coeffs
        observed_rich_club_coeff = nx.rich_club_coefficient(adj_G, normalized=False)
        rand_G                  = self.get_random_graphs(adj_matrix,N=1000)
        random_rich_club_coeffs = [nx.rich_club_coefficient(rg, normalized=False) for rg in rand_G]
        random_rc_arr           = np.array(list(random_rich_club_coeffs[0].values()))
        observed_rc_arr         = np.array(list((observed_rich_club_coeff.values())))[0:len(random_rc_arr)]
        # Statistical Significane
        p_values                 = dict()
        for k in observed_rich_club_coeff:
            random_coeffs_k  = np.array([rc[k] for rc in random_rich_club_coeffs if k in rc])
            observed_coeff_k = observed_rich_club_coeff[k]
            # Calculate the p-value for this degree k
            p_value     = (100 - percentileofscore(random_coeffs_k, observed_coeff_k, kind='strict')) / 100
            p_values[k] = p_value
        
        return observed_rich_club_coeff,random_rich_club_coeffs,p_values

    def degree_distribution(self,similarity_matrix):
        """
        Computes the degree distribution from a binarized similarity numpy array,
        considering -1 values as a special case.
        
        Parameters:
        similarity_matrix (numpy.ndarray): The input binary similarity matrix.

        Returns:
        dict: A dictionary where keys are degrees (including -1) and values are the number of nodes with that degree.
        """
        # Replace -1 with 0 to create a valid adjacency matrix for NetworkX
        adjacency_matrix = np.where(similarity_matrix == -1, 0, similarity_matrix)
        
        # Convert the numpy array to a NetworkX graph
        G = nx.from_numpy_array(adjacency_matrix)
        
        # Get the degree of each node
        degrees = [d for n, d in G.degree()]
        
        # Count the frequency of each degree, including -1 values
        degree_count = Counter(degrees)
        
        # Include the count of -1 values
        negative_ones_count = np.sum(similarity_matrix == -1)
        if negative_ones_count > 0:
            degree_count[-1] = negative_ones_count
        
        # Convert to dictionary
        degree_distribution = dict(degree_count)
        
        return degree_distribution

    def get_degree_per_node(self,similarity_matrix):
        adjacency_matrix = np.where(similarity_matrix == -1, 0, similarity_matrix)
        
        # Convert the numpy array to a NetworkX graph
        G = nx.from_numpy_array(adjacency_matrix)
        
        # Get the degree of each node
        degrees = [d for n, d in G.degree()]
        return degrees


    def clustering_coefficient_distribution(self,similarity_matrix):
        """
        Computes the clustering coefficients and corresponding degrees from a binarized 
        similarity numpy array, considering -1 values as a special case.
        
        Parameters:
        similarity_matrix (numpy.ndarray): The input binary similarity matrix.

        Returns:
        tuple: Two numpy arrays, X (degrees) and Y (clustering coefficients).
        """
        # Replace -1 with 0 to create a valid adjacency matrix for NetworkX
        adjacency_matrix = np.where(similarity_matrix == -1, 0, similarity_matrix)
        
        # Convert the numpy array to a NetworkX graph
        G = nx.from_numpy_array(adjacency_matrix)
        
        # Get the clustering coefficient of each node
        clustering_coeffs = nx.clustering(G)
        
        # Get the degree of each node
        degrees = dict(G.degree())
        
        # Extract degrees and clustering coefficients into arrays
        X = np.array([degrees[node] for node in G.nodes()])
        Y = np.array([clustering_coeffs[node] for node in G.nodes()])
        
        return X, Y



    def get_rc_distribution(self, simmatrix_binarized, threshold_degree=0.8):
        ######### RichClub ########
        # adjacency_matrix = np.where(simmatrix_binarized == -1, 1, simmatrix_binarized)
        adjacency_matrix = copy.deepcopy(simmatrix_binarized)
        np.fill_diagonal(adjacency_matrix, 0)
        G = nx.from_numpy_array(adjacency_matrix)
        reference_degrees = np.array(sorted(set(d for n, d in G.degree())))
        
        # Compute rich-club coefficient distribution for Metabolic network
        rc_coefficients = self.rich_club_coefficient_curve(G, reference_degrees)
        
        # Compute rich-club coefficient distribution for random network
        mean_random_rc, std_random_rc = self.rich_club_random_distribution(G, reference_degrees)
        
        degree_at_rc_1 = None
        p_values = []
        
        for degree, rc_coeff, mean_rc, std_rc in zip(reference_degrees, rc_coefficients, mean_random_rc, std_random_rc):
            # Compute z-score
            z_score = (rc_coeff - mean_rc) / std_rc
            
            # Compute p-value (two-tailed test)
            p_value = 2 * (1 - norm.cdf(abs(z_score)))
            p_values.append(p_value)
            
            # Check for degree where rc coefficient crosses the threshold and is significant
            if rc_coeff >= threshold_degree and p_value < 0.05 and degree_at_rc_1 is None:
                degree_at_rc_1 = degree
                
        
        return reference_degrees, rc_coefficients, mean_random_rc, std_random_rc, degree_at_rc_1, p_values

  


    def extract_subnetwork(self,simmatrix_binarized,degree_cutoff=None,parcel_indices=[]):
        
        adjacency_matrix = simmatrix_binarized.copy()  # Use a copy to avoid modifying the original matrix
        np.fill_diagonal(adjacency_matrix, 0)
        G = nx.from_numpy_array(adjacency_matrix)
        
        # Identify rich-club nodes
        if degree_cutoff is not None:
            subnetwork_node_indices = [n for n, d in G.degree() if d >= degree_cutoff]
        else: 
            subnetwork_node_indices = [n for n, d in G.degree()]
        # Compute degrees for the rich-club nodes
        subnetwork = G.subgraph(subnetwork_node_indices)
        degrees = dict(subnetwork.degree())
        
        # Convert degrees to a list in the same order as subnetwork_node_indices
        degrees_list = [degrees[node] for node in subnetwork_node_indices]
        if len(parcel_indices)>0:
            subnetwork_node_indices = parcel_indices[subnetwork_node_indices]
        return subnetwork_node_indices, degrees_list




    def rich_club_coefficient_curve(self,G, reference_degrees):
        """
        Computes the rich-club coefficient curve for a graph G, aligned to the reference degrees.
        
        Parameters:
        G (networkx.Graph): The input graph.
        reference_degrees (np.array): The reference degrees to align the rich-club coefficients.

        Returns:
        np.array: Aligned rich-club coefficients.
        """
        rc = nx.rich_club_coefficient(G, normalized=False)
        rc_coefficients = np.zeros_like(reference_degrees, dtype=float)
        for i, degree in enumerate(reference_degrees):
            rc_coefficients[i] = rc.get(degree, 0)
        return rc_coefficients

    def generate_random_graph(self,G):
        """
        Generates a random graph with the same degree sequence as G and removes self-loops and parallel edges.
        
        Parameters:
        G (networkx.Graph): The input graph.

        Returns:
        networkx.Graph: A random simple graph with the same degree sequence as G.
        """
        degree_sequence = [d for n, d in G.degree()]
        random_G = nx.configuration_model(degree_sequence)
        
        # Convert to a simple graph (removing self-loops and parallel edges)
        random_G = nx.Graph(random_G)  # This removes parallel edges
        random_G.remove_edges_from(nx.selfloop_edges(random_G))  # This removes self-loops
        
        return random_G

    def rich_club_random_distribution(self,G, reference_degrees, num_random_graphs=50):
        """
        Computes the rich-club coefficient curves for random equivalent models of G, aligned to the reference degrees.
        
        Parameters:
        G (networkx.Graph): The input graph.
        reference_degrees (np.array): The reference degrees to align the rich-club coefficients.
        num_random_graphs (int): Number of random equivalent models to generate.

        Returns:
        tuple: Two numpy arrays, mean coefficients, and std coefficients.
        """
        random_curves = []
        
        for _ in range(num_random_graphs):
            random_G = self.generate_random_graph(G)
            coefficients = self.rich_club_coefficient_curve(random_G, reference_degrees)
            random_curves.append(coefficients)
        
        random_curves = np.array(random_curves)
        mean_coefficients = np.nanmean(random_curves, axis=0)
        std_coefficients = np.nanstd(random_curves, axis=0)
        
        return mean_coefficients, std_coefficients
  

    def edge_distance(self,matrix,length_matrix):
        indices_all = np.argwhere(matrix == 1)
        # Calculate the differences of their indices
        # distances_all = indices_all[:, 0]-indices_all[:, 1]
        distances_all = length_matrix[indices_all[:, 0],indices_all[:, 1]]
        distances_all = distances_all[distances_all!=-1]

        distances_all = distances_all[distances_all!=-1]
        distance_pdf,bins    = np.histogram(distances_all,bins="auto")
        X_meas = (bins[:-1] + bins[1:]) / 2
        if len(np.where(distance_pdf==0))>0:
            distance_pdf=distance_pdf+0.1
        X_fit,y_pred_huber,params = self.lin_regression(X_meas,np.log(distance_pdf+0.1))
        return distances_all,X_meas, X_fit,y_pred_huber,params

    def lin_regression(self,x,y):
        params = dict()
        ransac = RANSACRegressor()
        ransac.fit(x.reshape(-1, 1), y)
        X_fit        = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
        y_pred_huber = ransac.predict(X_fit)
        params["intercept"] = ransac.estimator_.coef_[0]
        params["slope"]     = ransac.estimator_.intercept_
        params["r_squared"] = ransac.score(x.reshape(-1, 1), y)
        return X_fit,y_pred_huber,params

    


    