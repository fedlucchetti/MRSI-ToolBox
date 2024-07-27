import numpy as np
from tools.debug import Debug
import os , math, copy



debug    = Debug()


class Simmilarity:
    def __init__(self):
        pass

    def filter_sparse_matrices(self,matrix_list):
        n_zeros_arr = list()
        for i,sim in enumerate(matrix_list):
            n_zeros = len(np.where(sim==0)[0])
            n_zeros_arr.append(n_zeros)

        n_zeros_arr = np.array(n_zeros_arr)
        debug.info("0 nodal strength count",n_zeros_arr.mean(),"+-",n_zeros_arr.std())

        include_indices = list()
        exclude_indices = list()
        matrix_list_refined = list()

        for i,sim in enumerate(matrix_list):
            n_zeros = len(np.where(sim==0)[0])
            if n_zeros<n_zeros_arr.mean()+n_zeros_arr.std():
                matrix_list_refined.append(sim)
                include_indices.append(i)
            else:
                exclude_indices.append(i)
        return matrix_list_refined,include_indices,exclude_indices

    @staticmethod
    def nodal_strength_map(nodal_similarity_matrix, parcellation_data_np, label_indices):
        nodal_strength_map_np = np.zeros(parcellation_data_np.shape)
        
        # Create a dictionary to map label indices to nodal similarity values
        label_to_similarity = {label: similarity for label, similarity in zip(label_indices, nodal_similarity_matrix)}
        
        # Use vectorized approach to fill the nodal_strength_map_np
        for label, similarity in label_to_similarity.items():
            nodal_strength_map_np[parcellation_data_np == label] = similarity
        
        return nodal_strength_map_np


    def _rgb_to_grayscale_with_zero(self,R, B ):
        grayscale = 0.299 * R  + 0.114 * B
        return grayscale

    def _polar_to_angle(self,x,y):
        # Calculate the angle in radians
        theta = math.atan2(y, x)
        # Normalize the gle to [0, 1]
        theta_normalized = (theta + math.pi) / (2 * math.pi)
        # Scale the normalized angle to [0, 255]an
        scalar_value = 255 * theta_normalized
        return scalar_value

    def polar_to_angle(self,feature_x,feature_y):
        vectorized_to_scalar   = np.vectorize(self._polar_to_angle)
        return vectorized_to_scalar(feature_x,feature_y)

    def rgb_to_grayscale_with_zero(self,feature_vector):
        vectorized_to_scalar   = np.vectorize(self._rgb_to_grayscale_with_zero)
        return vectorized_to_scalar(feature_vector)

    def nodal_similarity(self,matrix):
        # Exclude the diagonal elements (self-connections) by setting them to zero
        np.fill_diagonal(matrix, 0)
        # Sum the weights of connections from each node to all other nodes
        similarities = np.sum(matrix, axis=1)
        return similarities

    def get_feature_nodal_similarity(self,simmatrix_matrix):
        simmatrix_pop_weighted_plus = copy.deepcopy(simmatrix_matrix)
        simmatrix_pop_weighted_neg  = copy.deepcopy(simmatrix_matrix)
        simmatrix_pop_weighted_plus[simmatrix_matrix<0] = 0
        simmatrix_pop_weighted_neg[simmatrix_matrix>0]  = 0
        NS_parcel_plus     = self.nodal_similarity(simmatrix_pop_weighted_plus)
        NS_parcel_neg      = self.nodal_similarity(simmatrix_pop_weighted_neg)
        # NS_map_plus        = self.nodal_strength_map(NS_parcel_plus,parcellation_data_np,label_indices)
        # NS_map_neg         = self.nodal_strength_map(NS_parcel_neg,parcellation_data_np,label_indices)
        # nodal_strength_map_np_scalar        = self.polar_to_angle(NS_map_plus, NS_map_neg)
        features     = np.zeros((NS_parcel_plus.shape)+(2,))
        features[:,0] = NS_parcel_plus
        features[:,1] = NS_parcel_neg
        return features

    def get_feature_similarity(self,simmatrix_matrix4D):
        features = np.hstack((simmatrix_matrix4D[:,:,0], simmatrix_matrix4D[:,:,1]))
        return features

    def get_4D_feature_nodal_similarity(self,weighted_metab_sim_4D_avg):
        features2D_1  = self.get_feature_nodal_similarity(weighted_metab_sim_4D_avg[:,:,0])
        features2D_2  = self.get_feature_nodal_similarity(weighted_metab_sim_4D_avg[:,:,1])
        features4D    = np.zeros((features2D_1.shape[0],)+(4,))
        features4D[:,0:2] = features2D_1
        features4D[:,2:4] = features2D_2
        return features4D


    def get_homotopy2(self,simmatrix_matrix,parcellation_data_np,label_indices):
        features_1 = self.get_homotopy(simmatrix_matrix[:,:,0],parcellation_data_np,label_indices)
        features_2 = self.get_homotopy(simmatrix_matrix[:,:,1],parcellation_data_np,label_indices)
        features_4d     = np.zeros((features_1.shape[0],)+(4,))
        features_4d[:,0] = features_1[:,0]
        features_4d[:,1] = features_1[:,1]
        features_4d[:,2] = features_2[:,0]
        features_4d[:,3] = features_2[:,1]
        return features_4d


