import numpy as np
from nilearn import plotting, image, datasets
import nibabel as nib
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import copy 

from graphplot.colorbar import ColorBar

colorbar = ColorBar()

class NodalSimilarity():
    def __init__(self):
        pass

    def color_bars(self,color_bar="blueblackred"):
        if color_bar=="blueblackred":
            colors = [
                (0, "blue"),  # Blue for lowest values
                (0.5, "black"),  # Black for middle values
                (1, "red")  # Red for highest values
            ]
        elif color_bar=="redblackblue":
            colors = [
                (0, "red"),  # Blue for lowest values
                (0.5, "black"),  # Black for middle values
                (1, "blue")  # Red for highest values
            ]
        elif color_bar=="bluewhitered":
            colors = [
                (0, "blue"),  # Blue for lowest values
                (0.5, "white"),  # Black for middle values
                (1, "red")  # Red for highest values
            ]
        elif color_bar=="redwhiteblue":
            colors = [
                (0, "red"),  # Blue for lowest values
                (0.5, "white"),  # Black for middle values
                (1, "blue")  # Red for highest values
            ]
        elif color_bar=="vwo":
            colors = [
                (0, "darkviolet"),  # Blue for lowest values
                (0.5, "white"),  # Black for middle values
                (1, "darkorange")  # Red for highest values
            ]
        elif color_bar=="owv":
            colors = [
                (0, "darkorange"),  # Blue for lowest values
                (0.5, "white"),  # Black for middle values
                (1, "darkviolet")  # Red for highest values
            ]
        else: 
            return color_bar
        return LinearSegmentedColormap.from_list("custom_cmap", colors)

    # Function to apply Fisher's Z-transformation
    def fisher_z_transform(self,r):
        return 0.5 * np.log((1 + r) / (1 - r))

    # Function to apply inverse Fisher's Z-transformation
    def inverse_fisher_z_transform(self,z):
        return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)

    def nodal_similarity(self,matrix):
        K = matrix.shape[0]
        similarities = np.zeros(K)
        for i in range(K):
            # Sum the weights of connections from node i to all other nodes
            sum_of_weights = np.sum(matrix[i, :]) - matrix[i, i]  # Exclude self-connection
            # Compute the average weight
            similarities[i] = sum_of_weights
        return similarities

    def compute_higher_order_correlations(self,matrix):
        """
        Compute higher-order correlations based on the matrix M.

        Parameters:
        - M: numpy array, symmetric matrix representing pairwise correlations between nodes.

        Returns:
        - M_pp: numpy array of size n, where M_pp[i] is the sum of all positive-positive correlations for node i.
        - M_mm: numpy array of size n, where M_mm[i] is the sum of all negative-negative correlations for node i.
        - M_pm: numpy array of size n, where M_pm[i] is the sum of all positive-negative correlations for node i.
        - M_mp: numpy array of size n, where M_mp[i] is the sum of all negative-positive correlations for node i.
        """
        M = copy.deepcopy(matrix)
        n = M.shape[0]
        M_pp = np.zeros(n)
        M_mm = np.zeros(n)
        M_pm = np.zeros(n)
        M_mp = np.zeros(n)
        
        for i in range(n):
            for j in range(n):
                if i != j and M[i, j] > 0:
                    M_pp[i] += M[i, j]
                    M_pm[i] += M[i, j]
                elif i != j and M[i, j] < 0:
                    M_mm[i] -= M[i, j]
                    M_mp[i] -= M[i, j]
                for k in range(n):
                    if i != j and i != k and j != k:
                        if M[i, j] > 0 and M[j, k] > 0:
                            M_pp[i] += M[i, k] + M[j, k]
                        elif M[i, j] < 0 and M[i, k] < 0 and M[j, k] < 0:
                            M_mm[i] += -M[i, k] - M[j, k]
                        elif M[i, j] > 0 and M[i, k] > 0 and M[j, k] < 0:
                            M_pm[i] += M[i, k] - M[j, k]
                        elif M[i, j] < 0 and M[i, k] < 0 and M[j, k] > 0:
                            M_mp[i] += -M[i, k] + M[j, k]
        
        return M_pp, M_mm, M_pm, M_mp



    def nodal_strength_map(self,nodal_similarity_matrix,parcellation_data_np,label_indices):
        nodal_strength_map_np = np.zeros(parcellation_data_np.shape)
        # Fill the similarity map with the similarity values
        for i, value in enumerate(label_indices):
            try:
                nodal_strength_map_np[parcellation_data_np == value] = nodal_similarity_matrix[i]  # Assuming parcel indices start from 1
            except:pass
        # Convert to Nifti1Image
        return nodal_strength_map_np
        


    def __plot(self, parcel_mni_img_nii, nodal_similarity_matrix, label_indices, output_file,vmin=None,vmax=None,colormap="jet",slices=[-5,-13,0]):
        colormap                = colorbar.bars(colormap)
        mni_template            = datasets.load_mni152_template()
        parcellation_data_np    = parcel_mni_img_nii.get_fdata()
        if vmin is None:
            vmin=np.min(nodal_similarity_matrix)
        if vmax is None:
            vmax=np.max(nodal_similarity_matrix)

        # Map strength to parcellation image
        nodal_strength_map_np = self.nodal_strength_map(nodal_similarity_matrix,parcellation_data_np,label_indices)
        # Convert to Nifti1Image
        similarity_img = nib.Nifti1Image(nodal_strength_map_np, mni_template.affine)
        # Save the resulting image (Optional)
        slice_str_arr = ["x", "y", "z"]
        titles = ["Sagittal", "Coronal", "Axial"]
        bar_flag      = False
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, slice_str in enumerate(slice_str_arr):
            # slice = round(slices[i]/2)-similarity_img.shape[i]
            if i==2:bar_flag=True
            plotting.plot_stat_map(similarity_img, cmap=colormap, 
                                    vmin=vmin,
                                    vmax=vmax,
                                    bg_img=mni_template,
                                    cut_coords=[slices[i]],
                                    display_mode=slice_str,
                                    colorbar=bar_flag,
                                    axes=axes[i],
                                    annotate=False)
            axes[i].set_title(slices[i])
        if output_file:
            fig.savefig(f"{output_file}")
        return nodal_strength_map_np
    

    def plot(self, image3D, output_file,fig=None, axes=[],vmin=None,vmax=None,colormap="jet",slices=[-5,-13,0]):
        colormap                = colorbar.bars(colormap)
        mni_template            = datasets.load_mni152_template()
        if vmin is None:
            vmin=np.min(image3D.min())
        if vmax is None:
            vmax=np.max(image3D.max())
        # Convert to Nifti1Image
        similarity_img = similarity_img
        # Save the resulting image (Optional)
        slice_str_arr = ["x", "y", "z"]
        titles = ["Sagittal", "Coronal", "Axial"]
        bar_flag      = False
        if len(axes) == 0:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, slice_str in enumerate(slice_str_arr):
            # slice = round(slices[i]/2)-similarity_img.shape[i]
            if i==2:bar_flag=True
            plotting.plot_stat_map(similarity_img, cmap=colormap, 
                                    vmin=vmin,
                                    vmax=vmax,
                                    bg_img=mni_template,
                                    cut_coords=[slices[i]],
                                    display_mode=slice_str,
                                    colorbar=bar_flag,
                                    axes=axes[i],
                                    annotate=False)
            axes[i].set_title(slices[i])
        if output_file:
            fig.savefig(f"{output_file}")
        return fig,axes

