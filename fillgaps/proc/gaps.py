import numpy as np
import random
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata
from scipy.stats import binom
from scipy.stats import binomtest

from tools.debug import Debug
debug = Debug()


class Gaps(object):
    def __init__(self):
        pass

    def determine_gap(self,tensors_qmask,p=0.5, alpha=0.31):
        # Assuming 'tensors' is a 4D numpy array: (num_tensors, dim1, dim2, dim3)
        count = np.sum(tensors_qmask, axis=0)
        n     = tensors_qmask.shape[0]
        threshold = int(tensors_qmask.shape[0]/2)
        while True:
            pv=binomtest(threshold, n, p).pvalue
            print(threshold,pv)
            if  pv< alpha:
                break
            else:
                threshold+=1
        debug.info("Binary cutoff at",threshold)
        return (count > threshold).astype(bool)

    def elementwise_agreement(self,tensor1, tensor2):
        """
        Performs an element-wise agreement comparison between two 3D tensors.

        Args:
        tensor1: A 3D numpy array.
        tensor2: A 3D numpy array.

        Returns:
        result: A 3D numpy array, where each element is 1 if the corresponding elements
                in tensor1 and tensor2 agree, and 0 if they disagree.
        """
        result = np.invert(np.logical_xor(tensor1, tensor2)).astype(int)

        return result
    
    def compute_tensor_holes(self,tensors_qmask,tensors_basic):
        """
        Computes the final tensors where holes are representetd as -1
        by first determining the population qmask (1s above alpha)
        then determing agreement between population qmask and individual qmasks
        then multiplying that agreement with the its rrespective tensor_basic
        and replacing NaN values by -1

        Args:
        tensors_qmask: 4D numpy tensor, loaded from Qmask.nii
        tensor_q_basic: 4D numpy tensor, laoded from Basic.nii

        Returns:
        tensors_basic_holes

        """
        tensors_basic_holes = np.zeros(tensors_basic.shape)
        tensor_qmask_uni= self.determine_gap(tensors_qmask,p=0.5, alpha=0.31)
        for idt, tensor in enumerate(tensors_basic):
            _tensor = self.elementwise_agreement(tensor,tensor_qmask_uni)
            _tensor = _tensor * tensors_basic[idt]
            _tensor[np.isnan(_tensor)] = -1
            tensors_basic_holes[idt] = _tensor
        return tensors_basic_holes

        

    def elementwise_or(self,tensor4d):
        result = tensor4d[0]
        for tensor3d in tensor4d:
            result = np.bitwise_or(result,tensor3d)
        return result

    def dfs(self,matrix, x, y, z, visited):
        # Base conditions
        if (x < 0 or x >= len(matrix) or y < 0 or y >= len(matrix[0]) or z < 0 or z >= len(matrix[0][0])
                or visited[x][y][z] or matrix[x][y][z] != 0):
            return 0

        # Mark the voxel as visited
        visited[x][y][z] = True

        # Count the current voxel
        count = 1

        # Check all adjacent voxels (6-connectivity)
        for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
            count += self.dfs(matrix, x + dx, y + dy, z + dz, visited)

        return count

    def count_clusters(self,matrix):
        visited       = np.zeros_like(matrix, dtype=bool)
        cluster_sizes = []

        for x in tqdm(range(len(matrix))):
            for y in range(len(matrix[0])):
                for z in range(len(matrix[0][0])):
                    if matrix[x][y][z] == 0 and not visited[x][y][z]:
                        cluster_size = self.dfs(matrix, x, y, z, visited)
                        cluster_sizes.append(cluster_size)
        
        return cluster_sizes

    def plot_cluster_sizes(self,cluster_sizes):
        counter = Counter(cluster_sizes)
        sizes, counts = zip(*sorted(counter.items()))

        plt.bar(sizes, counts)
        plt.xlabel('Cluster Size',fontsize=16)
        plt.ylabel('Number of 0-Clusters',fontsize=16)
        plt.title('Number of 0-Clusters vs. Cluster Size')
        plt.show()

    def create_cluster(self,tensor, K):
        """
        Creates a cluster of size K within a 3D tensor, setting the values of voxels
        in the cluster to 0. The cluster is formed by contiguous voxels, starting
        from a random non-NaN voxel.

        Parameters:
        tensor (numpy.ndarray): A 3D tensor (numpy array) in which the cluster is to be created.
                                The tensor should contain float values and may include NaNs.
        K (int): The size of the cluster to be created. This is the number of voxels
                that will be set to 0.

        Returns:
        numpy.ndarray: A new 3D tensor with the same shape as the input `tensor`, 
                    where a cluster of size K has been set to 0.

        Notes:
        - The function creates a copy of the input tensor to avoid modifying it in place.
        - The starting voxel for the cluster is chosen randomly from non-NaN voxels.
        - The function uses a depth-first search approach to grow the cluster.
        - The function returns early if it's not possible to grow the cluster to size K.
        """
        tensor_holes = tensor.copy()
        dims = tensor_holes.shape
        cluster = set()
        directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

        ############### Function to get valid neighbors ###############
        def get_neighbors(voxel):
            x, y, z = voxel
            neighbors = []
            for dx, dy, dz in directions:
                nx, ny, nz = x + dx, y + dy, z + dz
                if 0 <= nx < dims[0] and 0 <= ny < dims[1] and 0 <= nz < dims[2]:
                    neighbors.append((nx, ny, nz))
            return neighbors

        ############### Start from a random voxel ###############
        # start_voxel = (random.randint(0, dims[0]-1), random.randint(0, dims[1]-1), random.randint(0, dims[2]-1))
        # non_nan_coords = np.argwhere(~np.isnan(tensor))
        # start_voxel = tuple(non_nan_coords[np.random.choice(len(non_nan_coords))])

        non_zero_non_nan_coords = np.argwhere((~np.isnan(tensor_holes)) & (tensor_holes != 0))
        # debug.info("non_zero_non_nan_coords",len(non_zero_non_nan_coords),"K",K)
        # Make sure there are valid voxels to choose from
        while True:
            if len(non_zero_non_nan_coords) > 0:
                start_voxel = tuple(non_zero_non_nan_coords[np.random.choice(len(non_zero_non_nan_coords))])
                break
            else:
                pass
                # debug.warning("No valid non-zero, non-NaN voxels found in the tensor. K=",K)
        
        cluster.add(start_voxel)

        ############### Grow the cluster ###############
        while len(cluster) < K:
            # Randomly select a voxel from the cluster to grow from
            grow_voxel = random.choice(list(cluster))
            neighbors = get_neighbors(grow_voxel)
            random.shuffle(neighbors)
            
            for neighbor in neighbors:
                if neighbor not in cluster and tensor_holes[neighbor] != 0:
                    # Add neighbor to the cluster only if the tensor value at neighbor is not 0
                    cluster.add(neighbor)
                    break
            else:
                # If no valid neighbors can be added, try a different voxel in the next iteration
                continue
            if len(cluster) >= K:
                break


        ############### Set the cluster voxels to 0 ###############
        for voxel in cluster:
            tensor_holes[voxel] = -1
        return tensor_holes
    
    def create_dataset(self,tensors,nHoles=100):
        inputs                 = np.zeros(tensors.shape)
        labels                 = np.zeros(tensors.shape)
        mean_val               = np.nanmean(tensors)
        for idt, _tensor in enumerate(tqdm(tensors)):
            nan_mask          = np.isnan(_tensor)
            _tensor[nan_mask] = mean_val
            _tensor[nan_mask] = 0
            inputs[idt]       = self.create_cluster(_tensor,K=nHoles)
            labels[idt]       = _tensor
        return inputs, labels


    def find_bounds(self,tensor, margin_percent=20):
        """Find bounds of the region containing -1 and add a margin."""
        indices = np.argwhere(tensor == -1)
        if indices.size == 0:
            return None  # No -1 found

        min_bounds = indices.min(axis=0)
        max_bounds = indices.max(axis=0)


        # Calculate margin as a percentage of the tensor's dimensions
        margin = np.array([int(tensor.shape[i] * margin_percent / 100) for i in range(3)])

        # Adjust bounds with the margin
        min_bounds = np.maximum(min_bounds - margin, 0)
        max_bounds = np.minimum(max_bounds + margin, np.array(tensor.shape) - 1)

        return tuple(map(int, min_bounds)), tuple(map(int, max_bounds))

    def interpolate_region(self,tensor, min_bounds, max_bounds):
        """Interpolate values within a specific region of the tensor using nearest-neighbor interpolation."""
        region = tensor[min_bounds[0]:max_bounds[0]+1, min_bounds[1]:max_bounds[1]+1, min_bounds[2]:max_bounds[2]+1]

        # Replace NaN values with 0
        region[np.isnan(region)] = 0

        # Prepare points for interpolation
        x, y, z = np.mgrid[min_bounds[0]:max_bounds[0]+1, min_bounds[1]:max_bounds[1]+1, min_bounds[2]:max_bounds[2]+1]
        valid_mask = region != -1

        # Coordinates of valid and missing points
        valid_points = np.column_stack((x[valid_mask], y[valid_mask], z[valid_mask]))
        values = region[valid_mask]
        missing_points = np.column_stack((x[~valid_mask], y[~valid_mask], z[~valid_mask]))

        # Interpolate the missing points using nearest-neighbor method
        interpolated_values = griddata(valid_points, values, missing_points, method='nearest')

        # Replace missing values in the region
        region[~valid_mask] = interpolated_values

        return region

    def interpolate_missing_values(self,tensor, margin_percent=20):

        """Interpolate missing values in the tensor."""
        interpolated_tensor = tensor.copy()
        bounds = self.find_bounds(tensor, margin_percent)
        if bounds is None:
            debug.warning("No -1 values to interpolate")
            return tensor  # No -1 values to interpolate

        min_bounds, max_bounds = bounds
        interpolated_region = self.interpolate_region(tensor, min_bounds, max_bounds)
        interpolated_tensor[min_bounds[0]:max_bounds[0]+1, min_bounds[1]:max_bounds[1]+1, min_bounds[2]:max_bounds[2]+1] = interpolated_region

        return interpolated_tensor



    def reconstruction_error(self,original_tensor, interpolated_tensor, reference_tensor):
        """
        Compute the reconstruction error between the interpolated tensor and a reference tensor.

        Args:
        original_tensor: The original tensor with missing values.
        interpolated_tensor: The tensor with interpolated values.
        reference_tensor: The reference tensor for comparison.

        Returns:
        error: The reconstruction error.
        """
        # Mask to identify the originally missing values and exclude NaN values
        missing_mask = (original_tensor == -1) & ~np.isnan(reference_tensor)

        # Check if the masked array slice is empty
        if np.any(missing_mask):
            # Calculate error only on the originally missing values, ignoring NaNs
            error = np.nanmean((interpolated_tensor[missing_mask] - reference_tensor[missing_mask]) ** 2)
        else:
            # Return NaN or some other appropriate value when no computation is possible
            error = np.nan

        return error

