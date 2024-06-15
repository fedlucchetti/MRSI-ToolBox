import numpy as np
from scipy.ndimage import label
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, PathPatch

from matplotlib.path import Path
import random

# Function to generate a random irregular shape
def generate_random_shape(center, max_radius, num_points):
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    np.random.shuffle(angles)
    radii = np.random.randint(1, max_radius, num_points)
    points = np.array([np.cos(angles) * radii, np.sin(angles) * radii]).T + center
    return points

# Generate a 50x50 binary array with all zeros
array_shape = (50, 50)
array = np.zeros(array_shape, dtype=int)

# Generate random irregular shaped clusters
num_clusters = 2
cluster_centers = [(random.randint(15, 35), random.randint(15, 35)) for _ in range(num_clusters)]
max_radius = 10
num_points = 10

for center in cluster_centers:
    shape_points = generate_random_shape(center, max_radius, num_points)
    path = Path(shape_points, closed=True)
    patch = PathPatch(path, facecolor='none')

    # Draw the path onto the array
    for x in range(array_shape[0]):
        for y in range(array_shape[1]):
            if path.contains_point([x, y]):
                array[x, y] = -1

# Convert to binary array (True for -1, False for others)
binary_array = array == -1

# Label the connected components
labeled_array, num_features = label(binary_array)

# Plot the original array
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].matshow(array, cmap='gray')
ax[0].set_title('Original Array')

# Plot the labeled array with red contours
ax[1].matshow(array, cmap='gray')
ax[1].set_title('Labeled Clusters')

# Draw red rectangles around the clusters
for i in range(1, num_features + 1):
    cluster_indices = np.argwhere(labeled_array == i)
    # Compute bounding box
    min_row, min_col = np.min(cluster_indices, axis=0)
    max_row, max_col = np.max(cluster_indices, axis=0)
    width = max_col - min_col + 1
    height = max_row - min_row + 1
    rect = Rectangle((min_col, min_row), width, height, fill=False, edgecolor='red', linewidth=2)
    ax[1].add_patch(rect)

# Show the plots
plt.show()

