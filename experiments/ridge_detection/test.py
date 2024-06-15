import numpy as np
import matplotlib.pyplot as plt
from tools.datautils import DataUtils
import cv2
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

utils    = DataUtils()

def detect_ridges(image, sigma=1.0):
    """
    Detect ridges in the image using the Hessian matrix method.
    Parameters:
    - image: Grayscale image in which to detect ridges.
    - sigma: Scale at which ridges are detected.
    Returns:
    - Tuple of images (maxima_ridges, minima_ridges).
    """
    # Ensure the image is in grayscale
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute the Hessian matrix
    H_elems = hessian_matrix(image, sigma=sigma, order='rc')

    # Extract the eigenvalues (ridge information)
    maxima_ridges, minima_ridges = hessian_matrix_eigvals(H_elems)

    return maxima_ridges, minima_ridges



def normalize_image(image):
    """
    Normalize the image to the range 0-255.

    Parameters:
    - image: The input image.

    Returns:
    - Normalized image.
    """
    normalized = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    return normalized

def apply_threshold(image, threshold):
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image

def plot_ridges(original, sigma_values, maxima_images, minima_images):
    N = len(sigma_values)  # Number of sigma values
    plt.figure(figsize=(N * 5, 15))  # Adjust the figure size as needed

    for i in range(N):
        # Plotting original image in the first row
        plt.subplot(3, N, i + 1)
        plt.imshow(original, cmap='grey')
        plt.title(f'Original (Sigma={sigma_values[i]})')
        plt.axis('off')

        # Plotting maxima ridges in the second row
        plt.subplot(3, N, N + i + 1)
        plt.imshow(normalize_image(maxima_images[i]), cmap='summer')
        plt.title(f'Maxima Ridges (Sigma={sigma_values[i]})')
        plt.axis('off')

        # Plotting minima ridges in the third row
        threshold = 125
        img = maxima_images[i]
        img = normalize_image(img)
        A,B = 105, 150
        mask = (img >= A) & (img <= B)
        img[mask] = 255
        img[~mask] = 0

        plt.subplot(3, N, 2 * N + i + 1)
        # plt.imshow(img, cmap='hot')
        plt.imshow(minima_images[i], cmap='winter')
        plt.title(f'Minima Ridges (Sigma={sigma_values[i]})')
        plt.axis('off')

    plt.show()

sigma_values = [0.1,0.2,0.5, 1.0]

maxima_images = []
minima_images = []
titles = []
data = utils.load_nii(file_type="Basic",fileid=1)
image,_ = data
slice_img = np.nan_to_num(image[:,50,:])
for sigma in sigma_values:
    maxima_ridges, minima_ridges = detect_ridges(slice_img, sigma=sigma)
    maxima_images.append(np.array(maxima_ridges))
    minima_images.append(np.array(minima_ridges))

# normalized_slice = (slice_img - np.min(slice_img)) / (np.max(slice_img) - np.min(slice_img))
plot_ridges(slice_img, sigma_values, maxima_images, minima_images)
