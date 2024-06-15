import numpy as np
import matplotlib.pyplot as plt
from tools.datautils import DataUtils
from tools.debug import Debug

import cv2, sys
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from tqdm import tqdm

utils    = DataUtils()
debug    = Debug()


class RidgeDetect(object):
    def __init__(self) -> None:
        pass


    def proc(self,image, sigma=1.0):
        """
        Detect ridges in the image using the Hessian matrix method.
        Parameters:
        - image: Grayscale image in which to detect ridges.
        - sigma: Scale at which ridges are detected.
        Returns:
        - Tuple of images (maxima_ridges, minima_ridges).
        """


        # Compute the Hessian matrix
        H_elems = hessian_matrix(image, sigma=sigma, order='rc')

        # Extract the eigenvalues (ridge information)
        eigval1, eigval2, eigval3 = hessian_matrix_eigvals(H_elems)

        return eigval1, eigval2, eigval3



    def normalize_image(self,image):
        norm_img = (image - np.min(image)) / (np.max(image) - np.min(image))
        norm_img = (image - np.min(image)) / (np.max(image) - np.min(image)) 
        return norm_img,np.min(image),np.max(image)
    
    def rescale_image(self,image,min_v,max_v):
        return image * (max_v-min_v)+min_v

    def apply_threshold(self,image, window=[0.3,0.7]):
        image,min_v,max_v = self.normalize_image(image)
        mask = (image >= window[0]) & (image <= window[1])
        image[mask] = 255
        image[~mask] = 0
        return self.rescale_image(image,min_v,max_v)
    

if __name__=="__main__":
    ridge = RidgeDetect()
    # utils.list_nii()
    # debug.info(utils.nii_files)
    images,_,fileids = utils.load_all_orig_res()
    debug.info("images.shape",images.shape)
    # images = images[0:3]
    
    eigen_val_A = np.zeros(images.shape)
    eigen_val_B = np.zeros(images.shape)
    eigen_val_C = np.zeros(images.shape)
    maxima_ridges, minima_ridges = np.zeros(images.shape),np.zeros(images.shape)
    for idm,image in enumerate(tqdm(images)):
        image = np.nan_to_num(image)
        eigen_val_A[idm], eigen_val_B[idm], eigen_val_C[idm] = ridge.proc(image)
    debug.info("eigen_val_A.shape",eigen_val_A.shape)
    debug.info("eigen_val_B.shape",eigen_val_B.shape)
    debug.info("eigen_val_C.shape",eigen_val_C.shape)
    debug.success("Done")
    utils.save_nii_files("OrigResEigenA",fileids,eigen_val_A)
    utils.save_nii_files("OrigResEigenB",fileids,eigen_val_B)
    utils.save_nii_files("OrigResEigenC",fileids,eigen_val_C)
    




        
