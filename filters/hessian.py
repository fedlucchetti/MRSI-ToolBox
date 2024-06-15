import numpy as np
import matplotlib.pyplot as plt
from tools.datautils import DataUtils
from tools.debug import Debug
import copy, os
import cv2 as cv
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from tqdm import tqdm
import SimpleITK as sitk
from scipy.stats import wasserstein_distance
from scipy.ndimage import sobel
import nibabel as nib
dutils    = DataUtils()
debug    = Debug()

class Hessian(object):
    def __init__(self) -> None:
        pass


    def sobel_edge_3d(self,image_3d):
        """
        Apply the 
         
           edge detector to a 3D image.

        :param image_3d: A 3D numpy array representing the image.
        :return: A 3D numpy array representing the gradient magnitude.
        """
        # Apply the Sobel filter along each axis
        dx = sobel(image_3d, axis=0, mode='constant')
        dy = sobel(image_3d, axis=1, mode='constant')
        dz = sobel(image_3d, axis=2, mode='constant')

        # Compute the gradient magnitude
        grad_magnitude = np.sqrt(dx**2 + dy**2 + dz**2)
        return grad_magnitude

    def edges(self,image,level=4):
        image = sitk.GetImageFromArray(image.astype(np.float32))
        sigma=image.GetSpacing()[0]
        feature_img = sitk.GradientMagnitude(image)
        return sitk.GetArrayFromImage(feature_img)

    def sharpen(self,image):
        input_image = sitk.GetImageFromArray(image.astype(np.float32))
        # Create the Laplacian sharpening filter
        sharpening_filter = sitk.LaplacianSharpeningImageFilter()

        # Apply the filter to the input image
        sharpened_img=  sharpening_filter.Execute(input_image)
        return sitk.GetArrayFromImage(sharpened_img)
    
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



    def normalize_image(self,image,return_min_max=True):
        norm_img = (image - np.min(image)) / (np.max(image) - np.min(image))
        norm_img = (image - np.min(image)) / (np.max(image) - np.min(image)) 
        if return_min_max:
            return norm_img,np.min(image),np.max(image)
        else: return norm_img
    
    def rescale_image(self,image,min_v,max_v):
        return image * (max_v-min_v)+min_v

    def apply_threshold(self,image, window=[0.3,0.7]):
        image,min_v,max_v = self.normalize_image(image)
        mask = (image >= window[0]) & (image <= window[1])
        image[mask] = 255
        image[~mask] = 0
        return self.rescale_image(image,min_v,max_v)


def get_cluster_mrsi(subject_id,lipid_rem,metabolic):
    cluster_img_path = os.path.join(dutils.DATAPATH,
                                    "MindfullTeen_cluster",
                                    subject_id,"Lipid"+"{:02d}".format(lipid_rem),
                                    "Results_" + subject_id,"MRSI_Nifti",
                                    "OrigRes_" + metabolic+"_conc.nii.gz")
    return nib.load(cluster_img_path).get_fdata()
    

if __name__=="__main__":  
    hessian = Hessian()
    FONTSIZE = 20
    # METABOLITE= "Glu+Gln"
    METABOLITE= "Glu+Gln"
    SUBJECT_ID ="S005_V1"
    # METABOLITE= "Cr+PCr"
    # METABOLITE= "GPC+PCh"
    LIPREM = 8
    SLICE    = [9,10]
    # SLICE = [:,:,11]
    images_list=list()

    # tensors_basic, _ ,_     = dutils.load_nii_all("Basic")
    # tensors_qmask, _,_      = dutils.load_nii_all("Qmask")
    # img,_  = dutils.load_nii(file_type="Conc",fileid=1,metabolic_str=None,normalization=False,rawnii=False)
    # mask,_ = dutils.load_nii(file_type="Qmask",fileid=1,metabolic_str=None,normalization=False,rawnii=False)
    # img[np.isnan(img)] = 0
    # mask[np.isnan(mask)] = 0
    # debug.info(img.shape,img.mean())
    

    mask,_                = dutils.load_origres_nii(SUBJECT_ID,str(LIPREM),"WaterSignal")
    img,_                 = dutils.load_origres_nii(SUBJECT_ID,str(LIPREM),METABOLITE+"_conc")
    img_cluster           = get_cluster_mrsi(SUBJECT_ID,LIPREM,METABOLITE)

    img,mask,img_cluster  = np.squeeze(img),np.squeeze(mask),np.squeeze(img_cluster)
    # images_list.append([img[:,:,10],img_cluster[:,:,10],SUBJECT_ID+" "+ str(LIPREM) + " " + METABOLITE])

    # Hessian
    img_hessian,_,_         = hessian.proc(img)
    img_cluster_hessian,_,_ = hessian.proc(img_cluster)
    mask_hessian,_,_        = hessian.proc(mask)
    # images_list.append([img_hessian,mask_hessian,"Hessian"])

    # Sobel
    img_sobel             = hessian.sobel_edge_3d(img)
    img_cluster_sobel     = hessian.sobel_edge_3d(img_cluster)
    mask_sobel            = hessian.sobel_edge_3d(mask)
    # images_list.append([img_sobel,mask_sobel,"Sobel"])

    # Sharpen
    img_sharpen          = hessian.sharpen(img_hessian)
    img_cluster_sharpen  = hessian.sharpen(img_cluster_sobel)
    mask_sharpen         = hessian.sharpen(mask_hessian)
    # img_internal[img_internal<0]=0
    # mask_internal[mask_internal<0]=0
    images_list.append([img[:,:,3],img_cluster_sharpen[:,:,3],"EDGE Sharpen sl5"])
    images_list.append([img[:,:,5],img_cluster_sharpen[:,:,5],"EDGE Sharpen sl5"])
    images_list.append([img[:,:,10],img_cluster_sharpen[:,:,10],"EDGE Sharpen sl10"])
    images_list.append([img[:,:,12],img_cluster_sharpen[:,:,12],"EDGE Sharpen sl12"])
    images_list.append([img[:,:,19],img_cluster_sharpen[:,:,19],"EDGE Sharpen sl18"])

    # SLICE    = int(img.shape[-1]/2)
    
    debug.info(len(images_list))
    fig, axs = plt.subplots(2, len(images_list),figsize=(24, 14), dpi=100)
    dpi = 100
    plt.gcf().set_dpi(dpi)
    title = SUBJECT_ID+"_"+str(LIPREM)+"_"+METABOLITE
    for idi, images in enumerate(images_list):

        # image2D = images[0][:,:,SLICE].mean(axis=2)
        axs[0,idi].imshow(images[0], cmap='viridis')
        axs[0,idi].set_title("OG "+METABOLITE,fontsize=FONTSIZE)

        # image2D = images[1][:,:,SLICE].mean(axis=2)
        axs[1,idi].imshow(images[1], cmap='viridis')
        axs[1,idi].set_title(images[2],fontsize=FONTSIZE)



    # axs[0, 2].imshow(img_features, cmap='winter')
    # axs[0, 2].set_title('Features Images ')

    # axs[1, 2].imshow(img_sobel, cmap='winter')
    # axs[1, 2].set_title('Sobel OG')

    # Flatten the images
    # A,_,_ = hessian.normalize_image(segmented_img_hessian)
    # B,_,_ = hessian.normalize_image(binary_mask)
    # img1_flat = A.flatten()
    # img2_flat = B.flatten()
    # distance = wasserstein_distance(img1_flat, img2_flat)
    # debug.info("wasserstein_distance:",distance)


    plt.tight_layout()
    fig.savefig("EdgeDetect_"+title+".pdf")
    # Show the plot
    plt.show()




        
