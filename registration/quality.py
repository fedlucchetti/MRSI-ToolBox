import numpy as np
import ants
import copy
from tools.debug import Debug

from rich.progress import track, Progress
import multiprocessing
debug  = Debug()


class Quality:
    def __init__(self) -> None:
        pass

    def get_MI(self,fixed,moving):
        mi = ants.image_similarity(fixed,moving,metric_type='MattesMutualInformation')
        return mi
    
    def get_qmask(self,img_dict, snr_th=4, fwhm_th=0.15, crlb_th=25):
        snr_3D  = img_dict["snr"]
        fwhm_3D = img_dict["fwhm"]
        crlb_3D = img_dict["crlb"]

        qmask3D = np.ones(snr_3D.shape, dtype=bool)

        # Apply thresholds
        mask_conditions = (snr_3D < snr_th) | (fwhm_3D > fwhm_th) | (crlb_3D > crlb_th)
        qmask3D[mask_conditions] = 0
        
        return qmask3D

    def get_noise_mask(self,img3D_dict):
        signal3D       = img3D_dict["signal"].numpy()
        snr3D          = img3D_dict["snr"]+0.1
        mask           = img3D_dict["mask"].numpy()
        noise_mask     = np.zeros(signal3D.shape)
        valid_division = (mask != 0)
        noise_mask[valid_division] = signal3D[valid_division] / snr3D[valid_division]
        return noise_mask

    def get_noisy_img3(self,image3D, noise_mask):
        noisy_image3D = copy.deepcopy(image3D)
        # Iterate over each voxel in the image
        indices = np.where(noise_mask != 0)
        for i, j, k in zip(*indices):
            mu = image3D[i, j, k]  # Mean is the original voxel value
            sigma = noise_mask[i, j, k]  # Std is the value from noise_mask at this index
            # Generate a random number from a normal distribution with mean=mu and std=sigma
            noisy_image3D[i, j, k] = np.random.normal(mu, sigma)
        return noisy_image3D
    
    def transform(self,fixed_image, moving_image, transform):

        warped_image = ants.apply_transforms(fixed=fixed_image, 
                                             moving=moving_image, 
                                             transformlist=transform)
        return warped_image      

    def threshold(self,image):
        image[image>0]  = 1
        image[image<=0] = 0
        return image

    def randomize_transform(self,fixedImg3D,origRes_dict,transform,N=50):
        image3D            = origRes_dict["signal"].numpy()
        noise_mask         = origRes_dict["noise"]
        mask               = origRes_dict["mask"]
        transformed_mask   = self.transform(fixedImg3D,
                                        mask,
                                        transform).numpy()
        transformed_mask = self.threshold(transformed_mask)
        noisy_warpedImages = np.zeros((N,)+transformed_mask.shape)
        noisy_Images       = np.zeros((N,)+image3D.shape)

        for i in track(range(N), description="Randomize"):
            noisy_image3D   = self.get_noisy_img3(image3D,noise_mask)
            noisy_Images[i] = noisy_image3D
            noisy_image3D   = ants.from_numpy(noisy_image3D)
            
            warpedImage3D   = self.transform(fixedImg3D,
                                           noisy_image3D,
                                           transform).numpy()
            warpedImage3D[transformed_mask==0] = np.nan
            noisy_warpedImages[i] = warpedImage3D
        return noisy_warpedImages,noisy_Images, transformed_mask
        
    

if __name__=="__main__":
    q = Quality()

