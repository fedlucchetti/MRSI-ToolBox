import numpy as np


class ArrayUtils:
    def __init__(self) -> None:
        pass

    def scale_to_01(self,image,return_min_max=True):
        norm_img = (image - np.min(image)) / (np.max(image) - np.min(image))
        norm_img = (image - np.min(image)) / (np.max(image) - np.min(image)) 
        if return_min_max:
            return norm_img,np.min(image),np.max(image)
        else: return norm_img
    
    def scale_to_og(self,image,min_v,max_v):
        return image * (max_v-min_v)+min_v