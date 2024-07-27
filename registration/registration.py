import os, sys, glob, time, json, stat
from tools.debug import Debug
import nibabel as nib
import ants
import numpy as np

from tools.datautils import DataUtils

from os.path import join, split


dutils = DataUtils()
debug  = Debug()



class Registration():
    def __init__(self):
        self.DATAPATH           = dutils.DATAPATH
        self.MRSI_OG_PATH       = join(dutils.DATAPATH,"MindfullTeen")
        self.mni152_path        = join(dutils.DATAPATH,"MNI","MNI152_T1_1mm.nii.gz")
        self.mni152_brain_path  = join(dutils.DATAPATH,"MNI","MNI152_T1_1mm_brain.nii.gz")

   

   
    def skull_strip(self,inpath,image_type="t1"):

        
        if image_type=="t1":
            dirpath, filename = os.path.split(inpath)
            mask_path = os.path.join(dirpath,"T1Brain_noSkull_mask.nii.gz")
            if not os.path.exists(inpath):
                debug.error("skull_strip: path does not exist",inpath)
            outpath = os.path.join(dirpath,"T1Brain_noSkull.nii")
            if os.path.exists(outpath+".gz"):
                # debug.info("T1 Skull already stripped, NEXT")
                return outpath, mask_path
            else:
                debug.info("Start T1 Skull stripping")

        elif image_type=="template":
            inpath  = self.mni152_path
            dirpath, filename = os.path.split(inpath)
            outpath = self.mni152_brain_path
            if os.path.exists(outpath):
                debug.info("Template Skull already stripped, NEXT")
                mask_path = os.path.join(dirpath,"MNI152_T1_1mm_brain_mask.nii.gz")
                return outpath, mask_path
            else:
                inpath = self.mni152_path
                debug.info("Start MNI152 Skull stripping")

        start = time.time()
        command = f"bet  {inpath} {outpath} -S -B"
        os.system(command)
        debug.success("Skull stripping completed in", round(time.time() - start, 1), "sec")
        debug.success("Brain T1 saved to:", outpath)

        return outpath, mask_path


    def __load_ants_image(self, input):
        """
        Checks if the input is a path, an ANTsImage, or a Nifti2Image. If it's a path, it loads the image.
        Returns the ANTsImage, Nifti2Image, or None if the input is invalid.
        """
        if isinstance(input, str):
            if not os.path.exists(input):
                debug.error("__load_ants_image: Image path does not exist: %s", input)
                return None
            try:
                return ants.image_read(input)
            except Exception as e:
                debug.error("__load_ants_image: Error reading image from path %s: %s", input, e)
                return None
        elif isinstance(input, ants.core.ants_image.ANTsImage):
            return input
        elif isinstance(input, nib.Nifti1Image):
            return ants.from_nibabel(input)
        else:
            debug.error("__load_ants_image: Input must be a path, an ANTsImage, or a Nifti1Image object.")
            return None

    def register(self, fixed_input, moving_input,fixed_mask=None,moving_mask=None ,transform="sr",verbose=False):
        debug.info("Start registration")
        ## Preformat ANTS images
        fixed_image = self.__load_ants_image(fixed_input)
        if fixed_image is None:
            return  
        moving_image = self.__load_ants_image(moving_input)
        if moving_image is None:
            return
        if fixed_mask is None:
            debug.warning("No fixed mask specified")
        else:
            fixed_mask  = self.__load_ants_image(fixed_mask)
        if moving_mask is None:
            debug.warning("No moving mask specified")
        else:
            moving_mask = self.__load_ants_image(moving_mask)

        # Perform registration
        start = time.time()
        syn_tx = ants.registration(fixed=fixed_image,
                                   moving=moving_image,
                                   fixed_mask=fixed_mask,
                                   moving_mask=moving_mask,
                                   verbose=verbose,
                                   type_of_transform=f'antsRegistrationSyN[{transform}]')
        elapsed_time = round(time.time() - start, 1)
        debug.success("Registration completed in",elapsed_time , "sec")
        return syn_tx, elapsed_time

    def transform(self,fixed_image,moving_image,transform,interpolator_mode="linear"):
        # Apply the transformations to align the T1w image to the MNI template
        fixed_image = self.__load_ants_image(fixed_image)
        if fixed_image is None: return  
        moving_image = self.__load_ants_image(moving_image)
        if moving_image is None: return
        # debug.success("transform: moving and fixed path exist")

        warped_image = ants.apply_transforms(fixed=fixed_image, 
                                             moving=moving_image, 
                                             transformlist=transform,
                                             interpolator=interpolator_mode)
        return warped_image

    def get_deformation(self,DF_image,mask=None):
        """
        # Compute the average of the norm of the displacement field at every voxel
        """
        normDF_per_voxel  = np.linalg.norm(DF_image,axis=-1)
        if mask is not None:
            total_deformation = normDF_per_voxel[mask == 1].mean()
        else:
            total_deformation = normDF_per_voxel.mean()
        return total_deformation
 



            



