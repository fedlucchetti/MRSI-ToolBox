import os, sys, glob
from os.path import join, split
import nibabel as nib
from tools.debug import Debug 
from tools.datautils import DataUtils
import re
import numpy as np
import json
debug  = Debug()
dutils = DataUtils()

STRUCTURE_PATH = join(dutils.DEVPATH,"Analytics","bids","structure.json")

subject_id_exc_list = ["CHUVA016","CHUVA028"]

METABOLITES = ["NAANAAG", "Ins", "GPCPCh", "GluGln", "CrPCr"]



class DynamicData:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                value = DynamicData(**value)
            setattr(self, key, value)

class  MRIData:
    def __init__(self, subject_id,session,group="Mindfulness-Project"):
        self.ROOT_PATH           = join(dutils.SAFEDRIVE,"Connectome","Data",group)
        self.PARCEL_PATH         = join(self.ROOT_PATH,"derivatives","chimera-atlases")
        if group=="LPN-Project":
            if subject_id in subject_id_exc_list:
                subject_id=subject_id.replace("CHUV","CHUVN")
            else:
                self.subject_id = subject_id
        elif group=="Mindfulness-Project":
            self.subject_id = subject_id
        
        self.data = json.load(open(STRUCTURE_PATH))
        self.PARCEL_PATH = join(self.ROOT_PATH,"derivatives","chimera-atlases")
        self.TRANSFORM_PATH    = join(self.ROOT_PATH,"derivatives","transforms","ants")
        self.session     = session

        self.load_mrsi_all()
        self.load_t1w()
        self.load_dwi_all()
        self.load_parcels()
        debug.success("Loaded MRI data from",subject_id,session)
        self.metabolites = np.array(METABOLITES)
        # self.data = DynamicData(**self.data)
 

    def get_mri_dir_path(self,modality="anat"):
        path = join(self.ROOT_PATH,f"sub-{self.subject_id}",f"ses-{self.session}",modality)
        if os.path.exists(path):
            return path
        else:
            debug.warning("path does not exists")
            debug.warning(path)
            return 
        
    def get_mri_parcel_dir_path(self,modality="anat"):
        path = join(self.PARCEL_PATH,f"sub-{self.subject_id}",f"ses-{self.session}",modality)
        if os.path.exists(path):
            return path
        else:
            debug.warning("path does not exists")
            debug.warning(path)
            return 
        
    def load_dwi_all(self):
        dirpath = self.get_mri_dir_path("dwi")
        if dirpath==None:
            return 
        filenames = os.listdir(dirpath)
        if len(filenames)==0:
            return
        for filename in filenames:
            if "dwi.bval" in filename:
                self.data["dwi"]["bval"] = join(dirpath,filename)
            elif "dwi.bvec" in filename:
                self.data["dwi"]["bvec"] = join(dirpath,filename)
            elif "dwi.nii.gz" in filename:
                self.data["dwi"]["nifti"] = join(dirpath,filename)
            elif "dwi.mif" in filename:
                self.data["dwi"]["mif"] = join(dirpath,filename)     

    def load_mrsi_all(self):
        dirpath = self.get_mri_dir_path("spectroscopy")
        if dirpath==None:
            return 
        filenames = os.listdir(dirpath)
        if len(filenames)==0:
            return
        for filename in filenames:
            if ".nii" in filename:
                space = self.extract_suffix(filename,"space")
                acq   = self.extract_suffix(filename,"acq")
                desc  = self.extract_suffix(filename,"desc")
                if desc in METABOLITES and acq=="conc":
                    comp = desc
                elif desc in METABOLITES and acq=="crlb":
                    comp = f"{desc}-crlb"
                else:
                    continue
                self.data["mrsi"][comp][space]["nifti"] = nib.load(join(dirpath,filename))
                self.data["mrsi"][comp][space]["path"]  = join(dirpath,filename)
        self.get_mrsi_mask_image()
    
    def load_t1w(self):
        dirpath = self.get_mri_dir_path("anat")
        if dirpath==None:
            return 
        filenames = os.listdir(dirpath)
        if len(filenames)==0:
            return
        for filename in filenames:
            path = join(dirpath,filename)
            if "T1w_brain.nii.gz" in filename:
                self.data["t1w"]["brain"]["orig"]["nifti"] = nib.load(path)
                self.data["t1w"]["brain"]["orig"]["path"]  = path
            elif "T1w_brainmask.nii.gz" in filename:
                self.data["t1w"]["mask"]["orig"]["nifti"] = nib.load(path)
                self.data["t1w"]["mask"]["orig"]["path"] = path

    def load_parcels(self):
        dirpath = self.get_mri_parcel_dir_path("anat")
        # debug.info("load_parcels:dirpath",dirpath)
        if dirpath==None:
            return 
        filenames = os.listdir(dirpath)
        if len(filenames)==0:
            return
        filename = filenames[0]
        for filename in filenames:
            # debug.info("load_parcels:filename",filename)
            path = join(dirpath,filename)
            if ".nii.gz" in filename and "chimera" in filename:
                space = self.extract_suffix(filename,"space")
                if space == "orig":
                    scale,scheme  = self.extract_scale_number(filename)
                    self.data["parcels"][f"{scheme}-{scale}"]["orig"]["path"] = path
                    self.data["parcels"][f"{scheme}-{scale}"]["orig"]["nifti"] = nib.load(path)
                    self.data["parcels"][f"{scheme}-{scale}"]["orig"]["labelpath"] = path.replace("nii.gz","tsv")



    def extract_suffix(self,filename,suffix):
        # Use a regular expression to search for the pattern matching 'space-{SPACE}'
        match = re.search(rf"_{suffix}-([^_]+)", filename)
        if match:
            return match.group(1)  # Returns the captured group, which corresponds to {SPACE}
        else:
            return None  # Return None if the pattern is not found

    def extract_scale_number(self,filename):
        """
        Extracts the number following 'scale' in the filename.

        Args:
        filename (str): The filename from which to extract the scale number.

        Returns:
        int: The number following 'scale' or None if no such number is found.
        """
        # Define the regular expression to find 'scale' followed by any number
        match_scale = re.search(r"scale(\d+)", filename)
        match_scheme = re.search(r"atlas-chimera(\w+)", filename)
        # debug.info("extract_scale_number:match_scheme",match_scheme)
        # Extract the scale number
        scale_number = int(match_scale.group(1)) if match_scale else None

        # Extract the scheme
        scheme = match_scheme.group(1) if match_scheme else None
        scheme = scheme.replace("_desc","")
        return scale_number, scheme

    def get_mrsi_mask_image(self):
        dirpath   = self.get_mri_dir_path("spectroscopy")
        filenames = os.listdir(dirpath)
        mrsi_mask = None
        if len(filenames)==0:
            return
        for filename in filenames:
            path = join(dirpath,filename)
            if "WaterSignal_spectroscopy.nii.gz" in filename:
                water_signal  = nib.load(path).get_fdata()
                header        = nib.load(path).header
                mrsi_mask     = np.zeros(water_signal.shape)
                mrsi_mask[water_signal>0]  = 1
                mrsi_mask[water_signal<=0] = 0
                affine = header.get_best_affine()
        # Preserve affine transform
        header.set_data_dtype(np.float32)
        nifti_img = nib.Nifti1Image(mrsi_mask.astype(np.float32), affine)
        for key in header.keys():
            try:
                nifti_img.header[key] = header[key]
            except Exception as e:
                debug.warning(f"Could not set header field '{key}': {e}")
        filename = f"sub-{self.subject_id}_ses-{self.session}_space-orig_acq-conc_desc-brainmask_spectroscopy.nii.gz"
        outpath = join(dirpath,filename)
        nifti_img.to_filename(f"{outpath}")
        filename = f"sub-{self.subject_id}_ses-{self.session}_space-origfilt_acq-conc_desc-brainmask_spectroscopy.nii.gz"
        outpath = join(dirpath,filename)
        nifti_img.to_filename(f"{outpath}")
        self.data["mrsi"]["mask"]["orig"]["nifti"]     = nifti_img
        self.data["mrsi"]["mask"]["origfilt"]["nifti"] = nifti_img
        self.data["mrsi"]["mask"]["orig"]["path"]      = outpath
        self.data["mrsi"]["mask"]["origfilt"]["path"]  = outpath





    def __get_parcel_image(self):
        scheme = "LFMIIIFIF"
        scale  = "3"
        subject_id = self.subject_id
        S,V = subject_id[0:4],subject_id[-2::]
        filename = f"run-1_space-orig_atlas-chimera{scheme}_desc-scale{scale}grow2mm_dseg_mrsi.nii.gz"
        path = join(self.PARCEL_PATH,f"sub-{S}", 
                                    f"ses-{V}", 
                                    "anat", 
                                    f"sub-{S}_ses-{V}_{filename}")
        if not os.path.exists(path):
            path = join(self.PARCEL_PATH,f"sub-{S}", 
                                        f"ses-{V}", 
                                        "anat", 
                                        f"sub-{S}_ses-{V}_{filename}")
        image,header       = nib.load(path).get_fdata(),nib.load(path).header
        return image,path, header

    def get_transform(self,direction,space,metabolite_ref="CrPCr"):
        self.ROOT_PATH            = join(self.TRANSFORM_PATH,f"sub-{self.subject_id}",f"ses-{self.session}",space)

        if space=="spectroscopy":
            transform_prefix     = f"sub-{self.subject_id}_ses-{self.session}_desc-mrsi_to_t1w"
        elif  space=="anat":
            transform_prefix     = f"sub-{self.subject_id}_ses-{self.session}_desc-t1w_to_mni"
        elif  space=="dwi":
            transform_prefix     = f"sub-{self.subject_id}_ses-{self.session}_desc-dwi_to_t1w"

        transform_list = list()
        if direction=="forward":  
            transform_list.append(join(self.ROOT_PATH,f"{transform_prefix}.syn.nii.gz"))
            transform_list.append(join(self.ROOT_PATH,f"{transform_prefix}.affine.mat"))
        elif  direction=="inverse":
            transform_list.append(join(self.ROOT_PATH,f"{transform_prefix}.affine_inv.mat"))
            transform_list.append(join(self.ROOT_PATH,f"{transform_prefix}.syn_inv.nii.gz"))
        return transform_list




if __name__=="__main__":
    mrsiData = MRIData(subject_id="CHUVA009",session="V5")
