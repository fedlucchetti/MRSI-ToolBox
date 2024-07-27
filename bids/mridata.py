import os, sys, glob
from os.path import join, split
import nibabel as nib
from tools.debug import Debug 
from tools.datautils import DataUtils
import re
import numpy as np
import json
from registration.registration import Registration
from tools.filetools import FileTools
from nilearn import datasets



debug  = Debug(verbose=False)
dutils = DataUtils()
reg    = Registration()
ftools = FileTools()

STRUCTURE_PATH = dutils.BIDS_STRUCTURE_PATH
subject_id_exc_list = ["CHUVA016","CHUVA028"]
METABOLITES         = ["NAANAAG", "Ins", "GPCPCh", "GluGln", "CrPCr"]
ATLAS_LIST          = ["aal","destrieux","jhu_icbm_wm","wm_cubeK15mm","wm_cubeK18mm",
                       "geometric_cubeK18mm","geometric_cubeK23mm","cerebellum","chimera"]


class DynamicData:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                value = DynamicData(**value)
            setattr(self, key, value)

class  MRIData:
    def __init__(self, subject_id,session,group="Mindfulness-Project"):
        self.ROOT_PATH           = join(dutils.DATAPATH,group)
        self.PARCEL_PATH         = join(self.ROOT_PATH,"derivatives","chimera-atlases")
        self.CONNECTIVITY_PATH   = join(self.ROOT_PATH,"derivatives","connectomes")
        self.HOMOTOPY_PATH       = join(self.ROOT_PATH,"derivatives","homotopy")
        os.makedirs(self.HOMOTOPY_PATH,exist_ok=True)

        if group=="LPN-Project":
            if subject_id in subject_id_exc_list:
                subject_id=subject_id.replace("CHUV","CHUVN")
            else:
                self.subject_id = subject_id
        elif group=="Mindfulness-Project":
            self.subject_id = subject_id
        elif group=="PilotProject":
            self.subject_id = subject_id
        
        self.data = json.load(open(STRUCTURE_PATH))
        self.PARCEL_PATH = join(self.ROOT_PATH,"derivatives","chimera-atlases")
        self.TRANSFORM_PATH    = join(self.ROOT_PATH,"derivatives","transforms","ants")
        self.session     = session

        self.load_mrsi_all()
        self.load_t1w()
        self.load_dwi_all()
        self.load_parcels()
        self.load_connectivity("dwi")
        self.load_connectivity("spectroscopy")
        self.load_homotopy()
        self.load_featured4d()
        # debug.success("Loaded MRI data from",subject_id,session)
        self.metabolites = np.array(METABOLITES)
        # self.data = DynamicData(**self.data)
 

    def get_mri_dir_path(self,modality="anat"):
        path = join(self.ROOT_PATH,f"sub-{self.subject_id}",f"ses-{self.session}",modality)
        if os.path.exists(path):
            return path
        else:
            debug.warning("get_mri_dir_path: path does not exists")
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
        
    def get_connectivity_dir_path(self,modality="dwi"):
        path = join(self.CONNECTIVITY_PATH,f"sub-{self.subject_id}",f"ses-{self.session}",modality)
        if os.path.exists(path):
            return path
        else:
            debug.warning("connectivity path does not exists")
            debug.warning(path)
            return 
        
    def get_homotopy_dir_path(self):
        path = join(self.HOMOTOPY_PATH,f"sub-{self.subject_id}",f"ses-{self.session}","spectroscopy")
        if os.path.exists(path):
            return path
        else:
            debug.warning("homotopy path does not exists")
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
                elif desc == "Voxel" and acq=="fwhm":
                    comp = "fwhm"
                elif desc == "Voxel" and acq=="snr":
                    comp = "snr"
                else:
                    continue
                self.data["mrsi"][comp][space]["nifti"] = nib.load(join(dirpath,filename))
                self.data["mrsi"][comp][space]["path"]  = join(dirpath,filename)
        self.get_mrsi_mask_image()
    
    def get_mrsi_volume(self,comp,space):
        path = self.get_path("spectroscopy",comp,space)
        try:
            _ = nib.load(path).get_fdata()
            return nib.load(path)
        except Exception as e:
            if space=="orig":
                __nifti        = self.data["mrsi"][comp]["orig"]["nifti"]
                return __nifti
            elif space=="t1w":
                t1w_ref        = self.data["t1w"]["brain"]["orig"]["path"]
                transform_list = self.get_transform("forward","spectroscopy")
                if "snr" in comp or "fwhm" in comp or "crlb" in comp:
                    _space = "orig"
                else: _space = "origfilt"
                mrsi_orig_path = self.data["mrsi"][comp][_space]["path"]
                # print("get_mrsi_volume ",mrsi_orig_path)
                mrsi_anat_np   = reg.transform(t1w_ref,mrsi_orig_path,transform_list).numpy()
                header         = nib.load(t1w_ref).header
                return ftools.numpy_to_nifti( mrsi_anat_np, header)
            elif space=="mni":
                mrsi_anat_nii  = self.get_mrsi_volume(comp,"t1w")
                mni_ref        = datasets.load_mni152_template()
                transform_list = self.get_transform("forward","anat")
                mrsi_mni_np    = reg.transform(mni_ref,mrsi_anat_nii,transform_list).numpy()
                header         = mni_ref.header
                return ftools.numpy_to_nifti( mrsi_mni_np, header)

    def get_path(self,modality,comp,space):
        debug.error("1:modality",modality)
        dir_path = self.get_mri_dir_path(modality)
        debug.error("2:dir_path",dir_path)
        filename = f"sub-{self.subject_id}_ses-{self.session}"
        if modality=="spectroscopy":
            if "crlb" in comp:
                acq = "crlb"
                desc = comp.replace("-crlb","")
            elif comp in METABOLITES or comp == "brainmask" or comp == "mask":
                acq = "conc"
                desc = comp
            elif comp == "snr" or comp == "fwhm":
                acq = comp
                desc = "Voxel"
            else:
                debug.error("MRIData:get_path unrecognized component")
                return
            filename += f"_space-{space}_acq-{acq}"
            filename += f"_desc-{desc}_spectroscopy.nii.gz"
        elif modality=="anat":
            if space=="orig":
                filename += f"_run-01_acq-memprage_{comp}.nii.gz"
            elif space=="mrsi" or space=="mni":
                filename += f"_run-01_space-{space}_acq-memprage_{comp}.nii.gz"
            else:
                debug.error("MRIData:get_path unrecognized component")
                return
        else:
            debug.error("MRIData:get_path unrecognized modality")
            return
        return join(dir_path,filename)
    
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
                # self.data["t1w"]["brain"]["orig"]["nifti"] = nib.load(path)
                self.data["t1w"]["brain"]["orig"]["path"]  = path
            elif "T1w_brainmask.nii.gz" in filename:
                # self.data["t1w"]["mask"]["orig"]["nifti"] = nib.load(path)
                self.data["t1w"]["mask"]["orig"]["path"] = path

    def load_parcels(self):
        dirpath = self.get_mri_parcel_dir_path("anat")
        # debug.info("load_parcels:dirpath",dirpath)
        if dirpath==None:
            return 
        filenames = os.listdir(dirpath)
        debug.info("load_parcels:found n",len(filenames),"Filenames")
        if len(filenames)==0:
            return
        filename = filenames[0]
        for filename in filenames:
            path = join(dirpath,filename)
            if ".nii.gz" in filename and "dseg" in filename and "wm_mask" not in filename:
                
                space = self.extract_suffix(filename,"space")
                # debug.info("load_parcels:filename",filename)
                # if space == "orig":
                for atlas in ATLAS_LIST:
                    if atlas in filename and atlas!="chimera":
                        debug.info("load_parcels:Found",atlas,space,filename)
                        self.data["parcels"][atlas][space]["path"]      = path
                        self.data["parcels"][atlas][space]["labelpath"] = path.replace("nii.gz","tsv")
                    elif atlas in filename and atlas=="chimera":
                        scale,scheme  = self.extract_scale_number(filename)
                        self.data["parcels"][f"{scheme}-{scale}"][space]["path"]      = path
                        self.data["parcels"][f"{scheme}-{scale}"][space]["labelpath"] = path.replace("nii.gz","tsv")

    def load_connectivity(self,mode="dwi"):
        dirpath = self.get_connectivity_dir_path(mode)
        # debug.info("load_parcels:dirpath",dirpath)
        if dirpath==None:
            return 
        filenames = os.listdir(dirpath)
        if len(filenames)==0:
            return
        # filename = filenames[0]
        for filename in filenames:
            # debug.info("load_parcels:filename",filename)
            path = join(dirpath,filename)
            if "_simmatrix.npz" in path:
                for atlas in ATLAS_LIST:
                    if atlas in filename and atlas!="chimera":
                        self.data["connectivity"][mode][atlas]["path"] = path
                    elif atlas in filename and atlas=="chimera":
                        parc_scheme = self.extract_parcellation_substring(filename)
                        self.data["connectivity"][mode][parc_scheme]["path"] = path 

    def load_homotopy(self,include_wm=True):
        include_wm = int(include_wm)
        dirpath = self.get_homotopy_dir_path()
        if dirpath==None:
            return 
        filenames = os.listdir(dirpath)
        if len(filenames)==0:
            return
        # filename = filenames[0]
        for filename in filenames:
            # debug.info("load_parcels:filename",filename)
            path = join(dirpath,filename)
            if f"-homotopy_WM_{include_wm}.nii.gz" in path:
                for atlas in ATLAS_LIST:
                    if atlas in filename and atlas!="chimera":
                        self.data["homotopy"][atlas]["path"] = path
                    elif "LFMIHIFIF-3" in filename:
                        self.data["homotopy"]["LFMIHIFIF-3"]["path"] = path
                    elif "LFMIHIFIF-2" in filename:
                        self.data["homotopy"]["LFMIHIFIF-2"]["path"] = path
                    elif "LFMIHIFIF-4" in filename:
                        self.data["homotopy"]["LFMIHIFIF-4"]["path"] = path
                    

    def load_featured4d(self,include_wm=True):
        include_wm = int(include_wm)
        dirpath = self.get_homotopy_dir_path()
        if dirpath==None:
            return 
        filenames = os.listdir(dirpath)
        if len(filenames)==0:
            return
        # filename = filenames[0]
        for filename in filenames:
            path = join(dirpath,filename)
            if f"-featured4D_WM_{include_wm}.npz" in path:
                debug.info("load_featured4d 2",filename)
                for atlas in ATLAS_LIST:
                    if atlas in filename and atlas!="chimera":
                        self.data["featured4d"][atlas]["path"] = path
                    elif "LFMIHIFIF-3" in filename:
                        self.data["featured4d"]["LFMIHIFIF-3"]["path"] = path
                    elif "LFMIHIFIF-2" in filename:
                        self.data["featured4d"]["LFMIHIFIF-2"]["path"] = path
                    elif "LFMIHIFIF-4" in filename:
                        self.data["featured4d"]["LFMIHIFIF-4"]["path"] = path


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

    def extract_parcellation_substring(self,input_string):
        # Define regex patterns for the two types of substrings
        pattern1 = r'geometric_cubeK23mm'
        pattern2 = r'geometric_cubeK18mm'
        pattern3 = r'chimeraLFIIHIFIF\d+'
        pattern4 = r'chimeraLFMIHIFIF\d+'
        pattern5 = r'wm_cubeK18mm\d+'
        pattern6 = r'wm_cubeK15mm\d+'
        
        # Search for the patterns in the input string
        match1 = re.search(pattern1, input_string)
        match2 = re.search(pattern2, input_string)
        match3 = re.search(pattern3, input_string)
        match4 = re.search(pattern4, input_string)
        match5 = re.search(pattern5, input_string)
        match6 = re.search(pattern6, input_string)
       
        # Return the matched substring
        if match1:
            return match1.group()
        if match2:
            return match2.group()
        if match5:
            return match5.group()
        if match6:
            return match6.group()
        elif "LFIIHIFIF" in input_string or "chimeraLFMIHIFIF" in input_string:
            scale,scheme  = self.extract_scale_number(input_string)
            return f"{scheme}-{scale}"
        else:
            return None

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


    def get_transform(self,direction,space,metabolite_ref="CrPCr"):
        transform_dir_path            = join(self.TRANSFORM_PATH,f"sub-{self.subject_id}",f"ses-{self.session}",space)

        if space=="spectroscopy":
            transform_prefix     = f"sub-{self.subject_id}_ses-{self.session}_desc-mrsi_to_t1w"
        elif  space=="anat":
            transform_prefix     = f"sub-{self.subject_id}_ses-{self.session}_desc-t1w_to_mni"
        elif  space=="dwi":
            transform_prefix     = f"sub-{self.subject_id}_ses-{self.session}_desc-dwi_to_t1w"

        transform_list = list()
        if direction=="forward":  
            transform_list.append(join(transform_dir_path,f"{transform_prefix}.syn.nii.gz"))
            transform_list.append(join(transform_dir_path,f"{transform_prefix}.affine.mat"))
        elif  direction=="inverse":
            transform_list.append(join(transform_dir_path,f"{transform_prefix}.affine_inv.mat"))
            transform_list.append(join(transform_dir_path,f"{transform_prefix}.syn_inv.nii.gz"))
        return transform_list




if __name__=="__main__":
    mrsiData = MRIData(subject_id="S038",session="V1")
    print(mrsiData.data["connectivity"])
