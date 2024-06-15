
import os,sys, glob,re, shutil

from tools.datautils import DataUtils
from tools.debug import Debug 
import nibabel as nib
import numpy as np
from os.path import join, split
dutils = DataUtils()
debug  = Debug()
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


class FileTools:
    def __init__(self,group="LPN-Project") -> None:
        self.ROOTDIRPATH       = join(dutils.SAFEDRIVE,"Connectome","Data",group)
        debug.success("Load data from",self.ROOTDIRPATH)

        
    def save_nii_file(self, tensor3D, header,outpath):
        # nifti_img = self.numpy_to_nifti(tensor3D, header)
        affine = header.get_best_affine()
        # Preserve affine transform
        header.set_data_dtype(np.float32)
        nifti_img = nib.Nifti1Image(tensor3D.astype(np.float32), affine)
        # Update specific fields in the new header from the original header
        for key in header.keys():
            try:
                nifti_img.header[key] = header[key]
            except Exception as e:
                debug.warning(f"Could not set header field '{key}': {e}")
        #debug.info("Saving to",outpath)
        nifti_img.to_filename(f"{outpath}")

    def numpy_to_nifti(self, tensor3D, header):
        affine = header.get_best_affine()
        # Preserve affine transform
        header.set_data_dtype(np.float32)
        nifti_img = nib.Nifti1Image(tensor3D.astype(np.float32), affine)
        
        # Update specific fields in the new header from the original header
        for key in header.keys():
            try:
                nifti_img.header[key] = header[key]
            except Exception as e:
                debug.warning(f"Could not set header field '{key}': {e}")
        
        return nifti_img

    def save_transform(self,transform,dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        # Copy each file in the transform list to the custom directory
        for transform_file in transform:
            filename = split(transform_file)[1]
            if "Warp" in filename:
                filename="Warp.nii.gz"
            elif "Affine" in filename:
                filename="GenericAffine.mat"
            dest_file_path = join(dir_path, filename)
            # Copy the file to the new location
            shutil.copy(transform_file, dest_file_path)
            debug.success(f"Saved: {filename}")
 
    def save_nii4D_file(self,path_list,outpath):
        _data = nib.load(path_list[0])
        image_list=np.zeros([_data.get_fdata().shape+(len(path_list),)])
        header = _data.header
        for ids,path in enumerate(path_list):
            data = nib.load(path)
            image_list[:,:,:,ids] = data.get_fdata()
        self.save_nii_file(np.array(image_list),header,f"{outpath}.nii.gz")

    def list_nii_files(self,directory):
        """
        Lists the absolute paths of all .nii files within the given directory, including its subdirectories.
        
        Parameters:
        - directory: The root directory to search for .nii files.
        
        Returns:
        - A list of absolute paths to each .nii file found.
        """
        nii_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".nii") and not file.endswith(".nii.gz"):
                    nii_files.append(os.path.abspath(os.path.join(root, file)))
        return nii_files

    def list_recordings(self):
        recording_list = list()
        subject_list = os.listdir(self.ROOTDIRPATH)
        for subject_id in subject_list:
            if "sub-" not in subject_id:continue
            session_list = os.listdir(join(self.ROOTDIRPATH,subject_id))
            for session in session_list:
                if "ses-" in session:
                    acq_list = os.listdir(join(self.ROOTDIRPATH,subject_id,session))
                    mrsi_dir_path = join(self.ROOTDIRPATH,subject_id,session,"spectroscopy")
                    if os.path.exists(mrsi_dir_path):
                        n_mrsi = len(os.listdir(mrsi_dir_path))
                        if n_mrsi!=0 and "anat" in acq_list:
                            recording_list.append([subject_id[4::],session[4::]])
        recording_list = np.array(recording_list)
        ids = np.argsort(recording_list[:,0])
        return recording_list[ids,:]




    def align_spectra(self, *spectra, target_length=None):
        """
        Aligns multiple spectra arrays to a common size using interpolation.

        Parameters:
            spectra (list of np.ndarray): Spectra arrays to align.
            target_length (int, optional): Length to which all spectra will be aligned.
                                        If None, uses the maximum length found.

        Returns:
            list of np.ndarray: List of aligned spectra arrays.
        """
        if target_length is None:
            target_length = max(len(spectrum) for spectrum in spectra)

        aligned_spectra = []
        for spectrum in spectra:
            # Create interpolation function
            x = np.linspace(0, 1, len(spectrum))
            f = interp1d(x, spectrum, kind='linear', fill_value='extrapolate')
            # Interpolate to the target length
            new_x = np.linspace(0, 1, target_length)
            aligned_spectra.append(f(new_x))

        return aligned_spectra

    def plot_spectra(self, density_spectrum, length_spectrum, fa_spectrum, metab_spectrum,result_path=None,plotshow=False):
        """
        Plots aligned spectra on a single plot.

        Parameters:
            density_spectrum (np.ndarray): Laplacian spectrum for density.
            length_spectrum (np.ndarray): Laplacian spectrum for length.
            fa_spectrum (np.ndarray): Laplacian spectrum for fractional anisotropy (FA).
            metab_spectrum (np.ndarray): Laplacian spectrum for metabolic connectivity.
        """
        # Align all spectra
        aligned_spectra = self.align_spectra(density_spectrum, length_spectrum, fa_spectrum, metab_spectrum)
        labels = ['density', 'length', 'fa', 'metabolic']

        # Plot each aligned spectrum
        for spectrum, label in zip(aligned_spectra, labels):
            plt.plot(spectrum, label=label)

        plt.ylabel("Laplace Spectrum")
        plt.legend()
        plt.grid(True)
        if result_path:
            plt.savefig(f"{result_path}.pdf", dpi=300)  # Ensure dpi is set for saving as well
            plt.clf()
        if plotshow:
            plt.show()

    def compare_and_categorize(self,fa_bin, metab_bin):
        """
        Compare two binarized symmetric similarity matrices and categorize them into four groups:
        (fa_bin=0, metab_bin=0), (fa_bin=0, metab_bin=1), (fa_bin=1, metab_bin=0), and (fa_bin=1, metab_bin=1).

        Parameters:
            fa_bin (np.ndarray): First binarized symmetric similarity matrix.
            metab_bin (np.ndarray): Second binarized symmetric similarity matrix.

        Returns:
            tuple of lists: Lists of values categorized into the four groups.
        """
        if fa_bin.shape != metab_bin.shape:
            raise ValueError("Both matrices must have the same shape.")

        # Ensure the matrices are square
        if fa_bin.shape[0] != fa_bin.shape[1]:
            raise ValueError("Both matrices must be square.")

        # Extract the off-diagonal elements
        mask = ~np.eye(fa_bin.shape[0], dtype=bool)
        fa_elements = fa_bin[mask]
        metab_elements = metab_bin[mask]

        # Create four groups based on the combinations
        group_00 = fa_elements[(fa_elements == 0) & (metab_elements == 0)]
        group_01 = fa_elements[(fa_elements == 0) & (metab_elements == 1)]
        group_10 = fa_elements[(fa_elements == 1) & (metab_elements == 0)]
        group_11 = fa_elements[(fa_elements == 1) & (metab_elements == 1)]

        return group_00, group_01, group_10, group_11

    def plot_four_distributions(self,group_00, group_01, group_10, group_11):
        """
        Plots four distributions based on the comparison groups:
        (fa_bin=0, metab_bin=0), (fa_bin=0, metab_bin=1), (fa_bin=1, metab_bin=0), and (fa_bin=1, metab_bin=1).

        Parameters:
            group_00 (np.ndarray): Distribution where (fa_bin=0, metab_bin=0).
            group_01 (np.ndarray): Distribution where (fa_bin=0, metab_bin=1).
            group_10 (np.ndarray): Distribution where (fa_bin=1, metab_bin=0).
            group_11 (np.ndarray): Distribution where (fa_bin=1, metab_bin=1).
        """
        plt.figure(figsize=(10, 6))
        plt.hist(group_00, bins=20, alpha=0.6, label='fa_bin=0, metab_bin=0', color='blue')
        plt.hist(group_01, bins=20, alpha=0.6, label='fa_bin=0, metab_bin=1', color='green')
        plt.hist(group_10, bins=20, alpha=0.6, label='fa_bin=1, metab_bin=0', color='orange')
        plt.hist(group_11, bins=20, alpha=0.6, label='fa_bin=1, metab_bin=1', color='red')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.title('Distributions of Element Comparisons Between fa_bin and metab_bin')
        plt.grid(True)
        plt.show()

if __name__=="__main__":
    ft = FileTools()
    debug.info(ft.list_mrsi_subject_list())
