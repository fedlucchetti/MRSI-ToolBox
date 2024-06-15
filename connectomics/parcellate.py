import nibabel as nib
import numpy as np
import os
from os.path import join, split
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from scipy.stats import linregress
from scipy.stats import spearmanr
from registration.registration import Registration
from scipy.stats import chi2
from sklearn.metrics import mutual_info_score
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
from threading import Thread, Event
import csv, copy, sys, time
from tools.debug import Debug
from rich.progress import Progress,track
import scipy.stats as stats
from tools.datautils import DataUtils
from concurrent.futures import ProcessPoolExecutor
# from tqdm.auto import tqdm  # For progress bar
import itertools


dutils = DataUtils()
debug  = Debug()
reg    = Registration()

class Parcellate:
    def __init__(self) -> None:
        # self.PARCEL_PATH = "/media/flucchetti/77FF-B8071/Mindfulness-Project/derivatives/chimera-atlases/"
        self.PARCEL_PATH      = join(dutils.SAFEDRIVE,"Connectome","Data","Mindfulness-Project","derivatives","chimera-atlases")
        self.PARCEL_PATH_ARMS = join(dutils.SAFEDRIVE,"Connectome","Data","LPN-Project","derivatives","chimera-atlases")
        pass

    def dilate(img,new_shape = (193, 229, 193), extension_mode="constant"):
        # Calculate the zoom factors for each dimension
        data = np.nan_to_num(img.get_fdata())
        data = (data - data.min()) / (data.max() - data.min())
        img  = nib.nifti1.Nifti1Image(data, img.affine,header=img.header)
        data = img.get_fdata()
        # Calculate the zoom factors for each dimension
        zoom_factors = (new_shape[0] / data.shape[0], new_shape[1] / data.shape[1], new_shape[2] / data.shape[2])

        # Perform the zoom operation (interpolation)
        new_shape = new_shape+(data.shape[-1],)
        resized_data = np.zeros(new_shape)
        for i in tqdm(range(data.shape[-1])):
            resized_data[:,:,:,i] = zoom(data[:,:,:,i], zoom_factors, order=3,mode=extension_mode)  # 'order=3' for cubic interpolation

        resized_data[resized_data<0.001] = 0.0
        # Create a new NiBabel image from the resized data
        new_img = nib.nifti1.Nifti1Image(resized_data, img.affine,header=img.header)

        return new_img


    def get_parcel_path(self,subject_id):
        debug.info("get_parcel_path for",subject_id)
        filename = "run-01_acq-memprage_space-orig_atlas-chimeraLFMIHIFIF_desc-scale3grow2mm_dseg.tsv"
        # filename = "run-01_acq-memprage_space-orig_atlas-chimeraLFIIHIFIF_desc-scale1grow2mm_dseg.tsv"
        S,V = subject_id[0][0:4],subject_id[0][5:]
        if V=="V2_BIS":V="V3"
        path = join(self.PARCEL_PATH,f"sub-{S}", 
                                  f"ses-{V}", 
                                  "anat", 
                                  f"sub-{S}_ses-{V}_{filename}")
    
        return path

    def get_parcel_path_arms(self,subject_id):
        scheme = "LFMIIIFIF"
        scale  = "3"
        S,V = subject_id[0:4],subject_id[-2::]
        # filename = f"run-1_space-orig_atlas-chimeraLFMIIIFIF_desc-scale1grow2mm_dseg.tsv"
        filename = f"run-1_space-orig_atlas-chimera{scheme}_desc-scale{scale}grow2mm_dseg.tsv"
        path = join(self.PARCEL_PATH_ARMS,f"sub-CHUV{S}", 
                                    f"ses-{V}", 
                                    "anat", 
                                    f"sub-CHUV{S}_ses-{V}_{filename}")
        if not os.path.exists(path):
            path = join(self.PARCEL_PATH_ARMS,f"sub-CHUVN{S}", 
                                        f"ses-{V}", 
                                        "anat", 
                                        f"sub-CHUVN{S}_ses-{V}_{filename}")
        return path
    

    def get_parcel_header(self,parcel_header_path):
        '''
        Retrieves and filters a brain parcel image for a given subject 
        based on an ignore list. 

        Parameters:
        - subject_id (str): Unique identifier for the subject whose parcel data is to be retrieved.
        - ignore_list (list of str): Labels of parcels to be excluded from the output.

        Returns:
        - tuple: 
            - filtered parcel image (numpy array), 
            - filtered list of parcel IDs, 
            - filtered list of labels (where each label is a list split by '-'), 
            - filtered list of color codes. 
        '''
        parcel_ids, label_list, color_codes = self.read_tsv_file(parcel_header_path)
        label_list = [label.split('-') for label in label_list]
        
        header_dict = dict()
        for idx , parcel_id in enumerate(parcel_ids):
            if parcel_id<1000:
                header_dict[parcel_id] = {"label":label_list[idx],"color":color_codes[idx],"mask":1,
                                                "count":[],"mean":0,"std":0,"med":0,"t1cov":[]}
        header_dict[0] = {"label":["BND"],"color":0,"mask":0,"count":[0],"mean":0,"std":0,"med":0,"t1cov":[0]}

        return header_dict


    def tranform_parcel(self,target_image_path,parcel_path,transform):     
        parcel_image3D           = reg.transform(target_image_path,
                                                parcel_path,
                                                transform,
                                                interpolator_mode="genericLabel")
        return parcel_image3D.numpy()

    def filter_parcel(self,parcel_image,parcel_header_dict ,ignore_list=[]):
        '''
        Filters a given parcel image and its corresponding 
        metadata (parcel IDs, labels, and color codes) by removing 
        parcels specified in an ignore list. 

        Parameters:
        - parcel_image (numpy array): The brain parcel image 
        - parcel_ids (list of int): List of unique parcel IDs 
        - label_list (list of list of str): List of parcel labels 
        - color_codes (list of str): List of color codes 
        - ignore_list (list of str): Labels of parcels to be ignored

        Returns:
        - tuple: A tuple containing a  filtered version of the input
        '''
        filt_parcel_image = copy.deepcopy(parcel_image)


        # Iterate over a copy of the keys list to safely modify the dictionary
        for label_idx in list(parcel_header_dict.keys()):
            entry = parcel_header_dict[label_idx]
            subparcel_labels = entry["label"]
            for subparcel_label in subparcel_labels:
                if subparcel_label in ignore_list:
                    filt_parcel_image[parcel_image == label_idx] = 0
                    parcel_header_dict[label_idx]["mask"] = 0
                    del parcel_header_dict[label_idx]
                    break 
            else:
                parcel_header_dict[label_idx]["mask"] = 1

        return filt_parcel_image,parcel_header_dict

    def merge_parcels(self,parcel_image,parcel_header_dict, merge_parcels_dict):
        merged_parcel_image = copy.deepcopy(parcel_image)
        for key, entry in merge_parcels_dict.items():
            base_label = entry["merge"][0]
            merge_ids  = range(entry["merge"][0] + 1, entry["merge"][1] + 1)  # exclude parcel id to merge with
            parcel_header_dict[key]["label"] = entry["label"]
            for merge_idx in merge_ids:
                # Update the parcel image to merge the labels
                merged_parcel_image[parcel_image == merge_idx] = base_label
                # Delete the merged label's entry from the header dict if it exists
                parcel_header_dict.pop(merge_idx, None)
        return merged_parcel_image, parcel_header_dict

    def compute_pearson_correlation(self,image1, image2):
        num_modes, depth, height, width = image1.shape
        correlations = np.zeros(num_modes)  # Store correlation coefficients for each mode
        for mode in range(num_modes):
            # Flatten the 3D volumes of the current mode
            flat_image1 = image1[mode].reshape(-1)
            flat_image2 = image2[mode].reshape(-1)
            
            # Compute Pearson correlation coefficient for the current mode
            correlation = np.corrcoef(flat_image1, flat_image2)[0, 1]
            correlations[mode] = correlation
        
        return correlations
    

    def compute_autocorrelation(self,ranks, max_lag):
        N = len(ranks)
        mean = np.mean(ranks)
        var = np.var(ranks)

        autocorrelations = []
        for k in range(1, max_lag + 1):
            autocovariance = np.sum((ranks[:N-k] - mean) * (ranks[k:] - mean)) / N
            autocorrelation = autocovariance / var
            autocorrelations.append(autocorrelation)

        return autocorrelations


    def spearman_corr_adjusted(self,x, y, max_lag=3):
        # Rank transform
        x_ranks = stats.rankdata(x)
        y_ranks = stats.rankdata(y)

        # Compute autocorrelations for both rank-transformed arrays
        x_autocorrelations = self.compute_autocorrelation(x_ranks, max_lag)
        y_autocorrelations = self.compute_autocorrelation(y_ranks, max_lag)

        # Average autocorrelation
        avg_autocorrelation = (np.mean(x_autocorrelations) + np.mean(y_autocorrelations)) / 2

        # Effective sample size
        N = len(x)
        N_eff =  N / (1 + 2 * np.sum(avg_autocorrelation))
        # N_eff = effective_sample_size(N, [avg_autocorrelation])

        # Compute Spearman correlation
        spearman_corr, p_value = stats.spearmanr(x, y,alternative="two-sided")

        # Adjust the p-value using the effective sample size (assuming t-distribution for correlation)
        # Degrees of freedom adjustment: N_eff - 2
        adjusted_p_value = stats.t.sf(np.abs(spearman_corr) * np.sqrt((N_eff - 2) / (1 - spearman_corr**2)), df=N_eff - 2) * 2

        return spearman_corr, adjusted_p_value


    def read_tsv_file(self,filepath):
        """
        Reads a TSV file with three columns: an integer, a string label, and a color code.
        Returns the data as three separate lists.

        Parameters:
        - filepath: Path to the TSV file.

        Returns:
        - numbers: List of integers from the first column.
        - labels: List of string labels from the second column.
        - colors: List of color codes (strings) from the third column.
        """
        numbers, labels, colors = [], [], []

        with open(filepath, 'r') as file:
            tsv_reader = csv.reader(file, delimiter='\t')
            next(tsv_reader, None)  # Skip the header row
            for row in tsv_reader:
                # Assuming the file structure is exactly as described
                number, label, color = row
                numbers.append(int(number))  # Convert string to int
                labels.append(label)
                colors.append(color)

        return numbers, labels, colors


    def count_voxels_per_parcel(self,parcel_image, mask_mrsi,mask_t1,parcel_header_dict):
        """
        Identifies parcel IDs that are completely ignored based on the mask, 
        meaning there are fewer than 'threshold' voxels available for that specific parcel.

        Parameters:
        - parcel_image: A 3D numpy array with parcel IDs.
        - mask: A 3D numpy array with 0s for background and 1s for foreground.
        - threshold: The minimum number of voxels required to not ignore a parcel (default is 10).

        Returns:
        - ignored_parcels: A list of parcel IDs that are ignored based on the mask.
        """
        unique_parcels = np.array(list(parcel_header_dict.keys()))
        for parcel_id in unique_parcels:
            # Count voxels for the current parcel_id that are also in the foreground
            voxel_count_mrsi = np.sum((parcel_image == parcel_id) & (mask_mrsi == 1))
            parcel_header_dict[parcel_id]["count"].append(voxel_count_mrsi)

            voxel_count_t1 = np.sum((parcel_image == parcel_id) & (mask_t1 == 1))
            parcel_header_dict[parcel_id]["t1cov"].append(voxel_count_mrsi/voxel_count_t1)

        return  parcel_header_dict



    def parcellate_vectorized(self,met_image4D_data, parcel_image3D, parcel_header_dict, rescale="mean", parcel_concentrations=None):
        """
        Update or create parcel_concentrations with average metabolite concentrations.
        
        Parameters:
        - met_image4D_data: 4D numpy array of metabolite concentrations.
        - parcel_image3D: 3D numpy array mapping voxels to parcels.
        - parcel_header_dict: Dictionary with parcel IDs as keys.
        - rescale: Boolean, whether to rescale concentrations by their mean.
        - parcel_concentrations: Optional dictionary to update with new concentrations.
        
        Returns:
        - Updated or new dictionary with parcel IDs as keys and lists of average concentrations as values.
        """
        parcel_ids_list = list(parcel_header_dict.keys())
        n_metabolites   = met_image4D_data.shape[0]

        if rescale=="mean":
            for idm in range(n_metabolites):
                mean_val = np.mean(met_image4D_data[idm, ...])
                met_image4D_data[idm, ...] /= mean_val if mean_val != 0 else 1
        elif rescale=="zscore":
            for idm in range(n_metabolites):
                mean_val = np.mean(met_image4D_data[idm, ...])
                std_val  = np.std(met_image4D_data[idm, ...])
                if mean_val != 0 and  std_val != 0: 
                    met_image4D_data[idm, ...] = (met_image4D_data[idm, ...]-mean_val)/std_val

        if parcel_concentrations is None:
            parcel_concentrations = {}

        for parcel_id in parcel_ids_list:
            if parcel_id==0:continue
            parcel_mask = parcel_image3D == parcel_id
            for met_idx in range(n_metabolites):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    metabolite_image = met_image4D_data[met_idx, ...]
                    if parcel_mask.sum() != 0:
                        avg_concentration = np.nanmean(metabolite_image[parcel_mask])
                    else:
                        avg_concentration = 0
                    if parcel_id not in parcel_concentrations:
                        parcel_concentrations[parcel_id] = []
                    parcel_concentrations[parcel_id].append(avg_concentration)

        return parcel_concentrations
   

    def mutual_information(self, X_discretized, Y_discretized, n_permutations=1000):
        # Compute the observed mutual information
        Sx = mutual_info_score(X_discretized,X_discretized)
        Sy = mutual_info_score(Y_discretized,Y_discretized)
        observed_mi = 2*mutual_info_score(X_discretized, Y_discretized)
        observed_mi/=(Sx+Sy)
        # Perform permutation test to compute p-value
        permutation_mi = np.zeros(n_permutations)
        for i in range(n_permutations):
            Y_permuted = np.random.permutation(Y_discretized)
            Sy_perm    = mutual_info_score(Y_permuted,Y_permuted)
            perm_mi    = 2*mutual_info_score(X_discretized, Y_permuted)/(Sx+Sy_perm)
            permutation_mi[i] = perm_mi
        # Compute the p-value
        p_value = np.mean(permutation_mi >= observed_mi)
        return observed_mi, p_value



    def leave_one_out(self,simmatrix_ref,mrsirand,parcel_mrsi_np,parcel_header_dict,parcel_label_ids_ignore,N_PERT,corr_mode = "spearman",rescale="zscore"):
        METABOLITES           = ["NAANAAG", "Ins", "GPCPCh", "GluGln", "CrPCr"]
        pairwise_combinations = list(itertools.combinations(METABOLITES, 2))
        mets_remove_list      = [[a] for a in METABOLITES]
        # mets_remove_list.extend([[a,b] for a, b in pairwise_combinations])
        delta_arr = np.zeros(len(mets_remove_list))
        simmatrix_arr = list()
        for ids,mets_remove in enumerate(mets_remove_list):
            ids_to_remove  = list()
            metabolite_arr = copy.deepcopy(np.array(METABOLITES))
            for target in mets_remove:
                _ids = np.where(metabolite_arr==target)[0][0]
                ids_to_remove.append(_ids)
            metabolite_arr = np.delete(metabolite_arr, ids_to_remove)
            debug.warning("leave_one_out: Remove",ids_to_remove)
            mrsirand.metabolites = metabolite_arr
            debug.info(mrsirand.sample_noisy_img4D().shape[0])
            simmatrix, pvalue,_    = self.compute_simmatrix(mrsirand,parcel_mrsi_np,parcel_header_dict,parcel_label_ids_ignore,N_PERT,
                                                            corr_mode = "spearman",rescale="zscore",return_parcel_conc=False)
            simmatrix[pvalue>0.005] = 0
            delta = np.abs(simmatrix-simmatrix_ref).mean()
            debug.info(mets_remove,delta)
            delta_arr[ids] = delta
            simmatrix_arr.append(simmatrix)
            # fig, axs = plt.subplots(1,2, figsize=(16, 12))  # Adjust size as necessary
            # plot_outpath = outfilepath.replace(".npz","_simmatrix")
            # simplt.plot_simmatrix(simmatrix_ref,ax=axs[0],titles=f"All metabolites",
            #                     scale_factor=0.4,
            #                     parcel_ids_positions=parcel_ids_positions,colormap="magma") 
            # simplt.plot_simmatrix(simmatrix,ax=axs[1],titles=f"Leave out {mets_remove}",
            #                     scale_factor=0.4,
            #                     parcel_ids_positions=parcel_ids_positions,
            #                     colormap="magma",show_parcels="H",) 
            # plt.show()
            # break
        return np.array(simmatrix_arr)


    def extract_metabolite_per_parcel(self,array):
        N_metabolites = 5
        if len(array) % N_metabolites != 0:
            raise ValueError("The length of the array must be a multiple of 5.")
        N_pert = len(array) // N_metabolites
        metabolite_conc = np.zeros([N_metabolites,N_pert])
        for i,el in enumerate(array):
            met_pos = i % N_metabolites
            el_pos = round(np.floor(i/N_metabolites))
            metabolite_conc[met_pos,el_pos] = el
        return metabolite_conc

    def compute_simmatrix(self,mrsirand,parcel_mrsi_np,parcel_header_dict,parcel_label_ids_ignore=[],N_PERT=50,corr_mode = "spearman",rescale="mean",n_permutations=10,return_parcel_conc=True):
        parcel_concentrations = None
        for i in track(range(N_PERT), description="Parcellation..."):
            met_image4D_data           = mrsirand.sample_noisy_img4D()
            parcel_concentrations      = self.parcellate_vectorized(met_image4D_data,parcel_mrsi_np,
                                                                    parcel_header_dict,rescale=rescale,
                                                                    parcel_concentrations=parcel_concentrations)
        simmatrix, pvalue         = self.compute_simmatrix_parallel(parcel_concentrations,
                                                                    parcel_label_ids_ignore,
                                                                    corr_mode = corr_mode,
                                                                    show_progress=True,
                                                                    n_permutations=n_permutations)
        parcel_concentrations_np = None
        if return_parcel_conc:
            n_parcels     = len(list(parcel_concentrations.keys()))
            n_metabolites = met_image4D_data.shape[0]
            parcel_concentrations_np = np.zeros([n_parcels,n_metabolites,N_PERT])
            for idx,parcel_id in enumerate(parcel_concentrations):
                array = np.array(parcel_concentrations[parcel_id])
                parcel_concentrations_np[idx] = self.extract_metabolite_per_parcel(array)
        return simmatrix, pvalue, parcel_concentrations_np


    def compute_submatrix(self,worker_id,start_idx, end_idx, sub_parcel_indices, parcel_concentrations, parcel_labels_ignore, corr_mode, n_permutations, mutual_information):
        submatrix_size        = len(sub_parcel_indices)
        all_parcel_labels_ids = list(parcel_concentrations.keys())
        n_parcels             = len(all_parcel_labels_ids)
        submatrix             = np.zeros((submatrix_size, n_parcels))
        submatrix_pvalue      = np.zeros((submatrix_size, n_parcels))
        # for idx, i in enumerate(track(range(start_idx, end_idx))):
        # if start_idx==0:show_progress=True
        # else:show_progress=False
        with Progress() as progress:
            task = progress.add_task(f"[red]Worker {worker_id}", total=len(sub_parcel_indices))
            for i, parcel_idx_i in enumerate(sub_parcel_indices):
                progress.update(task, advance=1)
                if parcel_idx_i in parcel_labels_ignore:
                    continue
                parcel_x = np.array(parcel_concentrations[parcel_idx_i]).flatten()
                for j, parcel_idx_j in enumerate(all_parcel_labels_ids):
                    if parcel_idx_j in parcel_labels_ignore:
                        continue
                    parcel_y = np.array(parcel_concentrations[parcel_idx_j]).flatten()
                    if np.isnan(np.mean(parcel_x)) or np.isnan(np.mean(parcel_y)):
                        continue
                    elif parcel_x.sum() == 0 or parcel_y.sum() == 0:
                        continue
                    try:
                        if corr_mode == "spearman":
                            result = spearmanr(parcel_x, parcel_y, alternative="two-sided")
                            corr = result.statistic
                            pvalue = result.pvalue
                        elif corr_mode == "pearson":
                            result = linregress(parcel_x, parcel_y)
                            corr = result.rvalue
                            pvalue = result.pvalue
                        elif corr_mode == "mi":
                            corr, pvalue = mutual_information(parcel_x, parcel_y, n_permutations)
                        submatrix[i, j]        = corr
                        submatrix_pvalue[i, j] = pvalue
                        try:
                            submatrix[j, i]        = corr
                            submatrix_pvalue[j, i] = pvalue
                        except:pass
                    except Exception as e:
                        debug.warning(f"compute_submatrix: {e}, parcel X {parcel_idx_i} - parcel Y {parcel_idx_j}")
                        continue
        return start_idx, end_idx, submatrix, submatrix_pvalue


    def compute_simmatrix_parallel(self, parcel_concentrations, parcel_labels_ignore=[], corr_mode="pearson", show_progress=False, n_permutations=10, n_workers=32):
        parcel_labels_ids = list(parcel_concentrations.keys())
        parcel_indices    = list(range(len(parcel_labels_ids)))
        n_parcels         = len(parcel_labels_ids)
        simmatrix         = np.zeros((n_parcels, n_parcels))
        simmatrix_pvalue  = np.zeros((n_parcels, n_parcels))
        if corr_mode=="mi":
            for idp in parcel_concentrations.keys():
                parcel_conc = parcel_concentrations[idp] 
                _, bin_edges = np.histogram(parcel_conc, bins="auto")
                parcel_concentrations[idp]  = np.digitize(parcel_conc , bin_edges[:-1])
        batch_size = (n_parcels + n_workers - 1) // n_workers  # Ceiling division
        start = time.time()
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for k in range(n_workers):
                start_idx = k * batch_size
                end_idx   = min((k + 1) * batch_size, n_parcels)
                sub_parcel_ids = list()
                for idx in np.arange(start_idx,end_idx):
                    sub_parcel_ids.append(parcel_labels_ids[idx])
                if start_idx >= n_parcels:
                    break
                futures.append(executor.submit(self.compute_submatrix,k, start_idx, end_idx, sub_parcel_ids, parcel_concentrations, parcel_labels_ignore, corr_mode, n_permutations, self.mutual_information))

            for future in futures:
                start_idx, end_idx, submatrix, submatrix_pvalue = future.result()
                simmatrix[start_idx:end_idx, :] = submatrix
                simmatrix_pvalue[start_idx:end_idx, :] = submatrix_pvalue

        simmatrix = np.nan_to_num(simmatrix, nan=0.0)
        debug.success(f"Time elapsed: {round(time.time()-start)} sec")
        return simmatrix, simmatrix_pvalue

    def combine_p_values_chi2(self,p_value_3d_mat):
        # -2 * log of each p-value, summed over the first axis (N, the number of 2D p-value matrices)
        chi_square_stat = -2 * np.sum(np.log(p_value_3d_mat), axis=0)
        
        # Degrees of freedom: 2 times the number of tests being combined (here, N for each element)
        df = 2 * p_value_3d_mat.shape[0]
        
        # Calculate the combined p-value for each element using the chi-square survival function
        combined_p_value_mat = chi2.sf(chi_square_stat, df)
        
        return combined_p_value_mat


    def get_main_parcel_plot_positions(self,sel_parcel_list,label_list_concat):
        parcel_ids_positions=dict()
        start_l = 0
        for sel_parcel in sel_parcel_list:
            parcel_id_coord_list=list()
            for idp,parcel_label in enumerate(label_list_concat):
                if sel_parcel in parcel_label:
                    parcel_id_coord_list.append(idp)
            a,b = min(parcel_id_coord_list),max(parcel_id_coord_list)+0.5
            # debug.info(max(0,a-0.5),b)
            parcel_ids_positions[sel_parcel] = [max(0,a-0.5),b]
            start_l = b
        label_list_concat = np.array(label_list_concat)
        return parcel_ids_positions, label_list_concat

    def normalizeMetabolites4D(self,met_image4D_data):
        normalized_4d = np.zeros(met_image4D_data.shape)
        for i, image3D in enumerate(met_image4D_data):
            normalized_4d[i] = image3D/np.max(image3D)
        return normalized_4d
    
    def rescale_matrix(self,matrix):
        # Find the min and max values of the matrix
        min_val = np.nanmin(matrix)
        max_val = np.nanmax(matrix)
        
        # Rescale the matrix to the range [-1, 1]
        rescaled_matrix = -1 + 2 * (matrix - min_val) / (max_val - min_val)
        
        return rescaled_matrix


    def find_boundaries(self,mask, axis=0):
        """
        Find the minimum and maximum boundaries along a specified axis of a 3D mask.

        Parameters:
        - mask: np.ndarray, a 3D numpy array representing the mask.
        - axis: int, the axis along which to find the boundaries (0, 1, or 2).

        Returns:
        - A tuple containing the minimum and maximum indices along the specified axis
        where the mask transitions from 0 to 1. Returns (None, None) if no 1s are found.
        """
        assert mask.ndim == 3, "The mask must be a 3D numpy array."
        assert axis in [0, 1, 2], "Axis must be 0, 1, or 2."
        
        # Check if there's at least one 1 in the mask
        if np.any(mask == 1):
            # Find indices where mask is 1
            ones_indices = np.argwhere(mask == 1)
            
            # Extract the indices along the specified axis
            specific_axis_indices = ones_indices[:, axis]
            
            # Find min and max along the specified axis
            min_boundary = specific_axis_indices.min()
            max_boundary = specific_axis_indices.max()
            
            return min_boundary, max_boundary
        else:
            return None, None

    def plot_img_slices(self,images, slice_percentages, titles=None, outpath=None, PLOTSHOW=False, mask=None):
        """
        Plots slices from multiple 3D images at given percentages of the Z dimension within computed boundaries,
        with titles on the left.

        Parameters:
        - images: List of 3D numpy arrays representing the images.
        - slice_percentages: List of percentages (0-100) of the Z dimension to plot within boundaries.
        - titles: List of title prefixes for each image. If None, defaults to 'ImgN Slice'.
        - outpath: Path to save the output figure. If None, figure is not saved.
        - PLOTSHOW: Boolean, if True, shows the plot.
        - z_boundaries: Tuple of (min_z, max_z) boundaries within which to rescale the percentages. If None, use full range.
        """
        if mask is not None:
            z_boundaries = self.find_boundaries(mask,axis=2)
        else: z_boundaries=None
        debug.info("z_boundaries",z_boundaries)
        titles = titles or [f'Img{idx+1} Slice' for idx in range(len(images))]
        titles = titles[::-1]
        num_images = len(images)
        num_slices = len(slice_percentages)
        fig, axes = plt.subplots(num_images, num_slices, figsize=(num_slices * 3, num_images * 3))

        # Ensure axes is a 2D array for easy indexing if there's only one image or one slice
        if num_images == 1:
            axes = np.expand_dims(axes, axis=0)
        if num_slices == 1:
            axes = np.expand_dims(axes, axis=-1)

        for img_idx, img in enumerate(images):
            # Adjust slice indices based on z_boundaries
            min_z, max_z = z_boundaries if z_boundaries else (0, img.shape[-1] - 1)
            z_slices = [int(min_z + (max_z - min_z) * (p / 100.0)) for p in slice_percentages]

            for slice_idx, z in enumerate(z_slices):
                if img_idx==0:
                    debug.info("slice z",z)
                axes[img_idx, slice_idx].imshow(img[:, :, z])
                axes[img_idx, slice_idx].axis('off')  # Hide axes ticks

                # Set title for the row
                title_prefix = titles[img_idx] if titles else f'Img{img_idx + 1} Slice'
                # Use fig.text() to place titles on the left-hand side vertically
                fig.text(0.01, 0.5 * (2 * img_idx + 1) / num_images, title_prefix, va='center', rotation='vertical', fontweight='bold', fontsize=16, color='red')

        plt.tight_layout()
        if PLOTSHOW:
            plt.show()
        if outpath is not None:
            fig.savefig(f'{outpath}.pdf')
            plt.close(fig)
        else:
            plt.close(fig)

    def compute_centroids(self,parcel_image3D,label_ids):
        """
        Compute the centroid of each parcel (label) in a 3D image.

        Parameters:
        - parcel_image3D: a 3D numpy array or a path to a NIfTI image file containing labeled parcels.

        Returns:
        - centroids: list of tuples, where each tuple represents the (x, y, z) centroid coordinates of a parcel.
        """
        centroids = []
        for label in label_ids:
            if label == 0:  # Assuming 0 is the background
                continue
            # Find voxels belonging to the current label
            voxels = np.argwhere(parcel_image3D == label)
            # Compute the centroid of these voxels
            centroid = np.nanmean(voxels, axis=0)
            centroids.append(tuple(centroid))
        return centroids