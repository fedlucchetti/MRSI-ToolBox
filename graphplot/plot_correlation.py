import numpy as np
from os.path import join, split, exists
import sys
from tools.datautils import DataUtils
from tools.debug import Debug
import matplotlib.pyplot as plt
from connectomics.robustness import NetRobustness

from scipy.stats import linregress, spearmanr



GROUP       = "Mindfulness-Project"
METABOLITES = ["NAANAAG", "Ins", "GPCPCh", "GluGln", "CrPCr"]
FONTSIZE    = 16
dutils = DataUtils()
debug = Debug()
netrobust = NetRobustness()

subject_id = sys.argv[1]
session    = sys.argv[2]
parcel_x_id   = int(sys.argv[3])
parcel_y_id   = int(sys.argv[4])




dirpath  = join(dutils.DATAPATH,GROUP,"derivatives","connectomes",f"sub-{subject_id}",f"ses-{session}","spectroscopy")
filename = f"sub-{subject_id}_ses-{session}_run-01_acq-memprage_atlas-chimeraLFMIHIFIF_desc-scale3grow2mm_dseg_simmatrix.npz"

path = join(dirpath,filename)

if not exists(path):
    debug.error("File not found",path)
    sys.exit()


data = np.load(path)
similarity_matrix     = data["simmatrix_sp"]
parcel_concentrations = data["parcel_concentrations"]
labels_indices        = data["labels_indices"]
labels                = data["labels"][0:-1]

X_label_id = labels_indices[parcel_x_id]
Y_label_id = labels_indices[parcel_y_id]

x_array = parcel_concentrations[parcel_x_id,:,:].flatten()
y_array = parcel_concentrations[parcel_y_id,:,:].flatten()
x_range = np.linspace(x_array.min(),x_array.max(),100)

fig, axs = plt.subplots(1, figsize=(16, 12))  # Adjust size as necessary

corr_arr          = list()
N_pert            = round(len(x_array)/len(METABOLITES))
for i in range(N_pert):
    parcel_conc_x = parcel_concentrations[parcel_x_id,:,i]
    parcel_conc_y = parcel_concentrations[parcel_y_id,:,i]
    slope, intercept, r, p, se = linregress(parcel_conc_x, parcel_conc_y)
    res = spearmanr(parcel_conc_x, parcel_conc_y)
    axs.plot(x_range, intercept + slope*x_range,linewidth=0.2)
    corr_arr.append(netrobust.fisher_z_transform(res.statistic))

corr_agg = netrobust.inverse_fisher_z_transform(np.mean(corr_arr))
debug.info("Aggregated Corr:",corr_agg)

for i_met,met in enumerate(METABOLITES):
    parcel_conc_x = parcel_concentrations[parcel_x_id,i_met]
    parcel_conc_y = parcel_concentrations[parcel_y_id,i_met]
    axs.plot(parcel_conc_x,parcel_conc_y,".",label=f"{met}")
    debug.info({met},parcel_concentrations[:,i_met,:].std())


slope, intercept, r, p, se = linregress(x_array, y_array)
res = spearmanr(x_array, y_array)
axs.plot(x_range, intercept + slope * x_range, 'k', linewidth=2, 
         label=fr"$\rho_{{XY}} = {round(res.statistic, 2)}$" + "\n" +
               fr"$\rho_{{AGG}} = {round(corr_agg, 2)}$" + "\n" +
               fr"$\quad p = {round(res.pvalue, 5)}$")






axs.grid()
axs.set_title(f"{labels[parcel_x_id]} <---> {labels[parcel_y_id]}",fontsize=FONTSIZE)
axs.set_xlabel(f"{labels[parcel_x_id]}",fontsize=FONTSIZE)
axs.set_ylabel(f"{labels[parcel_y_id]}",fontsize=FONTSIZE)
axs.set_xlim(0,3)
axs.set_ylim(0,3)
axs.legend(fontsize=FONTSIZE)









plt.show()




plt.cla()





