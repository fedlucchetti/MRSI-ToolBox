import numpy as np
import copy
from bids.mridata import MRIData
from graphplot.slices import PlotSlices


class Randomize:
    def __init__(self,mriData,space,metabolites=None):
        self.signal3D_dict = dict()
        self.noise3D_dict  = dict()
        if metabolites==None:
            self.metabolites   = mriData.metabolites
        else:
            self.metabolites   = metabolites
        for metabolite in mriData.metabolites:
            self.signal3D_dict[metabolite]  = mriData.data["mrsi"][metabolite][space]["nifti"].get_fdata().squeeze()
            self.noise3D_dict[metabolite]   = mriData.data["mrsi"][f"{metabolite}-crlb"]["orig"]["nifti"].get_fdata().squeeze()
            self.mask                       = mriData.data["mrsi"]["mask"][space]["nifti"].get_fdata().squeeze()
        print("mask",self.mask.shape)
 
    def perturbate(self,image3D,crlb3D,sigma_scale=2):
        sigma = image3D * crlb3D / 100
        MAX   = image3D.mean() + 3*image3D.std()
        noisy_image3D = np.random.normal(image3D, sigma * sigma_scale)
        noisy_image3D = np.clip(noisy_image3D, 0, MAX)  # Clip values to stay within 0 and MAX
        noisy_image3D[self.mask==0]=0
        return noisy_image3D

   
    def sample_noisy_img4D(self,sigma=2):
        shape4d       = (len(self.metabolites),) + self.signal3D_dict["Ins"].shape
        noisy_image4D = np.zeros(shape4d)
        for idm,metabolite in enumerate(self.metabolites):
            image3D            = self.signal3D_dict[metabolite]
            crlb3D             = self.noise3D_dict[metabolite]
            noisy_image4D[idm] = self.perturbate(image3D,crlb3D,sigma_scale=sigma)
        return noisy_image4D

if __name__=="__main__":
    subject_id = "CHUVA009"
    session    = "V5"
    mrsiData = MRIData(subject_id,session)
    randMRSI = Randomize(mrsiData,"orig")
    noisy_image4D = randMRSI.sample_noisy_img4D()
    pltsl    = PlotSlices()

    image_list = list()
    title_list = list()
    for idi,met in enumerate(randMRSI.signal3D_dict):
        image_list.append(randMRSI.signal3D_dict[met])
        title_list.append(f"{met}-OG")
        image_list.append(noisy_image4D[idi])
        title_list.append(f"{met}-NOISE")
        break
    pltsl.plot_img_slices(image_list,
                        np.linspace(5,95,8),
                        titles   = title_list,
                        outpath  = None,
                        PLOTSHOW = True,
                        mask     = randMRSI.mask)


            
