import numpy as np
import matplotlib as plt
import math, sys, os, re
import nibabel as nib
from tqdm import tqdm

from os.path import join
from tools.debug import Debug


debug=Debug()

def find_root_path():
    current_path = os.path.abspath(__file__)
    while os.path.basename(current_path) != "Connectome":
        current_path = os.path.dirname(current_path)
        if current_path == os.path.dirname(current_path):
            return None
    return current_path

class DataUtils(object):
    def __init__(self):
        self.ROOTPATH       = find_root_path()
        # self.DATAPATH       = join("/media","flucchetti","NSA1TB1","Connectome","Data")
        self.DEVPATH        = join(self.ROOTPATH,"Dev")
        self.DEVANALYSEPATH = join(self.ROOTPATH,"Dev","MRSI-ToolBox")
        self.RECONPATH      = join(self.DEVPATH,"CreateMaps","MRSI_Recon_3D_V3")
        self.ANARESULTSPATH = join(self.DEVPATH,"MRSI-ToolBox","results")
        self.ANALOGPATH     = join(self.DEVPATH,"MRSI-ToolBox","logs")
        self.BIDS_STRUCTURE_PATH = join(self.DEVANALYSEPATH,"bids","structure.json")

        os.makedirs(self.ANALOGPATH ,exist_ok=True)
        self.SAFEDRIVE      = join("/media","veracrypt2")
        self.DATAPATH       = join(self.SAFEDRIVE,"Connectome","Data")

if __name__=='__main__':
    u = DataUtils()






