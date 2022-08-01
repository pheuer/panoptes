# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 09:33:39 2022

@author: pheu

import os
from HeuerLib.lle.kodi_analysis import *
data_dir = os.path.join("C:\\","Users","pvheu","Desktop","data_dir")
obj = XrayIP(103955, data_dir=data_dir, pinholes='D-PF-C-055_A')


"""

import numpy as np
import os, h5py

from collections import namedtuple

import astropy.units as u

from panoptes.util.misc import  find_folder, find_file
from panoptes.util.hdf import ensure_hdf5

import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor


from scipy.optimize import curve_fit, minimize, fmin
from scipy.special import erf

# Make plots appear in new windows
from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'qt')
get_ipython().run_line_magic('matplotlib', 'inline')


FrameHeader = namedtuple('FrameHeader', ['number', 'xpos', 'ypos', 
                                         'hits', 'BLENR', 'zpos', 
                                         'x_ind', 'y_ind'])

from cr39py.cr39 import CR39, Cut

from panoptes.pinholes import PinholeArray




            

class KoDIAnalysis:
    
    kodi_data_dir = os.path.join('//expdiv','kodi','ShotData')
    kodi_analysis_dir = os.path.join('//expdiv','kodi','ShotAnalysis')
    
    
    def __init__(self, id, data_dir=None, verbose=True):
        self.verbose = verbose
        
        if data_dir is None:
            self.data_dir = self.kodi_data_dir
        else:
            self.data_dir = data_dir
        self.data_dir = os.path.join(self.data_dir, str(id))
            
        # Verify the data_dir exists
        if not os.path.isdir(self.data_dir):
            raise ValueError(f"The data directory {self.data_dir} does not exist.")

        # Find the CR39 and x-ray data files        
        self.cr39_path = find_file(self.data_dir, [ '.cpsa'])
        self.xray_path = find_file(self.data_dir, ['phosphor', '.hdf'])
        
        self._load_xray()
        
        #self._load_cr39()
        
        
   
    def _load_cr39(self):
        
        self.cr39 = CR39(self.cr39_path, verbose=self.verbose)
        
        self.cr39.frames()
        
        #xax, yax, arr = self.cr39.frames()
            
            
if __name__ == '__main__':
    
    data_dir = os.path.join("C:\\","Users","pvheu","Desktop","data_dir")
    obj = XrayIP(103955, pinhole_array='D-PF-C-055_A', 
                 subregion = [(0.23, 9.6), (5.1, 0.1)],
                 rough_adjust={'dx':-0.2, 'dy':-0.3, 'mag':35.5, 'rot':-17},
                 auto_select_apertures=True,
                 )
    obj.fit_pinholes()