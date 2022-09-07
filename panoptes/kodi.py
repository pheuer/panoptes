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

if get_ipython() is not None:
    get_ipython().run_line_magic('matplotlib', 'inline')


FrameHeader = namedtuple('FrameHeader', ['number', 'xpos', 'ypos', 
                                         'hits', 'BLENR', 'zpos', 
                                         'x_ind', 'y_ind'])

from cr39py.cr39 import CR39, Cut

from panoptes.pinholes import PinholeArray




            

class KoDI:
    
    def __init__(self, *args, plots=True):

       
        
        
   
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