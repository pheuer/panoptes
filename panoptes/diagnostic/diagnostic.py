# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 13:15:23 2022

@author: pheu
"""
import numpy as np
import os, h5py

import astropy.units as u

from panoptes.util.hdf import ensure_hdf5

import matplotlib.pyplot as plt


from panoptes.pinholes import PinholeArray

from panoptes.detector.detector import Data2D
from panoptes.detector.xray import XrayIP

from panoptes.util.misc import identify_filetype
from panoptes.util.base import BaseObject



class Diagnostic(BaseObject):
    
    def __init__(self, *args, **kwargs):
        # TODO: Get the shot number at some point using lotus...
        # Maybe when loading the data into XrayIP?
        self.shotnumber = None
        
        # A figure, axis tuple for plotting
        self.figax = None
        
        super().__init__(**kwargs)
        
        
        
    def _save(self, grp):
        super()._save(grp)
        
        if self.shotnumber is not None:
            grp['shotnumber'] = int(self.shotnumber)
        
    def _load(self, grp):
        super()._load(grp)
      
        if 'shotnumber' in grp.keys():
            self.shotnumber = grp['shotnumber'][...]
    
         
        

        
    
       
    
        