# -*- coding: utf-8 -*-
"""



"""

import numpy as np
import os, h5py

import astropy.units as u

from panoptes.util.misc import  find_file, _compressed
from panoptes.util.hdf import ensure_hdf5

import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor

# Make plots appear in new windows
from IPython import get_ipython



from panoptes.pinholes import PinholeArray
from panoptes.tmat.tmat import TransferMatrix
from panoptes.reconstruction.gelfgat import gelfgat_poisson, GelfgatResult
from panoptes.detector.base import Data2D

from panoptes.util.misc import identify_filetype




class XrayDetector2D(Data2D):
    def __init__(self, *args):
        self.super().__init__(*args)
        
        
        
    


        
class XrayIP(XrayDetector2D):
    
    def __init__(self, filepath, h4toh5convert_path=None):
        
        # Frst initialize the empty parent class
        self.super().__init__()
        
        self.path = filepath
        
        # If an hdf4 file is given, require and run the conversion utility
        name, ext = os.path.splitext(self.path)
        if ext == '.hdf':
            if h4toh5convert_path is None:
                raise ValueError("Unfortunately you have and hdf4 file: you will "
                                 "need to provide the path to the h4toh5convert "
                                 "exe to convert it.")
                
            # TODO: Use lotus to get an h5 natively so we can skip this conversion
            # kerfuffle 
            self.path = ensure_hdf5(self.path, h4toh5convert_path=h4toh5convert_path)
            
            
        # Check if the file is Data object or some other data file based
        # on the keywords within it
 
        filetype = identify_filetype(self.path)
            
        if filetype == self.__class__.__name__:
            self.load()
            
        elif filetype == 'OMEGA IP':
            xaxis, yaxis, data = self._read_data()
            self.super().__init__(xaxis, yaxis, data)
            
        else:
            raise ValueError(f"File type {filetype} is not supported by "
                             f"class {self.__class__.__name__}")
            




    def _read_data(self):
        """
        Reads data from an OMEGA xray image plate scan file
        """
        
        print("Loading Xray data")
        with h5py.File(self.path, 'r') as f:
            # PSL is a custom unit equivalent to dimensonless
            u_psl = u.def_unit('PSL', u.dimensionless_unscaled)
            data = f['PSL_per_px'][...].T * u_psl
            
            dx = float(f['PSL_per_px'].attrs['pixelSizeX'][...]) * u.um
            dy = float(f['PSL_per_px'].attrs['pixelSizeY'][...]) * u.um
            

        # Arrays of the actual positions for each axis
        xaxis = np.arange(self.data.shape[0])*dx
        yaxis = np.arange(self.data.shape[1])*dy
        
        return xaxis, yaxis, data

        

            
