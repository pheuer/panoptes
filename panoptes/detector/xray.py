# -*- coding: utf-8 -*-
"""



"""

import numpy as np
import os, h5py

import astropy.units as u

from panoptes.util.hdf import ensure_hdf5


from panoptes.detector.detector import Data2D

from panoptes.util.misc import identify_filetype




class XrayIP(Data2D):
    
    def __init__(self, *args, **kwargs):
        
        # Frst initialize the empty parent class
        super().__init__(*args, **kwargs)
     
        # Check if the file is Data object or some other data file based
        # on the keywords within it
        filetype = identify_filetype(self.path)
            
        if filetype == self.__class__.__name__:
            self.load()
            
        elif filetype == 'OMEGA IP':
            xaxis, yaxis, data = self.read_data()
            super().__init__(xaxis, yaxis, data)
            
        else:
            raise ValueError(f"File type {filetype} is not supported by "
                             f"class {self.__class__.__name__}")
            

    def read_data(self):
        """
        Reads data from an OMEGA xray image plate scan file
        """
        
        # If an hdf4 file is given, require and run the conversion utility
        # TODO: Use lotus to get an h5 natively so we can skip this conversion
        # kerfuffle 
        self.path = ensure_hdf5(self.path)
        
        print("Loading Xray data")
        with h5py.File(self.path, 'r') as f:
            # PSL is a custom unit equivalent to dimensonless
            u_psl = u.def_unit('PSL', u.dimensionless_unscaled)
            data = f['PSL_per_px'][...].T * u_psl
            
            dx = float(f['PSL_per_px'].attrs['pixelSizeX'][...]) * u.um
            dy = float(f['PSL_per_px'].attrs['pixelSizeY'][...]) * u.um
            
        # Arrays of the actual positions for each axis
        xaxis = np.arange(data.shape[0])*dx
        yaxis = np.arange(data.shape[1])*dy
        
        return xaxis, yaxis, data

        

            
