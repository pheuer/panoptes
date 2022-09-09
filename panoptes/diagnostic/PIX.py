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

import astropy.units as u

from panoptes.util.hdf import ensure_hdf5

import matplotlib.pyplot as plt


from panoptes.diagnostic.diagnostic import Diagnostic
from panoptes.detector.detector import PenumbralImageGelfgat
from panoptes.detector.xray import XrayIP
from panoptes.util.misc import identify_filetype

        
class PIX(Diagnostic):
    
    def __init__(self, *args, pinhole_array = None):
        """
        arg : path or int
        
            filepath directly to the data file or to a saved copy of this
            object
        """
        super().__init__()
        
        self.data = None
     
        if len(args)==0:
            self.data = PenumbralImageGelfgat(pinhole_array)
        
        # If a single argument is given, assume it is the filepath to either
        # a saved PIX object or an x-ray datafile
        elif len(args)==1:
            #TODO: Support lookup by shot number here using lotus
            self.path = args[0]
            
            # If an hdf4 file is given, require and run the conversion utility
            # TODO: Use lotus to get an h5 natively so we can skip this conversion
            # kerfuffle 
            self.path = ensure_hdf5(self.path)
            
            filetype = identify_filetype(self.path)

            if filetype == self.__class__.__name__:
                self.load(self.path)
                
            elif filetype == 'XrayIP' or filetype == 'OMEGA IP':
                xrayip = XrayIP(self.path)
                self.data = PenumbralImageGelfgat(xrayip.xaxis, xrayip.yaxis, 
                                              xrayip.data, 
                                              pinhole_array=pinhole_array)
                
                    
            elif filetype == 'PenumbralImageGelfgat':
               self.data = PenumbralImageGelfgat(self.path)
               
                    
            else:
                raise ValueError(f"File {self.path} data type {filetype} "
                                 "is not valid for class "
                                 f"{self.__class__.__name__}")
                
        else:
            raise ValueError(f"Invalid number of arguments ({len(args)} for "
                             f"class {self.__class__.__name__}.")
                    
    def _save(self, grp):
        super()._save(grp)
        if self.data is not None:
            self.data.save(grp)
        
            
    def _load(self, grp):
        super()._load(grp)
        if 'data' in grp.keys():
            self.data = PenumbralImageGelfgat(grp)
        
        
    def plot_pinholes(self):
        self.data.plot_pinholes()
        

    
    
    
    
    
if __name__ == '__main__':
    
    data_dir = os.path.join("C:\\","Users","pvheu","Desktop","data_dir")
    
    data_dir = os.path.join('C:\\','Users','pheu','Data','data_dir')
    
    
    data_path = os.path.join(data_dir, '103955', 'pcis_s103955_pcis1_-1-[phosphor].hdf' )
    save_path = os.path.join(data_dir, '103955', 'result5.h5')
    tmat_path = os.path.join(data_dir, '103955', 'tmat3.h5')
    

        
       
    
    
    
    
    if os.path.isfile(save_path):
        print("Loading file")
        obj = PIX(save_path)
    else:
        obj = PIX(data_path, pinhole_array = 'D-PF-C-055_A')
        
        obj.data.set_subregion( np.array([(0.47, 9.3), (5, 0.73)])*u.cm)
        
        
        obj.plot_pinholes()
        obj.save(save_path)
        obj.save(save_path)
    
    
        obj.data.fit_pinholes(rough_adjust={'dx':-0.2, 'dy':-0.3, 'mag_s':35.5, 'rot':-17},
                         auto_select_apertures=True,)
        
        obj.save(save_path)
        
    print(type(obj))
    print(type(obj.data))
    print(type(obj.data.pinholes))
    
    if obj.data.stack is None:
        obj.data.stack_pinholes()
        obj.save(save_path)
        
        
    
    if not os.path.isfile(tmat_path):
        obj.data.make_tmat(tmat_path, R_ap=150*u.um, L_det=350*u.cm, oshape=(31, 31))
       
        
    
    if obj.data.reconstruction is None:
        obj.data.reconstruct(save_path, tmat_path)
        obj.save(save_path)
        
        

    obj.data.reconstruction.iter_plot()
    obj.data.plot_reconstruction()
    
    
    
    
    
    