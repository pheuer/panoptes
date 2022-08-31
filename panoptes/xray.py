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

from panoptes.util.misc import  find_file, _compressed
from panoptes.util.hdf import ensure_hdf5

import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor

# Make plots appear in new windows
from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'qt')
get_ipython().run_line_magic('matplotlib', 'inline')


from panoptes.pinholes import PinholeArray



        
class XrayIP:
    
    def __init__(self, *args, data_dir=None, pinhole_array=None, dxpx=None,
                 subregion=None, h4toh5convert_path=None):
        """
        arg : path or int
        
            Either a shot number or a filepath directly to the data file
        """
        
        self.figax = None
        self.penumbra = None
        
        self.subregion = subregion

        if pinhole_array is not None:
            self.set_pinholes(pinhole_array)
        else:
            self.pinholes = None # Locations of pinholes
            
        # If the first argument is a file path, load the file directly
        # if not, assume it is a shot number and look for it 
        if isinstance(args[0], str):
            self.path = args[0]
        else:
            if data_dir is None:
                raise ValueError("The 'data_dir' keyword is required in order "
                                 "to locate a file based on a shot number.")
            
            self.path = self._find_data(args[0], data_dir)
            

        
    
        self._load_xray(self.path, h4toh5convert_path=h4toh5convert_path)
        
        
        if self.subregion is not None:
            self._subregion(self.subregion)
        else:
            self._select_subregion()
        
    def _find_data(self, id, data_dir):
        
        
        self.file_dir = os.path.join(data_dir, str(id))
            
        # Verify the data_dir exists
        if not os.path.isdir(self.file_dir):
            raise ValueError(f"The file directory {self.file_dir} does not exist.")
        
        # Find the file path
        path = find_file(self.file_dir, ['phosphor', '.hdf'])
        
        return path
                
    
    def _load_xray(self, path, dxpx=None, h4toh5convert_path=None):
        
        # Ensure hdf5
        if h4toh5convert_path is not None:
            path = ensure_hdf5(path, h4toh5convert_path=h4toh5convert_path)
        else:
            raise ValueError("Keyword 'h4toh5convert_path' is required")
        
        print("Loading Xray data")
        with h5py.File(path, 'r') as f:
            self.data = f['PSL_per_px'][...].T
            
        nx, ny = self.data.shape
        
        # Set the spatial resolution of the IP
        if dxpx is None:
            self.dxpx = 25e-4 *u.cm
        else:
            self.dxpx = dxpx
        
        # Arrays of the actual positions for each axis
        self.xaxis = np.arange(nx)*self.dxpx.to(u.cm).value
        self.yaxis = np.arange(ny)*self.dxpx.to(u.cm).value
        
       
        
        self.plot()
        

    def set_pinholes(self, name):
        self.pinholes = PinholeArray(name)
        
        
    def fit_pinholes(self, *args, **kwargs):
        self.pinholes.fit(self.xaxis, self.yaxis, self.data,
                          *args, **kwargs)
        
        
    def hreflect(self):
        self.data = self.data[::-1, :]
        self.plot()
        
    def vreflect(self):
        self.data = self.data[:, ::-1]
        self.plot()
        

        
    def _select_subregion(self, ):
        """
        Select a subregion of the data
        """
        
        # Switch to qt plotting for interactive plots
        get_ipython().run_line_magic('matplotlib', 'qt')
        ax = self.plot()
        print("Select the data region")
        cursor = Cursor(ax, color='red', linewidth=1)

        subregion = plt.ginput(2, timeout=-1)
        print(f"Subregion: {subregion}")
        
        #cor = np.array([(0.12187499999999751, 9.854687499999999), (5.339062499999997, 0.28984374999999996)])
        self._subregion(subregion)
        
        # Switch back to inline plotting so as to not disturb the console 
        # plots
        get_ipython().run_line_magic('matplotlib', 'inline')
        self.plot()
        
        
    def _subregion(self, subregion):
        x = np.sort(np.array([np.argmin(np.abs(self.xaxis - c[0])) for c in subregion], dtype=np.int64))
        y = np.sort(np.array([np.argmin(np.abs(self.yaxis - c[1])) for c in subregion], dtype=np.int64))
        
        self.data = self.data[x[0]:x[1], y[0]:y[1]]
        self.xaxis = self.xaxis[x[0]:x[1]]
        self.yaxis = self.yaxis[y[0]:y[1]]
    
    
    
    def plot(self, *args):
        """
        Makes a plot of the x-ray data
        
        Uses a sparse sampling of the array to speed up the operation
        """
        # Clear figure if it already exists

        if len(args) == 0:
            fig = plt.figure(figsize=(10,3))
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        else:
            fig, ax = args
            ax.clear()
        
        x_c, y_c, data = _compressed(self.xaxis, self.yaxis, self.data)
        
        ax.set_aspect('equal')
        ax.pcolormesh(x_c, y_c, 
                      data.T, vmax=10*np.median(data))
        
        ax.set_xlim(np.min(x_c), np.max(x_c))
        ax.set_ylim(np.min(y_c), np.max(y_c))
        
        if self.pinholes is not None:
            ax.scatter(self.pinholes.xy[self.pinholes.use,0], 
                       self.pinholes.xy[self.pinholes.use,1], color='green')
            
            ax.scatter(self.pinholes.xy[~self.pinholes.use,0], 
                       self.pinholes.xy[~self.pinholes.use,1], color='red')
            
        plt.show()
            
        return  fig, ax
    
   
    
        

            
if __name__ == '__main__':
    
    data_dir = os.path.join("C:\\","Users","pvheu","Desktop","data_dir")
    
    h4toh5convert_path = os.path.join('C:\\','Program Files','HDF_Group','H4H5','2.2.5','bin', 'h4toh5convert.exe')
    
    obj = XrayIP(103955, pinhole_array='D-PF-C-055_A', 
                 data_dir=data_dir,
                 subregion = [(0.23, 9.6), (5.1, 0.1)],
                 h4toh5convert_path=h4toh5convert_path,
                 )
    obj.fit_pinholes(rough_adjust={'dx':-0.2, 'dy':-0.3, 'mag_s':35.5, 'rot':-17},
                     auto_select_apertures=True,)