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
    
    def __init__(self, *args, pinhole_array=None,
                 subregion=None, ):
        """
        arg : path or int
        
            Either a shot number or a filepath directly to the data file
        """
        
        # A figure, axis tuple for plotting
        self.figax = None
        
        # The subregion of the dataset to use
        if subregion is not None:
            self.subregion = subregion
            self.set_subregion(subregion)
        else:
            self.subregion = None
        


        # Initialize the pinhole array object
        if pinhole_array is not None:
            self.set_pinholes(pinhole_array)
        else:
            self.pinholes = None # Locations of pinholes
            
            
        self.shotnumber = None
    
            
       
            
        
        
        
    def load_data(self, *args, data_dir=None, h4toh5convert_path=None,
                  dxpx=None):
        """
        Loads xray IP data from an hdf5 file 
        
        
        Paramters
        ---------
        
        arg : str or int
            If arg is a string, assume it is a path to the file.
            If arg is an int, assume it is a shot number. 
            
        data_dir : str
            Directory within which to search for the data file by shot number.
            
        h4toh5convert_path : str
            Path to this executable. Sigh.
        
        """
        
        
        # If no arguments are provided, try loading based on the shot number?
        
        if len(args) == 1:
            if isinstance(args[0], str):
                self.data_path = args[0]
                
            elif isinstance(args[0], int):
                self.shotnumber = int(args[0])
                
        if self.shotnumber is not None:
            if data_dir is None:
                raise ValueError("The 'data_dir' keyword is required in order "
                                 "to locate a file based on a shot number.")
            self.data_path = self._find_data(args[0], data_dir)
        else:
            raise ValueError("Data cannot be automatically loaded without "
                             "specifying a shot number.")
            
        # Ensure hdf5
        if h4toh5convert_path is not None:
            self.data_path = ensure_hdf5(self.data_path, h4toh5convert_path=h4toh5convert_path)
        else:
            raise ValueError("Keyword 'h4toh5convert_path' is required")
        
        print("Loading Xray data")
        with h5py.File(self.data_path, 'r') as f:
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
            
        
    
        
    def _find_data(self, id, data_dir):
        """
        This function searches the data dir for a xray IP hdf5 file that 
        matches the shot number provided
        
        """
        
        self.file_dir = os.path.join(data_dir, str(id))
            
        # Verify the data_dir exists
        if not os.path.isdir(self.file_dir):
            raise ValueError(f"The file directory {self.file_dir} does not exist.")
        
        # Find the file path
        path = find_file(self.file_dir, ['phosphor', '.hdf'])
        
        return path
    
    
    # *************************************************************************
    # Methods for loading and saving 
    # *************************************************************************
            
    def save(self, grp):
        """
        Save the data about this object into an h5 group
          
        grp : h5py.Group or path string
            The location to save the h5 data. This could be the root group
            of its own h5 file, or a group within a larger h5 file.
              
            If a string is provided instead of a group, try to open it as
            an h5 file and create the group there
          
        """
        if isinstance(grp, str):
            with h5py.File(grp, 'w') as f:
                self._save(f)
        else:
            self._save(grp)
        
           
    def _save(self, grp):
        """
        See docstring for "save"
        """
        
        # Write pinhole object
        pinholes_grp = grp.create_group('pinholes')
        self.pinholes.save(pinholes_grp)
        
        """
        data_grp = grp['data']
        data_grp['data'] = self.data
        data_grp['xaxis'] = self.xaxis
        data_grp['yaxis'] = self.yaxis
        data_grp['dxpx'] = self.dxpx
        """
        
        stack_grp = grp.create_group('stack')
        stack_grp['data'] = self.stack
        stack_grp['xaxis'] = self.stack_x
        stack_grp['yaxis'] = self.stack_y
        
        

    def set_pinholes(self, name):
        self.pinholes = PinholeArray(name)
        
        
    def fit_pinholes(self, *args, **kwargs):
        self.pinholes.fit(self.xaxis, self.yaxis, self.data,
                          *args, **kwargs)
        
        
    def stack(self):
        
       sx, sy, stack =  self.pinholes.stack(self.xaxis, self.yaxis, self.data)
       
       self.stack_x = sx
       self.stack_y = sy
       self.stack = stack
       
       self.plot_stack()
       
       
    def plot_stack(self):
        
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.pcolormesh(self.stack_x, self.stack_y, self.stack.T)
        
        
    def hreflect(self):
        self.data = self.data[::-1, :]
        self.plot()
        
    def vreflect(self):
        self.data = self.data[:, ::-1]
        self.plot()
        

    def select_subregion(self):
        """
        Select a subregion of the data
        """

        # Switch to qt plotting for interactive plots
        get_ipython().run_line_magic('matplotlib', 'qt')
        fig, ax = self.plot()
        print("Select the data region")
        cursor = Cursor(ax, color='red', linewidth=1)

        subregion = plt.ginput(2, timeout=-1)
        print(f"Subregion: {subregion}")
        
        #cor = np.array([(0.12187499999999751, 9.854687499999999), (5.339062499999997, 0.28984374999999996)])
        self.set_subregion(subregion)
        
        # Switch back to inline plotting so as to not disturb the console 
        # plots
        get_ipython().run_line_magic('matplotlib', 'inline')
        self.plot()
        
        
    def set_subregion(self, subregion):
        """
        Take a subset of the data 
        
        Paramters
        ---------
        
        subregion : list of two tuples
            [ (xstart, ystart), (xstop, ystop)  ]
            

        """
        self.subregion = subregion
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
            ax.scatter(self.pinholes.xy[self.pinholes.use_for_fit,0], 
                       self.pinholes.xy[self.pinholes.use_for_fit,1], color='green')
            
            ax.scatter(self.pinholes.xy[~self.pinholes.use_for_fit,0], 
                       self.pinholes.xy[~self.pinholes.use_for_fit,1], color='red')
            
        plt.show()
            
        return  fig, ax
    
   
    
        

            
if __name__ == '__main__':
    
    data_dir = os.path.join("C:\\","Users","pvheu","Desktop","data_dir")
    h4toh5convert_path = os.path.join('C:\\','Program Files','HDF_Group','H4H5','2.2.5','bin', 'h4toh5convert.exe')
    
    #data_dir = os.path.join('C:\\','Users','pheu','Data','data_dir')
    #h4toh5convert_path = os.path.join('C:\\','Program Files','HDF_Group','H4H5','2.2.5','bin', 'h4toh5convert.exe')
    
    obj = XrayIP( pinhole_array='D-PF-C-055_A')
    
    obj.load_data(103955, data_dir=data_dir, h4toh5convert_path=h4toh5convert_path,)
    
    obj.set_subregion([(0.47, 9.3), (5, 0.73)])
    
    obj.fit_pinholes(rough_adjust={'dx':-0.2, 'dy':-0.3, 'mag_s':35.5, 'rot':-17},
                     auto_select_apertures=True,)
    
    obj.stack()
    
    save_path = os.path.join(data_dir, '103955', 'xray-stack.h5')
    obj.save(save_path)
    