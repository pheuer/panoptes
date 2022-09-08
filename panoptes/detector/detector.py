# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 09:16:08 2022

@author: pheu
"""

import numpy as np

import h5py
import astropy.units as u
from abc import ABC, abstractmethod
import warnings

from panoptes.util.misc import  _compressed


import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from IPython import get_ipython




class BaseObject(ABC):
    """
    An object with properties that are savable to an HDF5 file or a group
    within an HDF5 file.
    
    The _save and _load methods need to be extended in child classes
    to define exactly what is being loaded and/or saved.
    
    """
    
    def __init__(self):
        # The path to the file
        self.path = None
        # The group within the h5 file where this is stored.
        # Defaults to the root group
        self.group = '/'
    
    
    def _save(self, grp):
        """
        Save this object to an h5 group
        """
        
        # Empty the group before saving new data there
        for key in grp.keys():
            del grp[key]
        
        grp.attrs['class'] = self.__class__.__name__
        
        
    def _load(self, grp):
        """
        Load an object from an h5 group
        """
        pass
    
    
    def save(self, path, group=None):
        """
        Save this object to an h5 file or group within an h5 file
        """ 
        
        if isinstance(path, h5py.File):
            self.path = path.filename
            self.group = path.name
            self._save(path)
            
        elif isinstance(path, h5py.Group):
            self.path = path.file.filename
            self.group = path.name
            self._save(path)
                
        else:
            self.path = path
            self.group = '/'
            with h5py.File(self.path, 'a') as f:
                if group is not None:
                    grp = f[group]
                else:
                    grp = f
                
                self._save(grp)


    def load(self, path, group='/'):
        """
        Load this object from a file
        """
        
        if isinstance(path, (h5py.File, h5py.Group)):
            self.path = path.filename
            # Select a subgroup if the group keyword is used
            path = path[group]
            self.group = path.name
            self._load(path)
            
        else:
            self.path = path
            self.group = group
        
            with h5py.File(self.path, 'r') as f:
                
                if group is not None:
                    grp = f[group]
                else:
                    grp = f
                
                self._load(grp)
    



class Data(BaseObject):
    
    def __init__(self, *args):
        self.data = None
    
        if len(args) == 0:
            pass
        # If one argument is passed, assume it is a data array OR a filepath
        # to a object of this type that we can load
        elif len(args) == 1:
            if isinstance(args[0], (np.ndarray, u.Quantity)):
                self.data = args[0]
            else:
                self.load(args[0])
            
            
        
class Data1D(Data):
    
    def __init__(self, *args):
        # Default all to None
        super().__init__()
        
        self.xaxis = None

        # If no arguements are passed, leave as None
        if len(args)==0:
            pass
        # If one argument is passed, assume it is a data array OR a filepath
        # to a object of this type that we can load
        elif len(args)==1:
            super().__init__(args[0])
        # If two arguments are passed, assume they are xaxis and data
        elif len(args)==2:
            self.xaxis = args[0]
            self.data = args[1]
        else:
            raise ValueError(f"Invalid number of paramters for class "
                             f"{self.__class__.__name__}:"
                             f"{len(args)}")
            
            
    def _save(self, grp):
        
        # Call the parent class
        # in this case, this just saves the class name to the class 
        # attribute of the group
        super()._save(grp)
        
        if self.xaxis is not None:
            if isinstance(self.xaxis, u.Quantity):
                grp['xaxis'] = self.xaxis.to(u.cm).value
                grp['xaxis'].attrs['unit'] = str(self.xaxis.unit)
            else:
                grp['xaxis'] = self.xaxis
                # u.Unit('') = u.dimensionless_unscaled
                grp['xaxis'].attrs['unit'] = ''
            
        if self.data is not None:
            if isinstance(self.xaxis, u.Quantity):
                grp['data'] = self.data.value
                grp['data'].attrs['unit'] = str(self.data.unit)
            else:
                grp['data'] = self.data
                # u.Unit('') = u.dimensionless_unscaled
                grp['data'].attrs['unit'] = ''
                
            
            
    def _load(self, grp):
        
        # Call the parent class method
        # currently does nothing
        super()._load(grp)
        
        if 'xaxis' in grp.keys():
            
            try:
                unit_str = grp['xaxis'].attrs['unit']
                x_unit = u.Unit(unit_str)
            except ValueError:
                warnings.warn(f"Xaxis unit {unit_str} is not a valid astropy unit. "
                              "Assuming units of cm")
                x_unit = u.cm
            self.xaxis = grp['xaxis'][...] * x_unit
        else:
            self.xaxis = None
            
        if 'data' in grp.keys():
            try:
                unit_str = grp['data'].attrs['unit']
                data_unit = u.Unit(unit_str)
            except ValueError:
                warnings.warn("Data unit {unit_str} is not a valid astropy unit. "
                              "Assuming data is dimensionless.")
                data_unit = u.dimensionless_unscaled
                
            self.data = grp['data'][...]  * data_unit
        else:
            self.data = None
            
            
    def flip(self):
        """
        Reverse the order of the data array

        """
        self.data = self.data[::-1]
        
        
    @property
    def dx(self):
        if self.xaxis is None:
            raise ValueError("Cannot access property dx because xaxis is None.")
        
        return np.mean(np.gradient(self.xaxis))
        
        
    # TODO: implement subregion methods for Data1D
        
        


class Data2D(Data1D):
    
    def __init__(self, *args):
        
        # Default all to None
        super().__init__()
        self.yaxis = None
        
        # If no arguments are passed, leave it all as None
        if len(args)==0:
            pass
        # If one argument is passed, assume it is a filepath to a saved 
        # copy of this object or data array
        elif len(args)==1:
            super().__init__(args[0])
        # If three arguments are passed, assume they are x, y, and data
        elif len(args)==3:
            super().__init__(args[0], args[2]) 
            self.yaxis = args[1]
        else:
            raise ValueError("Invalid number of paramters for class Data2D: "
                             f" {len(args)}")
            
    def _save(self, grp):
        
        # Save the xaxis and data using the parent Data1D method
        super()._save(grp)
        
        
        grp.attrs['class'] = 'Data2D'
        
        # Extend Data1D by also saving the yaxis
        if self.yaxis is not None:
            
            if isinstance(self.yaxis, u.Quantity):
                grp['yaxis'] = self.yaxis.to(u.cm).value
                grp['yaxis'].attrs['unit'] = str(self.yaxis.unit)
            else:
                grp['yaxis'] = self.yaxis
                grp['yaxis'].attrs['unit'] = ''
            
        
        
        
    def _load(self, grp):
        
        # Load the xaxis and data using the parent Data1D method
        super()._load(grp)
            
        
        # Extend Data1D to also load the yaxis
        if 'yaxis' in grp.keys():
            try:
                unit_str = grp['yaxis'].attrs['unit']
                y_unit = u.Unit(unit_str)
            except ValueError:
                warnings.warn(f"Yaxis unit {unit_str} is not a valid astropy unit. "
                              "Assuming units of cm")
                y_unit = u.cm
                
            self.yaxis = grp['yaxis'][...] * y_unit
        else:
            self.yaxis = None
            
    
    def flip(self):
        raise ValueError("Flip is not defined for Data2D: use hflip or vflip.")
        
    def hflip(self):
        self.data = self.data[::-1, :]
        
    def vflip(self):
        self.data = self.data[:, ::-1]
        
        
    @property
    def dy(self):
        if self.yaxis is None:
            raise ValueError("Cannot access property dy because yaxis is None.")
        
        return np.mean(np.gradient(self.yaxis))
        
        
        
        
    def plot(self, *args, show=True):
        """
        Makes a plot of the 2D data
        
        Uses a sparse sampling of the array to speed up the operation
        """
        # Clear figure if it already exists

        if len(args) == 0:
            fig = plt.figure(figsize=(10,3))
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        else:
            fig, ax = args
            ax.clear()
        
        
        if np.max(self.data.shape) > 200:
            chunk =int(np.max(self.data.shape)/200)
            
            x, y, data = _compressed(self.xaxis.value, 
                                         self.yaxis.value, self.data,
                                         chunk=chunk)
        
        ax.set_aspect('equal')
        ax.pcolormesh(x, y, 
                      data.T, vmax=10*np.median(data))
        
        ax.set_xlim(np.min(x), np.max(x))
        ax.set_ylim(np.min(y), np.max(y))
        
        ax.set_xlabel(f"X ({self.xaxis.unit})")
        ax.set_ylabel(f"Y ({self.yaxis.unit})")
        

        if show:
            plt.show()
            
        return  fig, ax
        
        
        
    
        
    def select_subregion(self):
         """
         Select a subregion of the data
         """
         # TODO: come up with solutions for interactive plots etc. outside of
         # ipython...
         
         # Switch to qt plotting for interactive plots
         if get_ipython() is not None:
             get_ipython().run_line_magic('matplotlib', 'qt')
         else:
            raise ValueError("Interactive plot features currently only supported "
                             "in ipython enviroments.")
             
         fig, ax = self.plot()
         print("Select the data region")
         cursor = Cursor(ax, color='red', linewidth=1)

         subregion = plt.ginput(2, timeout=-1)
         print(f"Subregion: {subregion}")
         
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
         
         subregion : u.Quantity array with dimensions of length
             [ [xstart, ystart], [xstop, ystop]  ]
             

         """

         # Find the beginning and end of each subregion. 
         # Correct for order by sorting
         x = np.sort(np.array([np.argmin(np.abs(self.xaxis - c[0])) for c in subregion], dtype=np.int64))
         y = np.sort(np.array([np.argmin(np.abs(self.yaxis - c[1])) for c in subregion], dtype=np.int64))
         
         self.data = self.data[x[0]:x[1], y[0]:y[1]]
         self.xaxis = self.xaxis[x[0]:x[1]]
         self.yaxis = self.yaxis[y[0]:y[1]]

                
            
        
        
        
