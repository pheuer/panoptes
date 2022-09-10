# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 09:16:08 2022

@author: pheu
"""

import numpy as np

import h5py
import astropy.units as u

import warnings

from panoptes.util.misc import  _compressed


import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from IPython import get_ipython


from panoptes.util.base import BaseObject
from panoptes.pinholes import PinholeArray
from panoptes.reconstruction.tmat import TransferMatrix
from panoptes.reconstruction.gelfgat import gelfgat_poisson, GelfgatResult

class Data(BaseObject):
    
    def __init__(self, *args):
        
        self.data = None
        
        super().__init__()
        
        if len(args) == 0:
            pass
        # If one argument is passed, assume it is a data array OR a filepath
        # to a object of this type that we can load
        elif len(args) == 1:
            if isinstance(args[0], (np.ndarray, u.Quantity)):
                self.data = args[0]
            else:
                self.load(args[0])
                
    def _save(self, grp):
        
        # Call the parent class
        # in this case, this just saves the class name to the class 
        # attribute of the group
        super()._save(grp)
         
        if self.data is not None:
            if isinstance(self.data, u.Quantity):
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
        
         if 'data' in grp.keys():
             try:
                 unit_str = grp['data'].attrs['unit']
                 data_unit = u.Unit(unit_str)
             except ValueError:
                 warnings.warn(f"Data unit {unit_str} is not a valid astropy unit. "
                               "Assuming data is dimensionless.")
                 data_unit = u.dimensionless_unscaled
                 
             self.data = grp['data'][...]  * data_unit
         else:
             self.data = None
            
            
        
class Data1D(Data):
    
    def __init__(self, *args):
        
        self.xaxis = None
        
        # Default all to None
        super().__init__()
        
        

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
                grp['xaxis'] = self.xaxis.value
                grp['xaxis'].attrs['unit'] = str(self.xaxis.unit)
            else:
                grp['xaxis'] = self.xaxis
                # u.Unit('') = u.dimensionless_unscaled
                grp['xaxis'].attrs['unit'] = ''
            
            
            
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
        
        self.yaxis = None
        
        # Default all to None
        super().__init__()
        
        
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
        
        # Extend Data1D by also saving the yaxis
        if self.yaxis is not None:
            
            if isinstance(self.yaxis, u.Quantity):
                grp['yaxis'] = self.yaxis.value
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

                
            
        
        
class PinholeArrayImage(Data2D):
    
    def __init__(self, *args, pinhole_array = None, **kwargs):
        
        # The PinholeArray object
        self.pinholes = None
        
        # The stacked images
        self.stack = None
        
        super().__init__(*args, **kwargs)
        
        if pinhole_array is not None:
            # pinhole_array can be either a PinholeArray object or a 
            # name of a defined pinhole array
            
            # If a pinhole array object is provided, assign it
            if isinstance(pinhole_array, PinholeArray):
                self.pinholes = pinhole_array
            # If a string is provided, assume it is a pinhole array name or
            # a filepath
            elif isinstance(pinhole_array, str):
                self.pinholes = PinholeArray(pinhole_array)
                print(self.pinholes)
            else:
                raise ValueError("Invalid pinhole_array keyword: "
                                 f"{pinhole_array}, type {type(pinhole_array)}")

        
        
        
    def _save(self, grp):
        super()._save(grp)
        
        if self.pinholes is not None:
            # Write pinhole object
            pinholes_grp = grp.create_group('pinholes')
            self.pinholes.save(pinholes_grp)
           
        if self.stack is not None:
            stack_grp = grp.create_group('stack')
            self.stack.save(stack_grp)
            
    def _load(self, grp):
        super()._load(grp)
        
        if 'pinholes' in grp.keys():
            self.pinholes = PinholeArray()
            self.pinholes._load(grp['pinholes'])
            
        if 'stack' in grp.keys():
            self.stack = Data2D(grp['stack'])
            
            
    def fit_pinholes(self, *args, **kwargs):
        self.pinholes.fit(self.xaxis.to(u.cm).value,
                          self.yaxis.to(u.cm).value,
                          self.data,
                          *args, **kwargs)
        
        
    def stack_pinholes(self):
       sx, sy, stack =  self.pinholes.stack(self.xaxis.to(u.cm).value,
                                            self.yaxis.to(u.cm).value, 
                                            self.data)
       
       self.stack = Data2D(sx*u.cm, sy*u.cm, stack)
       self.plot_stack()
       
       
    def plot_stack(self, show=True):
        
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.pcolormesh(self.stack.xaxis.to(u.cm).value, 
                      self.stack.yaxis.to(u.cm).value, self.stack.data.T)   
        
        if show:
            plt.show()
        
        
    def plot_pinholes(self, *args):
        """
        Makes a plot of the x-ray data
        
        Uses a sparse sampling of the array to speed up the operation
        """
        
        fig, ax = self.plot(*args, show=False)
        
        if self.pinholes is not None:
            ax.scatter(self.pinholes.xy[self.pinholes.use_for_fit,0], 
                       self.pinholes.xy[self.pinholes.use_for_fit,1], color='green')
            
            ax.scatter(self.pinholes.xy[~self.pinholes.use_for_fit,0], 
                       self.pinholes.xy[~self.pinholes.use_for_fit,1], color='red')
            
        plt.show()
            
        return  fig, ax
    
    
    
class PenumbralImageGelfgat(PinholeArrayImage):
    
    def __init__(self, *args, **kwargs):
        
        # An instance of TransferMatrix
        self.tmat = None
        # An instance of GelfgatResult
        self.reconstruction = None
        
        super().__init__(*args, **kwargs)  
        
        
    def _save(self, grp):
        super()._save(grp)
            
        if self.tmat is not None:
            tmat_grp = grp.create_group('tmat')
            self.tmat.save(tmat_grp)
            
        if self.reconstruction is not None:
           recon_grp = grp.create_group('reconstruction')
           self.reconstruction.save(recon_grp)


    def _load(self, grp):
        
        super()._load(grp)
          
        if 'tmat' in grp.keys():
            self.tmat = TransferMatrix(grp['tmat'])
            
        if 'reconstruction' in grp.keys():
            self.reconstruction = GelfgatResult(grp['reconstruction'])

    
    def make_tmat(self, tmat_path, xyo=None, R_ap=None, L_det=350*u.cm, oshape=(101, 101),
                  mag=None):
        
            
        if self.stack is None:
            raise ValueError("Stack must be assembled before reconstructing")
            
        if R_ap is None:
            raise ValueError("Keyword R_ap must be set.")
            
            
        if xyo is None:
            xlim = 200
            xo = np.linspace(-xlim, xlim, num=oshape[0]) * u.um / R_ap
            yo = np.linspace(-xlim, xlim, num=oshape[1]) * u.um / R_ap
        else:
            xo = xyo[0]/R_ap
            yo = xyo[1]/R_ap
    
        # Assume for now that we are getting a tmat for a stacked image
        # so there is only one aperture and it is centered
        ap_xy = np.array([[0,0],])*u.cm / R_ap
        
        if mag is None:
            mag = float(self.pinholes.mag_r)
        
        psf = np.concatenate((np.ones(50), np.zeros(50)))
        psf_ax = np.linspace(0, 2*R_ap, num=100)/R_ap
        
        xi = self.stack.xaxis/ R_ap /mag
        yi = self.stack.yaxis/ R_ap / mag
        
        
        c = 1
        xi = xi[::c]
        yi = yi[::c]
        
        
        tmat = TransferMatrix(xo.to(u.dimensionless_unscaled).value,
                              yo.to(u.dimensionless_unscaled).value,
                              xi.to(u.dimensionless_unscaled).value,
                              yi.to(u.dimensionless_unscaled).value,
                              mag,
                              ap_xy.to(u.dimensionless_unscaled).value,
                              psf,
                              psf_ax.to(u.dimensionless_unscaled).value,
                              R_ap, 
                              L_det)
        
        
    
        tmat.calculate_tmat(tmat_path)
        
        self.tmat = tmat
            
    
    
    def reconstruct(self, tmat_path):
    
        print("Load tmat")
        tmat = TransferMatrix(tmat_path)
        self.tmat = tmat
        
        print("Do reconstruction")
        
        c=1
        data = self.stack.data[::c, ::c] 
        
        self.reconstruction = gelfgat_poisson(data.flatten(), tmat, 50, h_friction=3)
        
        
    def plot_reconstruction(self):
        
        if self.reconstruction is None:
            raise ValueError("Run reconstruction first!")
            
        
        xo = self.tmat.xo_scaled.to(u.um).value
        yo = self.tmat.yo_scaled.to(u.um).value
        img = self.reconstruction.solution
        
        
        # Pick vmax from the center region to avoid edge artifacts
        nx, ny = img.shape
        vmax = np.max(img[int(0.33*nx):int(0.66*nx),
                          int(0.33*ny):int(0.66*ny),])
        
        fig, ax = plt.subplots()
        ax.pcolormesh(xo, yo, img.T, vmax=vmax)
        ax.set_aspect('equal')
        ax.set_xlabel("X (um)")
        ax.set_ylabel("Y (um)")        
        
