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



from panoptes.pinholes import PinholeArray
from panoptes.tmat.tmat import TransferMatrix
from panoptes.reconstruction.gelfgat import gelfgat_poisson, GelfgatResult



        
class XrayIP:
    
    def __init__(self, *args, pinhole_array=None,
                 subregion=None, ):
        """
        arg : path or int
        
            Either a shot number or a filepath directly to the data file
        """
        
        # Make plots appear in new windows
        # If statement guards against running outside of ipython
        # TODO: support plots outside of iPython...
        if get_ipython() is not None:
            get_ipython().run_line_magic('matplotlib', 'inline')
        
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
        
        
        self.data = None
        self.stack = None
        self.tmat = None
        self.reconstruction = None
    
    
        if len(args) == 1:
            self.path = args[0]
            self.load(self.path)
            
       
            
        
        
        
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
        self.xaxis = np.arange(nx)*self.dxpx.to(u.cm)
        self.yaxis = np.arange(ny)*self.dxpx.to(u.cm)
        
       
        
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
            
   
           
    def save(self, path):
        """
        See docstring for "save"
        """ 
        self.path = path
        
        with h5py.File(path, 'a') as grp:
            
            # Empty the group
            for key in grp.keys():
                del grp[key]
            
            if self.pinholes is not None:
                # Write pinhole object
                pinholes_grp = grp.create_group('pinholes')
                self.pinholes.save(pinholes_grp)
            
            if self.data is not None:
     
                data_grp = grp.create_group('data')
                data_grp['data'] = self.data
                data_grp['xaxis'] = self.xaxis.to(u.cm).value
                data_grp['xaxis'].attrs['unit'] = 'cm'
                data_grp['yaxis'] = self.yaxis.to(u.cm).value
                data_grp['yaxis'].attrs['unit'] = 'cm'

      
            
            if self.stack is not None:
                stack_grp = grp.create_group('stack')
                stack_grp['data'] = self.stack
                stack_grp['xaxis'] = self.stack_x.to(u.cm).value
                stack_grp['xaxis'].attrs['unit'] = 'cm'
                stack_grp['yaxis'] = self.stack_y.to(u.cm).value
                stack_grp['yaxis'].attrs['unit'] = 'cm'
                
                
            if self.tmat is not None:
                tmat_grp = grp.create_group('tmat')
                self.tmat.save(tmat_grp)
                
                
            if self.reconstruction is not None:
               recon_grp = grp.create_group('reconstruction')
               self.reconstruction.save(recon_grp)
                
                
                
            
    def load(self, path):
        self.path = path
        
        with h5py.File(path, 'r') as f:
            
            if 'pinholes' in f.keys():
                self.pinholes = PinholeArray()
                self.pinholes.load(f['pinholes'])
                
            if 'data' in f.keys():
                self.data = f['data']['data'][...]
                self.xaxis = f['data']['xaxis'][...] * u.cm
                self.yaxis = f['data']['yaxis'][...]* u.cm
                
            if 'stack' in f.keys():
                self.stack = f['stack']['data'][...]
                self.stack_x = f['stack']['xaxis'][...]* u.cm
                self.stack_y = f['stack']['yaxis'][...]* u.cm
                
            if 'tmat' in f.keys():
                self.tmat = TransferMatrix(f['tmat'])
                
            if 'reconstruction' in f.keys():
                self.reconstruction = GelfgatResult(f['reconstruction'])
            
        
        
        
        
        

    def set_pinhole_array(self, name):
        self.pinholes = PinholeArray()
        self.pinholes.set_pinhole_array(name)
        
        
    def fit_pinholes(self, *args, **kwargs):
        self.pinholes.fit(self.xaxis.to(u.cm).value,
                          self.yaxis.to(u.cm).value,
                          self.data,
                          *args, **kwargs)
        
        
    def stack_pinholes(self):
        
       sx, sy, stack =  self.pinholes.stack(self.xaxis.to(u.cm).value,
                                            self.yaxis.to(u.cm).value, 
                                            self.data)
       
       self.stack_x = sx * u.cm
       self.stack_y = sy * u.cm
       self.stack = stack
       
       self.plot_stack()
       
       
    def plot_stack(self):
        
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.pcolormesh(self.stack_x.to(u.cm).value, 
                      self.stack_y.to(u.cm).value, self.stack.T)
        
        
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
        if get_ipython() is not None:
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
        
        subregion : u.Quantity array with dimensions of length
            [ [xstart, ystart], [xstop, ystop]  ]
            

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
        
        x_c, y_c, data = _compressed(self.xaxis.to(u.cm).value, 
                                     self.yaxis.to(u.cm).value, self.data)
        
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
    
    
    
    def make_tmat(self, tmat_path, xyo=None, R_ap=None, L_det=350*u.cm):
        
            
        if self.stack is None:
            raise ValueError("Stack must be assembled before reconstructing")
            
        if R_ap is None:
            raise ValueError("Keyword R_ap must be set.")
            
            
        if xyo is None:
            xlim = 80
            oshape = (31,31)
            xo = np.linspace(-xlim, xlim, num=oshape[0]) * u.um / R_ap
            yo = np.linspace(-xlim, xlim, num=oshape[1]) * u.um / R_ap
        else:
            xo = xyo[0]/R_ap
            yo = xyo[1]/R_ap

        # Assume for now that we are getting a tmat for a stacked image
        # so there is only one aperture and it is centered
        ap_xy = np.array([[0,0],])*u.cm / R_ap
        mag = float(self.pinholes.mag_r)
        
        psf = np.concatenate((np.ones(50), np.zeros(50)))
        psf_ax = np.linspace(0, 2*R_ap, num=100)/R_ap
        
        xi = self.stack_x/ R_ap /mag
        yi = self.stack_y/ R_ap / mag
        
        
        c = 4
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
            

    
    def reconstruct(self, recon_path, tmat_pat):

        print("Load tmat")
        tmat = TransferMatrix(tmat_path)
        self.tmat = tmat
        
        print("Do reconstruction")
        
        c=4
        data = self.stack[::c, ::c] 
        
        self.reconstruction = gelfgat_poisson(data.flatten(), tmat, 50, h_friction=3)
        
        
        
    
   
    
        

            
if __name__ == '__main__':
    
    data_dir = os.path.join("C:\\","Users","pvheu","Desktop","data_dir")
    h4toh5convert_path = os.path.join('C:\\','Program Files','HDF_Group','H4H5','2.2.5','bin', 'h4toh5convert.exe')
    
    data_dir = os.path.join('C:\\','Users','pheu','Data','data_dir')
    h4toh5convert_path = os.path.join('C:\\','Program Files','HDF_Group','H4H5','2.2.5','bin', 'h4toh5convert.exe')
    
    
    save_path = os.path.join(data_dir, '103955', 'result.h5')
    tmat_path = os.path.join(data_dir, '103955', 'tmat3.h5')
    

        
       
    obj = XrayIP()
    
    if os.path.isfile(save_path):
        print("Loading file")
        obj.load(save_path)
    else:
        obj.set_pinhole_array('D-PF-C-055_A')
        obj.load_data(103955, data_dir=data_dir, h4toh5convert_path=h4toh5convert_path,)
    
        obj.set_subregion( np.array([(0.47, 9.3), (5, 0.73)])*u.cm)
    
        obj.fit_pinholes(rough_adjust={'dx':-0.2, 'dy':-0.3, 'mag_s':35.5, 'rot':-17},
                         auto_select_apertures=True,)
        
        obj.save(save_path)
        
    if obj.stack is None:
        obj.stack_pinholes()
        obj.save(save_path)
        
        
    if not os.path.isfile(tmat_path):
        obj.make_tmat(tmat_path, R_ap=150*u.um, L_det=350*u.cm)
        
    if obj.reconstruction is None:
        obj.reconstruct(save_path, tmat_path)
        obj.save(save_path)
        
        
        
        
    obj = XrayIP(save_path)
    print(obj.reconstruction)
    
    
    
    
    
    
    