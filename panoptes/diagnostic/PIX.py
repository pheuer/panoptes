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


from panoptes.pinholes import PinholeArray
from panoptes.reconstruction.tmat import TransferMatrix
from panoptes.reconstruction.gelfgat import gelfgat_poisson, GelfgatResult
from panoptes.detector.detector import Data2D
from panoptes.detector.xray import XrayIP
from panoptes.diagnostic.diagnostic import Diagnostic
from panoptes.util.misc import identify_filetype

        
class PIX(Diagnostic):
    
    def __init__(self, *args, pinhole_array = None):
        """
        arg : path or int
        
            Either a shot number or a filepath directly to the data file
        """
        super().__init__()
        
        # A figure, axis tuple for plotting
        self.figax = None
        
        self.pinholes = None
        self.data = None
        self.stack = None
        self.tmat = None
        self.reconstruction = None
        
    
        # TODO: Get the shot number at some point using lotus...
        # Maybe when loading the data into XrayIP?
        self.shotnumber = None
        
        
        if len(args)==0:
            self.path = None
        
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
                
            elif filetype == 'XrayIP':
                with h5py.File(self.path, 'r') as f:
                    self._load_data(f)
                    
            elif filetype == 'OMEGA IP':
               self.data = XrayIP(self.path)
               
                    
            else:
                raise ValueError(f"File {self.path} data type {filetype} "
                                 "is not valid for class "
                                 f"{self.__class__.__name__}")
                    
    
        if pinhole_array is not None:
            self.set_pinhole_array(pinhole_array)




    
    
    # *************************************************************************
    # Methods for loading and saving 
    # *************************************************************************
            
   
           
    def _save(self, grp):
        super()._save(grp)
        
        if self.pinholes is not None:
            # Write pinhole object
            pinholes_grp = grp.create_group('pinholes')
            self.pinholes.save(pinholes_grp)
        
        if self.data is not None:
            data_grp = grp.create_group('data')
            self.data.save(data_grp)
            
        if self.stack is not None:
            stack_grp = grp.create_group('stack')
            self.stack.save(stack_grp)
            
        if self.tmat is not None:
            tmat_grp = grp.create_group('tmat')
            self.tmat.save(tmat_grp)
            
        if self.reconstruction is not None:
           recon_grp = grp.create_group('reconstruction')
           self.reconstruction.save(recon_grp)
                
                
                
            
    def _load(self, grp):
        super()._load(grp)
        
        if 'pinholes' in grp.keys():
            self.pinholes = PinholeArray()
            self.pinholes._load(grp['pinholes'])

        
        if 'data' in grp.keys():
            self.data = Data2D(grp['data'])

            
        if 'stack' in grp.keys():
            self.stack = Data2D(grp['stack'])
            
        if 'tmat' in grp.keys():
            self.tmat = TransferMatrix(grp['tmat'])
            
        if 'reconstruction' in grp.keys():
            self.reconstruction = GelfgatResult(grp['reconstruction'])

        
        

    def set_pinhole_array(self, name):
        self.pinholes = PinholeArray()
        self.pinholes.set_pinhole_array(name)
        
        
    def fit_pinholes(self, *args, **kwargs):
        self.pinholes.fit(self.data.xaxis.to(u.cm).value,
                          self.data.yaxis.to(u.cm).value,
                          self.data.data,
                          *args, **kwargs)
        
        
    def stack_pinholes(self):
       sx, sy, stack =  self.pinholes.stack(self.data.xaxis.to(u.cm).value,
                                            self.data.yaxis.to(u.cm).value, 
                                            self.data.data)
       
       self.stack = Data2D(sx*u.cm, sy*u.cm, stack)
       self.plot_stack()
       
       
    def plot_stack(self):
        
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.pcolormesh(self.stack.xaxis.to(u.cm).value, 
                      self.stack.yaxis.to(u.cm).value, self.stack.data.T)
        
        

    def plot(self, *args):
        """
        Makes a plot of the x-ray data
        
        Uses a sparse sampling of the array to speed up the operation
        """
        
        fig, ax = self.data.plot(*args, show=False)
        
        if self.pinholes is not None:
            ax.scatter(self.pinholes.xy[self.pinholes.use_for_fit,0], 
                       self.pinholes.xy[self.pinholes.use_for_fit,1], color='green')
            
            ax.scatter(self.pinholes.xy[~self.pinholes.use_for_fit,0], 
                       self.pinholes.xy[~self.pinholes.use_for_fit,1], color='red')
            
        plt.show()
            
        return  fig, ax
    
    
    
    def make_tmat(self, tmat_path, xyo=None, R_ap=None, L_det=350*u.cm, oshape=(101, 101)):
        
            
        if self.stack is None:
            raise ValueError("Stack must be assembled before reconstructing")
            
        if R_ap is None:
            raise ValueError("Keyword R_ap must be set.")
            
            
        if xyo is None:
            xlim = 80
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
        
        xi = self.stack.xaxis/ R_ap /mag
        yi = self.stack.yaxis/ R_ap / mag
        
        
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
        data = self.stack.data[::c, ::c] 
        
        self.reconstruction = gelfgat_poisson(data.flatten(), tmat, 50, h_friction=3)
        
        
    def plot_reconstruction(self):
        
        if self.reconstruction is None:
            raise ValueError("Run reconstruction first!")
            
        
        xo = self.tmat.xo_scaled.to(u.um).value
        yo = self.tmat.yo_scaled.to(u.um).value
        img = self.reconstruction.solution
        
        
        fig, ax = plt.subplots()
        ax.pcolormesh(xo, yo, img.T)
        ax.set_aspect('equal')
        ax.set_xlabel("X (um)")
        ax.set_ylabel("Y (um)")        
        
        
    
   
    
        

            
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
        
        
        obj.plot()
        obj.save(save_path)
        obj.save(save_path)
    
    
        obj.fit_pinholes(rough_adjust={'dx':-0.2, 'dy':-0.3, 'mag_s':35.5, 'rot':-17},
                         auto_select_apertures=True,)
        
        obj.save(save_path)
        
      
    
    
    
    if obj.stack is None:
        obj.stack_pinholes()
        obj.save(save_path)
        
        
    
    if not os.path.isfile(tmat_path):
        obj.make_tmat(tmat_path, R_ap=150*u.um, L_det=350*u.cm, oshape=(31, 31))
       
        
    
    if obj.reconstruction is None:
        obj.reconstruct(save_path, tmat_path)
        obj.save(save_path)
        
        

    obj.reconstruction.iter_plot()
    obj.plot_reconstruction()
    
    
    
    
    
    