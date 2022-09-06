# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 10:04:52 2022

@author: pheu
"""

import os
import pytest
import numpy as np
import astropy.units as u

import h5py

import matplotlib.pyplot as plt

from panoptes.tmat.tmat import TransferMatrix, calculate_tmat
from panoptes.pinholes import _penumbra_model

from scipy.optimize import fmin, curve_fit



def _tmat_mult(tmat, obj, ishape):
    
    obj_flat = obj.flatten()
    
    img_flat = np.matmul(obj_flat, tmat)
    
    return np.reshape(img_flat, ishape)





def _circ_model(x,  amp, center, radius):

    return np.where( np.abs(x-center)< radius, 
                     amp*np.sqrt(radius**2 - (x-center)**2),
                          0)
    
    

def _fit_img(xi, yi, arr):
    
    x = np.sum(arr, axis=-1)
    popt, _ = curve_fit(_circ_model, xi, x, p0=(25, 0, 0.05))
    fit = _circ_model(xi, *popt)
    _, x0, xr = popt
    
    y = np.sum(arr, axis=0)
    popt, _ = curve_fit(_circ_model, yi, y, p0=(25, 0, 0.05))
    fit = _circ_model(yi, *popt)
    _, y0, yr = popt

    r = np.mean(np.array([xr, yr]))
    
    return x0, y0, r

    
    
    
    


class TestSingleApertureTmat1:

    def __init__(self, tmp_path):
        
        self.path = tmp_path
        print(self.path)
        
        self.ishape= (201, 201)
        self.oshape=(41,41)
        

        self.mag = 30
        self.R_ap = 50*u.um
        self.L_det = 350*u.cm
        
        xo = np.linspace(-0.015, 0.015, num=self.oshape[0])*u.cm / self.R_ap
        yo=np.linspace(-0.015, 0.015, num=self.oshape[1])*u.cm / self.R_ap
        xi = np.linspace(-0.6,0.6, num=self.ishape[0])*u.cm / self.R_ap / self.mag
        yi=np.linspace(-0.6,0.6, num=self.ishape[1])*u.cm / self.R_ap / self.mag
        
        ap_xy = np.array([[0,0]])*u.cm / self.R_ap
        
        psf = np.concatenate((np.ones(50), np.zeros(50)))
        psf_ax = np.linspace(0, 2*self.R_ap, num=100) / self.R_ap
        
        
        self.tmat_obj = TransferMatrix(tmp_path)
        self.tmat_obj.set_constants(xo, yo, xi, yi, self.mag, ap_xy, psf=psf, psf_ax=psf_ax)
        self.tmat_obj.set_dimensions(self.R_ap, self.L_det)
        self.tmat_obj.save()
        calculate_tmat(self.tmat_obj)
        
        with h5py.File(self.path, 'r') as f:
            self.tmat = f['tmat'][...]
        
        
        self.xo_scaled = self.tmat_obj.xo_scaled.to(u.um).value
        self.yo_scaled = self.tmat_obj.yo_scaled.to(u.um).value
        self.dxo = np.mean(np.gradient(self.xo_scaled))
        
        self.xi_scaled = self.tmat_obj.xi_scaled.to(u.cm).value
        self.yi_scaled = self.tmat_obj.yi_scaled.to(u.cm).value
        self.dxi = np.mean(np.gradient(self.xi_scaled))
        

        # Create some objects and images
        
        """
        # ********************************************************************
        # Test 1: single centered aperture
        self.obj1 = np.zeros(self.oshape)
        self.obj1[10, 10] = 1
        self.img1 = _tmat_mult(self.tmat, self.obj1, self.ishape)
        
        fig, ax = plt.subplots()
        ax.plot(self.xi_scaled, np.sum(self.img1, axis=-1))
        plt.show()
        
        fig, axarr = plt.subplots(nrows=2)
        fig.subplots_adjust(hspace=0.4)
        ax = axarr[0]
        ax.pcolormesh(self.xo_scaled,
                      self.yo_scaled, 
                      self.obj1.T)
        ax.set_aspect('equal')
        ax.set_title('Obj 1')
        
        ax = axarr[1]
        ax.pcolormesh(self.xi_scaled,
                      self.yi_scaled,
                      self.img1.T)
        ax.set_aspect('equal')
        ax.set_title('Img 1')
        
        x0, y0, r = _fit_img(self.xi_scaled, self.yi_scaled, self.img1)
       
        mag = (r/self.R_ap.to(u.cm).value)
        print(mag)
       
        
        assert (np.abs(x0) < self.dxi and np.abs(y0) < self.dxi)
        assert np.isclose(mag, 30, rtol=0.01)
        """
        
        # ********************************************************************
        # Test 1: single off center aperture
        self.obj2 = np.zeros(self.oshape)
        i, j = np.argmin(np.abs(self.xo_scaled - 50)), np.argmin(np.abs(self.yo_scaled - 0))
        self.obj2[i,j] = 1
        self.img2 = _tmat_mult(self.tmat, self.obj2, self.ishape)
        
        fig, axarr = plt.subplots(nrows=2)
        fig.subplots_adjust(hspace=0.4)
        ax = axarr[0]
        ax.pcolormesh(self.xo_scaled,
                      self.yo_scaled, 
                      self.obj2.T)
        ax.set_aspect('equal')
        ax.set_title('Obj 2')
        
        ax = axarr[1]
        ax.pcolormesh(self.xi_scaled,
                      self.yi_scaled,
                      self.img2.T)
        ax.set_aspect('equal')
        ax.set_title('Img 2')
        
        x0, y0, r = _fit_img(self.xi_scaled, self.yi_scaled, self.img2)
        
        mag = (r/self.R_ap.to(u.cm).value)

        assert  np.abs(y0) < self.dxi
        assert np.isclose(mag, 30, rtol=0.01)
        assert np.isclose(x0, -0.15, rtol=0.05)
        

        
        
        
        
        
        
        

        

    

if __name__ == '__main__':
    tmp_path = os.path.join(os.getcwd(), 'tmat3.h5')
    x = TestSingleApertureTmat1(tmp_path)