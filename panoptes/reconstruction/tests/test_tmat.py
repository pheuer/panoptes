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

from panoptes.reconstruction.tmat import TransferMatrix


from scipy.optimize import  curve_fit



def _tmat_mult(tmat, obj, ishape, background_value=0.0):
    
    # Flatten the object and add the background pixel
    obj_flat = np.concatenate([obj.flatten(), np.array([background_value,])])
    img_flat = np.matmul(tmat, obj_flat)
    
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
        
        ishape= (201, 201)
        oshape=(41,41)
        

        mag = 30
        R_ap = 50*u.um
        L_det = 350*u.cm
        
        xo = np.linspace(-0.015, 0.015, num=oshape[0])*u.cm / R_ap
        yo=np.linspace(-0.015, 0.015, num=oshape[1])*u.cm / R_ap
        xi = np.linspace(-0.6,0.6, num=ishape[0])*u.cm / R_ap / mag
        yi=np.linspace(-0.6,0.6, num=ishape[1])*u.cm / R_ap / mag
        
        ap_xy = np.array([[0,0]])*u.cm / R_ap
        
        psf = np.concatenate((np.ones(50), np.zeros(50)))
        psf_ax = np.linspace(0, 2*R_ap, num=100) / R_ap
        
        
        self.tmat = TransferMatrix(xo, yo, xi, yi, mag, ap_xy, psf, psf_ax, 
                                   R_ap, L_det)
        
        self.tmat.calculate_tmat(path=tmp_path)
        
        
        
        self.xo_scaled = self.tmat.xo_scaled.to(u.um).value
        self.yo_scaled = self.tmat.yo_scaled.to(u.um).value
        self.dxo = np.mean(np.gradient(self.xo_scaled))
        
        self.xi_scaled = self.tmat.xi_scaled.to(u.cm).value
        self.yi_scaled = self.tmat.yi_scaled.to(u.cm).value
        self.dxi = np.mean(np.gradient(self.xi_scaled))
        

        # Create some objects and images
        
    
        # ********************************************************************
        # Test 1: single centered aperture
        self.obj1 = np.zeros(self.tmat.oshape)
        self.obj1[20, 20] = 1
        self.img1 = _tmat_mult(self.tmat.tmat, self.obj1, self.tmat.ishape)
        
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
       
        mag = (r/self.tmat.R_ap.to(u.cm).value)
        print(r)
        print(self.tmat.R_ap)
        print(mag)
       
        
        assert (np.abs(x0) < self.dxi and np.abs(y0) < self.dxi)
        assert np.isclose(mag, 30, rtol=0.01)
     
        
        # ********************************************************************
        # Test 1: single off center aperture
        self.obj2 = np.zeros(self.tmat.oshape)
        i, j = np.argmin(np.abs(self.xo_scaled - 50)), np.argmin(np.abs(self.yo_scaled - 0))
        self.obj2[i,j] = 1
        self.img2 = _tmat_mult(self.tmat.tmat, self.obj2, self.tmat.ishape)
        
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
        
        mag = (r/self.tmat.R_ap.to(u.cm).value)

        assert  np.abs(y0) < self.dxi
        assert np.isclose(mag, 30, rtol=0.01)
        assert np.isclose(x0, -0.15, rtol=0.05)
        

        
        
        
        
        
        
        

        

    

if __name__ == '__main__':
    tmp_path = os.path.join(os.getcwd(), 'tmat3.h5')
    x = TestSingleApertureTmat1(tmp_path)