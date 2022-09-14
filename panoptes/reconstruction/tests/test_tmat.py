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
import warnings


def _tmat_mult(tmat, obj, ishape, background_value=0.0):
    """
    Multiplies a transfer matrix by a 2D object and reshapes the result
    into an image of shape ishape
    """
    # Flatten the object and add the background pixel
    obj_flat = np.concatenate([obj.flatten(), np.array([background_value,])])
    img_flat = np.matmul(tmat, obj_flat)
    
    return np.reshape(img_flat, ishape)


def _circ_model(x,  amp, center, radius):
    """
    Model of a circle, used for fitting penumbra image 

    """
    return np.where( np.abs(x-center)< radius, 
                     amp*np.sqrt(radius**2 - (x-center)**2),
                          0)
    
def _fit_img(xi, yi, arr):
    """
    Fit the image arr with axes xi and yi to find the center and radius
    of a single pinhole image.
    """
    
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

    


"""
# Currently not supported until we add a safe way of doing it 
def test_initalizing_with_arguments():
    ishape= (201, 201)
    oshape=(31,31)
    
    mag = 10
    R_ap = 50*u.um
    xo = np.linspace(-0.015, 0.015, num=oshape[0])*u.cm / R_ap
    yo=np.linspace(-0.015, 0.015, num=oshape[1])*u.cm / R_ap
    xi = np.linspace(-0.6,0.6, num=ishape[0])*u.cm / R_ap / mag
    yi=np.linspace(-0.6,0.6, num=ishape[1])*u.cm / R_ap / mag
    
    ap_xy = np.array([[0,0]])*u.cm / R_ap

    # Test providing no paramters
    tmat = TransferMatrix()   
    assert tmat.xo is None
    assert tmat.R_ap is None
    
    # Test providing only constants
    tmat = TransferMatrix(xo=xo, yo=yo, xi=xi, yi=yi, mag=mag, ap_xy=ap_xy)
    assert tmat.xo is not None
    assert tmat.R_ap is None
    
    # Assert providing just dimensions
    tmat = TransferMatrix(R_ap=R_ap)
    assert tmat.xo is None
    assert tmat.R_ap == R_ap
    
    
    # Providing some but not all constants gives a warning.
    tmat = TransferMatrix(xo=xo, mag=mag, R_ap=R_ap)
"""
    
   
def test_set_constants_warning_or_error():
    """
    Validate that a warning is raised if dxo*mag<dxi
    """
    ishape= (201, 201)
    oshape=(201,201)
    
    mag = 120
    R_ap = 50*u.um
    xo = np.linspace(-0.015, 0.015, num=oshape[0])*u.cm / R_ap
    yo=np.linspace(-0.015, 0.015, num=oshape[1])*u.cm / R_ap
    xi = np.linspace(-0.6,0.6, num=ishape[0])*u.cm / R_ap / mag
    yi=np.linspace(-0.6,0.6, num=ishape[1])*u.cm / R_ap / mag
    
    ap_xy = np.array([[0,0]])*u.cm / R_ap

    tmat = TransferMatrix()
    
    with pytest.warns():
        tmat.set_constants(xo=xo, yo=yo, xi=xi, yi=yi, mag=mag,
                                   ap_xy=ap_xy, override_validation=True)
    
    with pytest.raises(ValueError):
        tmat.set_constants(xo=xo, yo=yo, xi=xi, yi=yi, mag=mag,
                                   ap_xy=ap_xy)
        
   

    


class TestSingleApertureTmat1:

    def __init__(self, tmp_path):
        
        self.path = tmp_path
        
        ishape= (201, 201)
        oshape=(41,41)
        

        mag = 10
        R_ap = 50*u.um
        L_det = 350*u.cm
        
        xo = np.linspace(-0.015, 0.015, num=oshape[0])*u.cm / R_ap
        yo=np.linspace(-0.015, 0.015, num=oshape[1])*u.cm / R_ap
        xi = np.linspace(-0.6,0.6, num=ishape[0])*u.cm / R_ap / mag
        yi=np.linspace(-0.6,0.6, num=ishape[1])*u.cm / R_ap / mag
        
        ap_xy = np.array([[0,0]])*u.cm / R_ap
        

        
        self.tmat = TransferMatrix()
        
        self.tmat.set_constants(xo=xo, yo=yo, xi=xi, yi=yi, mag=mag,
                                   ap_xy=ap_xy)
        self.tmat.set_dimensions(R_ap=R_ap, L_det=L_det)
    
        self.tmat.calculate_tmat(path=tmp_path)
        
        
        
        self.xo_scaled = self.tmat.xo_scaled.to(u.um).value
        self.yo_scaled = self.tmat.yo_scaled.to(u.um).value
        self.dxo = np.mean(np.gradient(self.xo_scaled))
        
        self.xi_scaled = self.tmat.xi_scaled.to(u.cm).value
        self.yi_scaled = self.tmat.yi_scaled.to(u.cm).value
        self.dxi = np.mean(np.gradient(self.xi_scaled))
        

        # Create some objects and images
        
        self.test_single_centered_aperture()
        
        self.test_off_center_aperture()
        
        
    def test_single_centered_aperture(self):
        
        # ********************************************************************
        # Test 1: single centered aperture
        obj1 = np.zeros(self.tmat.oshape)
        obj1[20, 20] = 1
        img1 = _tmat_mult(self.tmat.tmat, obj1, self.tmat.ishape)
        
        """
        
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
        """
        
        x0, y0, r = _fit_img(self.xi_scaled, self.yi_scaled, img1)
       
        mag = (r/self.tmat.R_ap.to(u.cm).value)
        
        assert (np.abs(x0) < self.dxi and np.abs(y0) < self.dxi)
        assert np.isclose(mag, self.tmat.mag, rtol=0.01)
     
        
    def test_off_center_aperture(self):
        # ********************************************************************
        # Test 2: single off center aperture
        obj2 = np.zeros(self.tmat.oshape)
        i, j = np.argmin(np.abs(self.xo_scaled - 50)), np.argmin(np.abs(self.yo_scaled - 0))
        obj2[i,j] = 1
        img2 = _tmat_mult(self.tmat.tmat, obj2, self.tmat.ishape)
        
        """
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
        """
        
        x0, y0, r = _fit_img(self.xi_scaled, self.yi_scaled, img2)
        
        mag = (r/self.tmat.R_ap.to(u.cm).value)

        assert  np.abs(y0) < self.dxi
        assert np.isclose(mag, self.tmat.mag, rtol=0.01)
        assert np.isclose(x0, -50*self.tmat.mag*1e-4, rtol=0.05)
        

if __name__ == '__main__':
    tmp_path = os.path.join(os.getcwd(), 'tmat3.h5')
    #x = TestSingleApertureTmat1(tmp_path)
    
    test_set_constants_warning_or_error()
    