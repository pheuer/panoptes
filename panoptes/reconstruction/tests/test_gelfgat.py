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
from panoptes.reconstruction.gelfgat import gelfgat_poisson, calculate_termination

from scipy.optimize import fmin, curve_fit


def _tmat_mult(tmat, obj, ishape):
    
    obj_flat = obj.flatten()
    
    img_flat = np.matmul(tmat, obj_flat)
    
    return np.reshape(img_flat, ishape)

def make_tmat(path):
    ishape= (101, 101)
    oshape= (21,21)
    

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
    
    
    tmat_obj = TransferMatrix(tmp_path)
    tmat_obj.set_constants(xo, yo, xi, yi, mag, ap_xy, psf=psf, psf_ax=psf_ax)
    tmat_obj.set_dimensions(R_ap, L_det)
    tmat_obj.save()
    calculate_tmat(tmat_obj)
    



def load_tmat(path):
    with h5py.File(path, 'r') as f:
        tmat = f['tmat'][...]
        
    return tmat



def resolution_wedges():
    oshape = (21,21)
    
    size = [[-150, 150],[-150, 150]]*u.um
    
    arr = np.zeros(oshape)
    hax = np.linspace(size[0][0], size[0][1], num=oshape[0])
    vax = np.linspace(size[1][0], size[1][1], num=oshape[1])
    
    x,y = np.meshgrid(hax, vax, indexing='ij')
    grid_r = np.sqrt(x**2 + y**2)
    grid_theta = np.arctan2(x.to(u.cm).value,y.to(u.cm).value) + np.pi
     

    radius = 0.012 * u.cm # 50 um, a bit bigger than a hotspot

    angle = np.deg2rad(45)
    
    # Determine an optimal nwedges number close to the angle given
    nwedges = np.floor(2*np.pi/angle)
    nbands = int(np.floor(0.5*nwedges))
    nwedges =int(nbands*2)
    angle = 2*np.pi/nwedges

    
    # Coloring of each band
    coloring = np.tile([0,1], nbands)
    edge_angles = np.linspace(0, 2*np.pi, num=nwedges+1) - angle/2
    
    # Go through and color the wedges
    for i in range(nwedges):
        t1 = grid_theta > edge_angles[i]
        t2 = grid_theta < edge_angles[i+1]
        arr = np.where(t1 & t2, coloring[i], arr)
    
    # Create the circle outer edge
    arr = np.where(grid_r < radius, arr, 0)
    
    return hax, vax, arr




def particleify(image, nparticles, background_fract):
    image_flat = image.flatten()
    
    # Draw data particle locations from this, then histogram them back onto the 
    # same grid to create 'data'
    nxi, nyi = image.shape
    xi, yi = np.arange(nxi), np.arange(nyi)
    
    # Create an array of what we will consider to be the center of the cells
    xi_arr, yi_arr = np.meshgrid(xi, yi, indexing='ij')
    xi_arr = xi_arr.flatten() 
    yi_arr = yi_arr.flatten()
    

    indices = np.arange(image.size, dtype=np.int32)
    
    pdist = image_flat/np.sum(image_flat)
    
    nbackground =  int(background_fract*nparticles)
    nsignal = int(nparticles - nbackground)
    
    # Draw indices for signal particles from the image probablility distributon
    i_sig = np.random.choice(indices, size=nsignal, replace=True, p=pdist)
    # Draw indices for background particles from a uniform distribution
    i_bkg = np.random.choice(indices, size=nbackground, replace=True, p=None)
    
    # Concatenate those lists of indices
    i = np.concatenate((i_sig, i_bkg))
       
    # Partice positions are the x and y coordinates for the index drawn
    particle_x = xi_arr[i]
    particle_y = yi_arr[i]
    
    # Create arrays of bins, which include the right point at the end of each axis
    xi_bins = np.append(xi, xi[-1] + 2)
    yi_bins = np.append(yi, yi[-1] + 2)
    
    data, xedges, yedges = np.histogram2d(particle_x, particle_y,
                                                 bins =[xi_bins, yi_bins])

    return data




def test_reconstruction(tmat):
    
    print(tmat.shape)
    
    oshape = (21, 21)
    ishape = (101, 101)
    
    xo, yo, objarr = resolution_wedges()
    
    obj = particleify(objarr, 1e6, 0.2)


    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.pcolormesh(xo.to(u.um).value, 
                  yo.to(u.um).value, 
                  obj.T)
    
    img = _tmat_mult(tmat, obj, ishape)
    
    
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.pcolormesh(img.T)
    
    
    # Reconstruct
    print("Running Reconstruction")
    B, logL, chisq = gelfgat_poisson(img.flatten(), tmat, 300)
    
    print(B.shape)
    
    ind = calculate_termination(B, logL)
    
    
    sol = np.reshape(B[ind, :], oshape)
  
    fig, ax = plt.subplots()
    ax.set_title(f"Termination iter {ind+1}")
    ax.set_aspect('equal')
    ax.pcolormesh(sol.T)
    
    
    
    
    
    objarr /= np.sum(objarr)
    sol /= np.sum(sol)
    nonzero = np.nonzero(objarr)
    
    
    error = np.sum((objarr[nonzero]-sol[nonzero])**2  / objarr[nonzero]**2  )
    print(error)
    
    assert error < 10*(objarr.size)/21**2

    

    

if __name__ == '__main__':

    tmp_path = os.path.join(os.getcwd(), 'tmat3.h5')
    
    if not os.path.isfile(tmp_path):
        tmat = make_tmat(tmp_path)
    tmat = load_tmat(tmp_path)
    
    test_reconstruction(tmat)
    
    