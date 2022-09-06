# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 21:14:48 2022

@author: pvheu
"""
import numpy as np
import astropy.units as u
import os, h5py

from panoptes.tmat.tmat import TransferMatrix, calculate_tmat
from panoptes.reconstruction.gelfgat import gelfgat_poisson, calculate_termination

import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    oshape = (50, 50)
    
    #__spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    
    data_dir = os.path.join("C:\\","Users","pvheu","Desktop","data_dir", "103955")
    data_dir = os.path.join('C:\\','Users','pheu','Data','data_dir', "103955")

    datafile = os.path.join(data_dir, 'xray-stack.h5')
    tmat_path = os.path.join(data_dir, 'tmat.h5')


    with h5py.File(datafile, 'r') as f:
        xi = f['stack']['xaxis'][...] * u.cm
        yi = f['stack']['yaxis'][...] * u.cm
        data = f['stack']['data'][...]

        R_ap = 0.5*f['pinholes']['pinhole_info']['diameter'][...] * u.um
        mag = f['pinholes']['mag_r'][...]
        
        
    L_det = 350 * u.cm
      
    print(f"R_ap: {R_ap}")
    print(f"Mag: {mag}")
    psf = np.concatenate((np.ones(50), np.zeros(50)))
    psf_ax = np.linspace(0, 2*R_ap, num=100)/R_ap
    
    
    print(f"xi shape: {xi.shape}")
    print(f"yi shape: {yi.shape}")
    
    

    
    xlim = 60
    xo = np.linspace(-xlim, xlim, num=oshape[0]) * u.um / R_ap
    yo = np.linspace(-xlim, xlim, num=oshape[1]) * u.um / R_ap
    
    xi = xi/R_ap/mag
    yi = yi/R_ap/mag
    ap_xy = np.array([[0,0],])*u.cm / R_ap
    
    
   

    # Thin data for testing
    c = 1
    data = data[::c, ::c]
    xi = xi[::c]
    yi = yi[::c]
    
    
    dxi = np.mean(np.gradient(xi)).to(u.dimensionless_unscaled).value
    dxo = np.mean(np.gradient(xo)).to(u.dimensionless_unscaled).value
    print(f"dxi: {dxi:.1e}, dxo: {dxo:.1e}")
    
    assert dxo > dxi, f"dxo must be greater than dxi, but {dxo}<{dxi}"
    
    
    nxi, nyi = xi.size, yi.size
    nxo, nyo = xo.size, yo.size

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.pcolormesh((xi*R_ap*mag).to(u.cm).value, (yi*R_ap*mag).to(u.cm).value, data.T)
    plt.show()
    
    #if not os.path.isfile(tmat_path):
    t = TransferMatrix(tmat_path)

    print("Calculating tmat")
    t.set_constants(xo, yo, xi, yi, mag, ap_xy, psf=psf, psf_ax=psf_ax)
    t.set_dimensions(R_ap, L_det)
    t.save(tmat_path)
    calculate_tmat(t)
    
        
    print("Reading tmat")  
    with h5py.File(tmat_path, 'r') as f:
        T = f['tmat'][...]

    
    print(f"Running reconstruction")
    B, logL, chisq, background= gelfgat_poisson(data.flatten(), T, 50, h_friction=3)
    
    fig, ax = plt.subplots()
    ax.plot(logL)

    print(B.shape)
    print(logL.shape)
    
    ind = calculate_termination(B, logL)
    
    sol = np.reshape(B[ind, :], oshape)
  
    fig, ax = plt.subplots()
    ax.set_title(f"Termination iter {ind+1}")
    ax.set_aspect('equal')
    ax.pcolormesh(t.xo_scaled.to(u.um).value, 
                  t.yo_scaled.to(u.um).value, 
                  sol.T)
