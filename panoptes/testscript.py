# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 21:14:48 2022

@author: pvheu
"""
import numpy as np
import astropy.units as u
import os, h5py

from panoptes.tmat.tmat import TransferMatrix
from panoptes.gelfgat import gelfgat_poisson

import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    #__spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    
    data_dir = os.path.join("C:\\","Users","pvheu","Desktop","data_dir", "103955")
    #data_dir = os.path.join('C:\\','Users','pheu','Data','data_dir', "103955")

    datafile = os.path.join(data_dir, 'xray-stack.h5')
    tmat_path = os.path.join(data_dir, 'tmat.h5')



    
    
    
    


    with h5py.File(datafile, 'r') as f:
        xi = f['stack']['xaxis'][...] * u.cm
        yi = f['stack']['yaxis'][...] * u.cm
        data = f['stack']['data'][...]

        R_ap = 0.5*f['pinholes']['pinhole_info']['diameter'][...] * u.um
        mag = f['pinholes']['mag_r'][...]
        
      
    print(f"R_ap: {R_ap}")
    print(f"Mag: {mag}")
    psf = np.concatenate((np.ones(50), np.zeros(50)))
    psf_ax = np.linspace(0, 2*R_ap, num=100)/R_ap
    
    
    xlim = 80
    xo = np.linspace(-xlim, xlim, num=50) * u.um / R_ap
    yo = np.linspace(-xlim, xlim, num=50) * u.um / R_ap
    
    xi = xi/R_ap/mag
    yi = yi/R_ap/mag
    ap_xy = np.array([[0,0],])*u.cm / R_ap

    # Thin data for testing
    data = data[::4, ::4]
    xi = xi[::4]
    yi = yi[::4]
    
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
    t.save(tmat_path)
    t.calc_tmat()
    
        
    print("Reading tmat")  
    with h5py.File(tmat_path, 'r') as f:
        T = f['tmat'][...]
        
        
    print(T.shape)
    arr = np.reshape(T[200, :], (nxi, nyi))
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.pcolormesh((xi*R_ap*mag).to(u.cm).value, (yi*R_ap*mag).to(u.cm).value, arr.T)
 
        
    
    # TODO: transpose the transfer matrix order in the tmat file
    T = T.T
    B, logL, chisq, DOFs = gelfgat_poisson(data.flatten(), T, 50, h_friction=3)
    
    fig, ax = plt.subplots()
    ax.plot(logL)
    
    
    print(B.shape)
    print(logL.shape)
    
    
    for i in [5, 10, 15, 20, 25, 30]:
    
        arr = np.reshape(B[i, :], (nxo, nyo))
        
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.pcolormesh((xo*R_ap).to(u.um).value, (yo*R_ap).to(u.um).value, arr.T)
        plt.show()
        
    