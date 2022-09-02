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
    data_dir = os.path.join('C:\\','Users','pheu','Data','data_dir', "103955")

    datafile = os.path.join(data_dir, 'xray-stack.h5')
    tmat_path = os.path.join(data_dir, 'tmat252.h5')

    psf = np.concatenate((np.ones(50), np.zeros(50)))
    psf_ax = (np.linspace(0, 300, num=100)*u.um).to(u.cm).value
    
    
    
    


    with h5py.File(datafile, 'r') as f:
        xi = f['stack']['xaxis'][...] * u.cm
        yi = f['stack']['yaxis'][...] * u.cm
        data = f['stack']['data'][...]

        pinhole_radius = f['pinholes']['pinhole_info']['diameter'][...] * u.um
        mag = f['pinholes']['mag_r'][...]
        
        

    # Thin data for testing
    data = data[::4, ::4]
    xi = xi[::4]
    yi = yi[::4]
    
    nxi, nyi = xi.size, yi.size
        
    xo = (np.linspace(-100, 100, num=25) * u.um).to(u.cm)
    yo = (np.linspace(-100, 100, num=25) * u.um).to(u.cm)
    nxo, nyo = xo.size, yo.size

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.pcolormesh(xi.value, yi.value, data.T)
    plt.show()
    
    if not os.path.isfile(tmat_path):
        t = TransferMatrix(tmat_path)
    
        print("Calculating tmat")
        t.set_constants(xo, yo, xi, yi, mag, np.zeros([1,2])*u.cm, psf=psf, psf_ax=psf_ax)
        t.save(tmat_path)
        t.calc_tmat()
        
        
    print("Reading tmat")  
    with h5py.File(tmat_path, 'r') as f:
        T = f['tmat'][...]
        
        
    print(T.shape)
    arr = np.reshape(np.sum(T, axis=0), (nxi, nyi))
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.pcolormesh(xi.value, yi.value, arr.T)
 
        
    
    # TODO: transpose the transfer matrix order in the tmat file
    T = T.T
    B, logL, chisq, DOFs = gelfgat_poisson(data.flatten(), T, 50, h_friction=3)
    
    
    print(B.shape)
    print(logL.shape)
    
    arr = np.reshape(B[40, :], (nxo, nyo))
    
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.pcolormesh(xo.value, yo.value, arr.T)
    plt.show()
    
    