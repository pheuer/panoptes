# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 21:14:48 2022

@author: pvheu
"""
import numpy as np
import astropy.units as u
import os, h5py

from panoptes.tmat.tmat import TransferMatrix




if __name__ == '__main__':
    
    #__spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    
    data_dir = os.path.join("C:\\","Users","pvheu","Desktop","data_dir", "103955")
    data_dir = os.path.join('C:\\','Users','pheu','Data','data_dir', "103955")

    datafile = os.path.join(data_dir, 'xray-stack.h5')
    tmat_path = os.path.join(data_dir, 'tmat.h5')

    psf = np.concatenate((np.ones(50), np.zeros(50)))
    psf_ax = np.linspace(0, 2, num=100)



    with h5py.File(datafile, 'r') as f:
        xi = f['stack']['xaxis'][...] * u.cm
        yi = f['stack']['yaxis'][...] * u.cm
        data = f['stack']['data'][...]

        pinhole_radius = f['pinholes']['pinhole_info']['diameter'][...] * u.um
        mag = f['pinholes']['mag_r'][...]
        
    xo = np.linspace(-100, 100, num=100) * u.um
    yo = np.linspace(-100, 100, num=100) * u.um


    t = TransferMatrix(tmat_path)


    print("Calculating tmat")
    t.set_constants(xo, yo, xi, yi, mag, np.zeros([1,2])*u.cm, psf=psf, psf_ax=psf_ax)
    t.save(tmat_path)
    t.calc_tmat()