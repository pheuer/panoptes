# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 11:53:07 2022

@author: pheu
"""

import os
import h5py

import numpy as np
import astropy.units as u

from panoptes.detector.detector import Data2D



def test_data2d_withunits(path):
    
    if os.path.isfile(path):
        os.remove(path)
    
    xaxis = np.arange(10)*u.cm
    yaxis = xaxis
    data = np.random.random((10,10)) * u.T
    
    obj = Data2D(xaxis, yaxis, data)
    
    obj.save(path)
    
    obj2 = Data2D(path)
    
    assert obj2.xaxis.unit == xaxis.unit
    assert np.allclose(obj2.data.value, data.value)
    
    
def test_data2d_withoutunits(path):
    
    if os.path.isfile(path):
        os.remove(path)
    
    xaxis = np.arange(10)
    yaxis = xaxis
    data = np.random.random((10,10))
    
    obj = Data2D(xaxis, yaxis, data)
    
    obj.save(path)
    
    obj2 = Data2D(path)
    

    assert np.allclose(obj2.data, data)


if __name__ == '__main__':
    path = os.path.join(os.getcwd(), 'testfile.h5')
    
    
        
    test_data2d_withunits(path)
    test_data2d_withoutunits(path)
    