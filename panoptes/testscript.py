# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 21:14:48 2022

@author: pvheu
"""

import os, h5py

from panoptes.tmat.tmat import TransferMatrix


data_dir = os.path.join("C:\\","Users","pvheu","Desktop","data_dir", "103955")
#data_dir = os.path.join('C:\\','Users','pheu','Data','data_dir', "103955")

datafile = os.path.join(data_dir, 'xray-stack.h5')


with h5py.File(datafile, 'r') as f:
    x = f['stack']['xaxis'][...]
    y = f['stack']['yaxis'][...]
    data = f['stack']['data'][...]