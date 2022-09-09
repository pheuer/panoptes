# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 09:33:39 2022

@author: pheu


"""

import numpy as np
import os, h5py


import astropy.units as u


import matplotlib.pyplot as plt



# Make plots appear in new windows
from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'qt')

if get_ipython() is not None:
    get_ipython().run_line_magic('matplotlib', 'inline')


from panoptes.detector.cr39.scan import Scan
from panoptes.detector.cr39.cuts import Cut, Subset
from panoptes.util.misc import identify_filetype
from panoptes.detector.detector import Data2D

from panoptes.pinholes import PinholeArray
from panoptes.diagnostic.diagnostic import PinholeImager, GelfgatReconstruction
from panoptes.reconstruction.tmat import TransferMatrix
from panoptes.reconstruction.gelfgat import gelfgat_poisson, GelfgatResult


            

class KoDI(PinholeImager, GelfgatReconstruction):
    
    def __init__(self, *args,  plots=True, mag=None,
                 run_cli=None, **kwargs):
        """
        arg : path or int
        
            filepath directly to the data file or to a saved copy of this
            object
        """
        super().__init__(**kwargs)

        if len(args) == 0:
            pass
        
        elif len(args)==1:
            path = args[0]
            filetype = identify_filetype(path)
    
            if filetype == self.__class__.__name__:
                self.load(path)
                if run_cli is None:
                    run_cli = False
                
            elif filetype == 'cpsa':
                self.data = Scan(path)
                if run_cli is None:
                    run_cli = True
        else:
            raise ValueError(f"Invalid number of arguments ({len(args)} for "
                             f"class {self.__class__.__name__}.")
                    
        if run_cli:
            self.scan.cli()
    
    def _save(self, grp):
        super()._save(grp)
        
    def _load(self, grp):
        super()._load(grp)

            
            

            
if __name__ == '__main__':
       
    data_dir = os.path.join("C:\\","Users","pvheu","Desktop","data_dir")
    #data_dir = os.path.join('//expdiv','kodi','ShotData')
    #data_dir = os.path.join('\\\profiles','Users$','pheu','Desktop','data_dir')
    data_dir = os.path.join("C:\\","Users","pheu","Data","data_dir")
    
    data_path = os.path.join(data_dir, '103955', '103955_TIM5_PR3148_2h_s7_20x.cpsa')
    save_path = os.path.join(data_dir, '103955', 'kodi_test.h5')
    
    
    if not os.path.isfile(save_path):
        obj = KoDI(data_path, pinhole_array = 'D-PF-C-055_A', run_cli=False)
        
        obj.data.current_subset.set_domain(Cut(xmax=-0.25))
        obj.data.add_cut(Cut(cmin=35))
        obj.data.add_cut(Cut(dmin=12))
        obj.data.cutplot()
        
        # Create a new subset and select it
        obj.data.add_subset()
        obj.data.select_subset(1)
        # Create cuts on the new subset
        obj.data.add_cut(Cut(cmin=25))
        obj.data.add_cut(Cut(dmin=15))
        obj.data.cutplot()
        
        obj.save(save_path)
    else:
        obj = KoDI(save_path)
       
    
    obj.fit_pinholes()
        
        