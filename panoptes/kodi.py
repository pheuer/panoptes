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
from panoptes.detector.detector import Data2D, PenumbralImageGelfgat

from panoptes.diagnostic.diagnostic import Diagnostic



            

class KoDI(Diagnostic):
    
    def __init__(self, *args,  plots=True, mag=None,
                 run_cli=None, pinhole_array = None, **kwargs):
        """
        arg : path or int
        
            filepath directly to the data file or to a saved copy of this
            object
        """
        
        # CR39 scan object
        self.scan = None
        self.pinhole_array = pinhole_array
        
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
                self.scan = Scan(path)
                if run_cli is None:
                    run_cli = True
        else:
            raise ValueError(f"Invalid number of arguments ({len(args)} for "
                             f"class {self.__class__.__name__}.")
                    
        if run_cli:
            self.scan.cli()
            
            
            
    def process(self):
        
        # Create a histogram, save as a Data2D object within
        # the dslice group
        xaxis, yaxis, data = self.scan.frames()
        dslice_data = PenumbralImageGelfgat(xaxis, yaxis, data,
                                           pinhole_array=self.pinhole_array)
        
        self.scan.current_subset.set_dslice_data(dslice_data)
            

    def _save(self, grp):
        
        super()._save(grp)
        
        scangrp = grp.create_group('scan')
        
        # First save the CR39 scan object 
        self.scan._save(scangrp)
        
        
        # Next, save every individual histogram as a PenumbraImageGelfgat
        # object within the folder
       
        print('saving')
        for i, subset in enumerate(self.scan.subsets):
            self.scan.select_subset(i)
            subset_grp = scangrp[f"subset_{i}"]
            
            
            print(f"ndslies: {self.scan.current_subset.ndslices}")
            print(f"dslice_data: {self.scan.current_subset.dslice_data}")
            for j in range(self.scan.current_subset.ndslices):
                dslice_grp = subset_grp.create_group(f"dslice_{j}")
            
                print(j)
                if self.scan.current_subset.dslice_data[j] != None:
                    self.scan.current_subset.dslice_data[j].save(dslice_grp)
                    
            
        
        
    def _load(self, grp):
        super()._load(grp)
        
        scangrp = grp['scan']
        
        # First load the CR39 scan object
        self.scan = Scan(scangrp)
        
        print('Loading')
        for i, subset in enumerate(self.scan.subsets):
            self.scan.select_subset(i)
            subset_grp = scangrp[f"subset_{i}"]
            
            print(f"ndslies: {self.scan.current_subset.ndslices}")
            print(f"dslice_data: {self.scan.current_subset.dslice_data}")
            for j in range(self.scan.current_subset.ndslices):
                dslice_grp = subset_grp[f"dslice_{j}"]
                
                print(j)
                # If the group isn't empty, try to load it as a 
                # PenumbralImageGelfgat object
                if len(list(dslice_grp.keys())) != 0:
                    
                    obj = PenumbralImageGelfgat(dslice_grp)
                    self.scan.current_subset.dslice_data[j] = obj
        

            
            

            
if __name__ == '__main__':
    
    data_dir = os.path.join("C:\\","Users","pvheu","Desktop","data_dir")
    #data_dir = os.path.join('//expdiv','kodi','ShotData')
    #data_dir = os.path.join('\\\profiles','Users$','pheu','Desktop','data_dir')
    data_dir = os.path.join("C:\\","Users","pheu","Data","data_dir")
    
    data_path = os.path.join(data_dir, '103955', '103955_TIM5_PR3148_2h_s7_20x.cpsa')
    save_path = os.path.join(data_dir, '103955', 'kodi_test.h5')
    
    
    if not os.path.isfile(save_path):
        obj = KoDI(data_path, pinhole_array = 'D-PF-C-055_A', run_cli=False)
        
        obj.scan.current_subset.set_domain(Cut(xmax=-0.25))
        obj.scan.add_cut(Cut(cmin=35))
        obj.scan.add_cut(Cut(dmin=12))
        obj.scan.cutplot()
        
        obj.process()
        
        # Create a new subset and select it
        obj.scan.add_subset()
        obj.scan.select_subset(1)
        # Create cuts on the new subset
        obj.scan.add_cut(Cut(cmin=25))
        obj.scan.add_cut(Cut(dmin=15))
        obj.scan.cutplot()
        obj.scan.current_subset.set_ndslices(4)
        
        obj.save(save_path)
    else:
        obj = KoDI(save_path)
