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
            
            
            
    def process(self, **kwargs):
        
        # Make sure all cuts are applied
        self.scan.apply_cuts()
        
        # Create a PenumbralImageGelfgat object using the data from
        # the current subset and 
        
        
        # TODO: intelligently choose the bin size if not specified by 
        # repeatedly making frames and looking at the median number in
        # each bin?
        
        # Note: we want a significantly higher resolution here than the frame
        # size, since we have combined images. Roughly we want N_ap*frame size
        
        
        # The way to handle that is to make new frames right before
        # doing the stacking.
        
        hax = self.scan.axes['X']
        hax = np.linspace(np.min(hax), np.max(hax), num=2*hax.size)
        vax = self.scan.axes['Y']
        vax = np.linspace(np.min(vax), np.max(vax), num=2*vax.size)
        
        xaxis, yaxis, data = self.scan.frames(hax=hax, vax=vax)

        obj = PenumbralImageGelfgat(xaxis*u.cm, yaxis*u.cm, data,
                                           pinhole_array=self.pinhole_array)
        
    
        obj.fit_pinholes(**kwargs)

        
        hax = np.linspace(np.min(hax), np.max(hax), num=8*hax.size)
        vax = np.linspace(np.min(vax), np.max(vax), num=8*vax.size)
        xaxis, yaxis, data = self.scan.frames(hax=hax, vax=vax)
        obj.xaxis = xaxis*u.cm
        obj.yaxis = yaxis*u.cm
        obj.data = data
        
        
        obj.stack_pinholes()
        # Store this object in the subset's dslice list.
        self.scan.current_subset.dslice_data[self.scan.current_subset.current_dslice_i] = obj
        
     
    def calculate_tmat(self, tmat_path, mag=None):
        obj = self.scan.current_subset.current_dslice_data
        
        obj.make_tmat(tmat_path, R_ap=150*u.um, L_det=350*u.cm, oshape=(31, 31),
                      mag = mag)
        

    def reconstruct(self, tmat_path):
        
        obj = self.scan.current_subset.current_dslice_data
        obj.reconstruct(tmat_path)
        
        obj.reconstruction.iter_plot()
        obj.plot_reconstruction()
        



    def _save(self, grp):
        
        super()._save(grp)
        
        if self.pinhole_array is not None:
            grp['pinhole_array'] = self.pinhole_array
        
        
        scangrp = grp.create_group('scan')
        
        # First save the CR39 scan object 
        self.scan._save(scangrp)
        
        
        # Next, save every individual histogram as a PenumbraImageGelfgat
        # object within the folder
       
        for i, subset in enumerate(self.scan.subsets):
            self.scan.select_subset(i)
            subset_grp = scangrp[f"subset_{i}"]
            
            for j in range(self.scan.current_subset.ndslices):
                dslice_grp = subset_grp.create_group(f"dslice_{j}")

                if self.scan.current_subset.dslice_data[j] != None:
                    self.scan.current_subset.dslice_data[j].save(dslice_grp)
                    
        
    def _load(self, grp):
        super()._load(grp)
        
        if 'pinhole_array' in grp.keys():
            # Note that h5py reads in strings as byte stirngs, so we need to convert
            # also, for some reason [()] seems to be the preferred indexing
            # to load this scalar?
            self.pinhole_array = grp['pinhole_array'][()].decode('utf-8')

        scangrp = grp['scan']
        
        # First load the CR39 scan object
        self.scan = Scan(scangrp)
        
        for i, subset in enumerate(self.scan.subsets):
            self.scan.select_subset(i)
            subset_grp = scangrp[f"subset_{i}"]
            
            for j in range(self.scan.current_subset.ndslices):
                dslice_grp = subset_grp[f"dslice_{j}"]
                
                # If the group isn't empty, try to load it as a 
                # PenumbralImageGelfgat object
                if len(list(dslice_grp.keys())) != 0:
                    
                    obj = PenumbralImageGelfgat(dslice_grp)
                    self.scan.current_subset.dslice_data[j] = obj
        

            
            

            
if __name__ == '__main__':
    
    data_dir = os.path.join("C:\\","Users","pvheu","Desktop","data_dir")
    #data_dir = os.path.join('//expdiv','kodi','ShotData')
    #data_dir = os.path.join('\\\profiles','Users$','pheu','Desktop','data_dir')
    #data_dir = os.path.join("C:\\","Users","pheu","Data","data_dir")
    
    data_path = os.path.join(data_dir, '103955', '103955_TIM5_PR3148_2h_s7_20x.cpsa')
    save_path = os.path.join(data_dir, '103955', 'kodi_test.h5')
    
    tmat_path = os.path.join(data_dir, '103955', 'tmat_kodi.h5')
    
    
    if not os.path.isfile(save_path):
        obj = KoDI(data_path, pinhole_array = 'D-PF-C-055_A', run_cli=False)
        
        obj.scan.current_subset.set_domain(Cut(xmin=-4.7, xmax=-0.25))
        obj.scan.add_cut(Cut(cmin=35))
        obj.scan.add_cut(Cut(dmin=12))
        obj.scan.cutplot()
        
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
        
        
    obj.scan.select_subset(0)
        
     
    #if obj.scan.current_subset.current_dslice_data is None:           
    obj.process( auto_select_apertures=True, 
                rough_adjust={'dx':-0.3, 'dy':-0.1, 'mag_s':36, 'rot':18},)
    

    #obj.save(save_path)

        
    obj.scan.current_subset.current_dslice_data.plot_stack()
    #if obj.scan.current_subset.current_dslice_data.tmat is None:
    obj.calculate_tmat(tmat_path)
        
        
    #if obj.scan.current_subset.current_dslice_data.reconstruction is None:
    obj.reconstruct(tmat_path)
    
    
    
    obj.save(save_path)
    
    x = obj.scan.current_subset.current_dslice_data
    
    x.reconstruction.iter_plot()
    x.plot_reconstruction()
    

    
