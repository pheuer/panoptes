# -*- coding: utf-8 -*-
"""
@author: Peter Heuer

Object representing a CR39 dataset

This is a thin wrapper on the 'CR39' class from cr39py
"""
import os
import numpy as np
import astropy.units as u

from panoptes.util.base import BaseObject

from panoptes.util.misc import identify_filetype

from cr39py.cr39 import CR39 as _CR39
from cr39py.cr39 import Cut

class CR39(BaseObject):
    
    def __init__(self, filepath, *args, **kwargs):
        """
        
        Additional arguments and keyword arguments are passed on to read_data
        """
        
        # Frst initialize the empty parent class
        super().__init__()
        
        self.cr39 = None
        
        self.path = filepath
        
        # Check if the file is Data object or some other data file based
        # on the keywords within it
 
        filetype = identify_filetype(self.path)
            
        if filetype == self.__class__.__name__:
            self.load(self.path)
            
        elif filetype == 'cr39':
           self.read_data(*args, **kwargs)

            
        else:
            raise ValueError(f"File type {filetype} is not supported by "
                             f"class {self.__class__.__name__}")

    def _save(self, grp):
        
        super()._save(grp)
        if self.cr39 is not None:
            self.cr39.save(grp)
            

    def _load(self, grp):
        super()._load(grp)
        
        if 'subsets' in grp.keys():
            self.cr39 = _CR39(grp)
            
    def read_data(self, cli=False):
        """
        Reads data from a cpsa file
        """
        self.cr39 = _CR39(self.path, verbose=True)
        
        if cli:
            self.cr39.cli()

  
        
  
if __name__ == "__main__":
    data_dir = os.path.join("C:\\","Users","pvheu","Desktop","data_dir")
    data_dir = os.path.join('C:\\','Users','pheu','Data','data_dir')
    
    data_path = os.path.join(data_dir, '103955', '103955_TIM5_PR3148_2h_s7_20x.cpsa' )
    #data_path = os.path.join(data_dir, '103955', 'OM220830_105521_PR2771.cpsa' )
    save_path = os.path.join(data_dir, '103955', 'cr39_test.h5' )
    
    #obj = CR39(data_path)
    
    obj = CR39(save_path)
    obj.cr39.add_cut(Cut(cmin=30))
    print(obj.cr39.current_subset)
    obj.cr39.add_subset()
    obj.cr39.select_subset(1)
    obj.cr39.add_cut(Cut(dmin=20))
    print(obj.cr39.current_subset)
    
    
    
    #obj = CR39(save_path)
    
        
        
            
