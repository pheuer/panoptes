# -*- coding: utf-8 -*-
"""
@author: Peter Heuer

Object representing a CR39 dataset
"""
import os
import numpy as np

from collections import namedtuple

import matplotlib.pyplot as plt

from panoptes.pinholes import PinholeArray, _adjust_xy
from panoptes.util.misc import  find_file


from cr39py.cr39 import CR39, Cut


class CR39Pinholes(CR39):
    
    def __init__(self, *args, pinhole_array, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        
        self.pinholes = PinholeArray(pinhole_array)
        
        
  
    def fit_pinholes(self, *args, **kwargs):
        
        xaxis, yaxis, data = self.frames(axes=['X', 'Y'], trim=True)
        
        self.pinholes.fit(xaxis, yaxis, data,
                          *args, **kwargs)
    
        
    def square_off(self):
        a = obj.pinholes.adjustment
        
        adjustment = {'dx':-a['dx'], 'dy':-a['dy'], 'rot':-a['rot'],
                      'mag_s':1/a['mag_s']}
        

    def plot(self, *args, **kwargs):
        """
        See keywords in cr39.CR39.plot()
        """
        
        fig, ax = super().plot(*args, **kwargs)
        
        
        if self.pinholes is not None:
            ax.scatter(self.pinholes.xy[self.pinholes.use,0], 
                       self.pinholes.xy[self.pinholes.use,1], color='green')
            
            ax.scatter(self.pinholes.xy[~self.pinholes.use,0], 
                       self.pinholes.xy[~self.pinholes.use,1], color='red')
        
        
        if 'figax' not in kwargs.keys():
            plt.show()
        
        
  
        
  
        
if __name__ == '__main__':
    data_dir = os.path.join("C:\\","Users","pvheu","Desktop","data_dir")
    obj = CR39Pinholes(103955, data_dir=data_dir, verbose=True, pinhole_array='D-PF-C-055_A')
    obj.hreflect()
    obj.add_cut(Cut(xmin=0.25, dmax=10))
    obj.apply_cuts()
    obj.cutplot()
    obj.fit_pinholes(rough_adjust={'dx':0.3, 'dy':-0.2, 'mag_s':35.5, 'rot':-17},
                     auto_select_apertures=True,)
    
    
    
        
        
            
