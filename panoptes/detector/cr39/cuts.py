# -*- coding: utf-8 -*-
"""
Classes fod cuts and subsets of CR39 data

@author: pvheu
"""
import numpy as np
import h5py

from panoptes.util.base import BaseObject
from panoptes.util.misc import identify_filetype

import warnings

class Subset(BaseObject):
    """
    A subset of the track data. The subset is defined by a domain, a list
    of cuts, and a number of diameter slices (Dslices). 
    
    Paramters
    ---------
    
    grp : h5py.Group or string file path
        An h5py Group or file path to an h5py file from which to 
        load the subset
    
    
    Notes
    -----
    
    Domain
    ------
    The domain is the area in parameter space the subset encompasses. This
    could limit the subset to a region in space (e.g. x=[-5, 0]) or another
    parameter (e.g. D=[0,20]). The domain is represented by a cut, but it is 
    an inclusive rather than an exclusive cut.
    
    List of Cuts
    ------------
    The subset includes a list of cuts that are used to exclude tracks that
    would otherwise be included in the domain.
    
    DSlices
    ------
    Track diameter is proportional to particle energy, so slicing the subset
    into bins sorted by diameter sorts the tracks by diameter. Slices are 
    created by equally partitioining the tracks in the subset into some
    number of dslices. 
    
    """

    def __init__(self, *args, domain=None, ndslices=None):
        
        self.cuts = []
        
        # This list is not used internally but can be filled with data
        # by an external function using set_dslice_data() and then
        # accessed using the current_dslice_data() function.
        # If you want to save or load this data, it will need to be done
        # externally! 
        self.dslice_data = []
        
        super().__init__()
        
        
        
        if domain is not None:
            self.set_domain(domain)
        # If no domain is set, set it with an empty cut
        else:
            self.domain = Cut()
            
            
        # By default, set the number of dslices to be 1
        if ndslices is None:
            ndslices = 1
        self.set_ndslices(ndslices)
            
        # Index of currently selected slice
        # if None, include all slices
        self.set_current_dslice(0)
        
        # If an argument is provided, load the cut from that
        if len(args)==0:
            pass
        elif len(args)==1:
            self.path = args[0]
            filetype = identify_filetype(self.path)

            if filetype == self.__class__.__name__:
                self.load(self.path)
            
            else:
                raise ValueError(f"File {self.path} data type {filetype} "
                                 "is not valid for class "
                                 f"{self.__class__.__name__}")
            
        else:
            raise ValueError(f"Invalid number of arguments ({len(args)} for "
                             f"class {self.__class__.__name__}.")
                    

    def _save(self, grp):
        
        super()._save(grp)

        grp.attrs['ndslices'] = self.ndslices
        if self.current_dslice_i is None:
            grp.attrs['current_dslice_i'] = np.nan
        else:
            grp.attrs['current_dslice_i'] = self.current_dslice_i
            
        # Save the domain as a cut
        domain_grp = grp.create_group('domain')
        self.domain.save(domain_grp)
        
        cuts_grp = grp.create_group('cuts')
        for i, cut in enumerate(self.cuts):
            c_grp = cuts_grp.create_group(f"cut_{i}")
            cut.save(c_grp)
       
    def _load(self, grp):
        """
        See documentation for 'load'
        """
        super()._load(grp)
        
        self.set_ndslices(int(grp.attrs['ndslices']))
        self.current_dslice_i = grp.attrs['current_dslice_i']
            
        # Load the domain
        self.domain = Cut(grp['domain'])
        
        # Load the cuts
        cuts_grp = grp["cuts"]
        for key in cuts_grp:
            self.cuts.append( Cut(cuts_grp[key]) )
            
    def __str__(self):
        s = ''
        s += "Domain:" + str(self.domain) + '\n'
        s += "Current cuts:\n"
        if len(self.cuts) == 0:
            s+= "No cuts set yet\n"
        else:
            for i,cut in enumerate(self.cuts):
                s += f"Cut {i}: {str(cut)}\n"
        s += f"Num. dslices: {self.ndslices} "
        s += f"[Selected dslice index: {self.current_dslice_i}]\n"
                
        return s
               
    def set_domain(self, cut):
        """
        Sets the domain cut: an inclusive cut that will not be inverted
        """
        self.domain = cut
        
    def set_current_dslice(self, i):
        if i is None:
            self.current_dslice_i = 0
        elif i > self.ndslices-1:
            print(f"Cannot select the {i} dslice, there are only "
                             f"{self.ndslices} dslices.")
        else:
            self.current_dslice_i = i
        
    def set_ndslices(self, ndslices):
        """
        Sets the number of ndslices
        """
        if not isinstance(ndslices, int) or ndslices < 0:
            print("ndslices must be an integer > 0, but the provided value"
                  f"was {ndslices}")
            
        if not all(x==None for x in self.dslice_data):
            
            raise ValueError("Changing the number of ndslices will delete "
                             "any existing data stored in the dslice_data "
                             "list! To do this, call the method "
                             "`reset_dslice_data` before attempting to "
                             "change the number of ndslicse.")
            
        else:
            self.ndslices = int(ndslices)
            
            self.reset_dslice_data()
            
    def reset_dslice_data(self):
        """
        Reset the dslice data array to be empty. 
        
        You are required to do this manually before changing the number of
        dslices, to confirm that you really want to do this!
        """
        
        # Create a new, empty list of dslice_data
        self.dslice_data = [None] * self.ndslices
            
    def set_dslice_data(self, *args):
        if len(args)==1:
            data = args[0]
            i = self.current_dslice_i
        elif len(args)==2:
            data = args[0]
            i=args[1]
        else:
            raise ValueError(f"Invalid number of arguments: {len(args)}")
        self.dslice_data[i] = data
        
    @property
    def current_dslice_data(self):
        return self.dslice_data[self.current_dslice_i]
        
            
    def add_cut(self, c):
        self.cuts.append(c)
           
    def remove_cut(self, i):
        if i > len(self.cuts)-1:
            print(f"Cannot remove the {i} cut, there are only "
                             f"{len(self.cuts)} cuts.")
        else:
            self.cuts.pop(i)
        
    def replace_cut(self, i, c):
        if i > len(self.cuts)-1:
            print(f"Cannot replace the {i} cut, there are only "
                             f"{len(self.cuts)} cuts.")
        else:
            self.cuts[i] = c
            
            

class Cut(BaseObject):
    """
    A cut is series of upper and lower bounds on tracks that should be
    excluded. 
    
    Parameters
    ----------
    
    grp : h5py.Group or string file path
        An h5py Group or file path to an h5py file from which to 
        load the cut
    
    
    """
    
    
    defaults = {'xmin':-1e6, 'xmax':1e6, 'ymin':-1e6, 'ymax':1e6,
                'dmin':0, 'dmax':1e6, 'cmin':0, 'cmax':1e6, 
                'emin':0, 'emax':1e6}
    
    indices = {'xmin':0, 'xmax':0, 'ymin':1, 'ymax':1,
                'dmin':2, 'dmax':2, 'cmin':3, 'cmax':3, 
                'emin':5, 'emax':5}

    
    def __init__(self,  *args,
                        xmin : float = None, xmax : float = None,
                        ymin : float = None, ymax : float = None,
                        dmin : float = None, dmax : float = None,
                        cmin : float = None, cmax : float = None,
                        emin : float = None, emax : float = None):
        
        super().__init__()
        
        self.dict = {'xmin':xmin, 'xmax':xmax,
                     'ymin':ymin, 'ymax':ymax,
                     'dmin':dmin, 'dmax':dmax,
                     'cmin':cmin, 'cmax':cmax,
                     'emin':emin, 'emax':emax}
        
        
        # If an argument is provided, load the cut from that
        if len(args)==0:
            pass
        elif len(args)==1:
            self.path = args[0]
            filetype = identify_filetype(self.path)

            if filetype == self.__class__.__name__:
                self.load(self.path)
            
            else:
                raise ValueError(f"File {self.path} data type {filetype} "
                                 "is not valid for class "
                                 f"{self.__class__.__name__}")
            
        else:
            raise ValueError(f"Invalid number of arguments ({len(args)} for "
                             f"class {self.__class__.__name__}.")
        
        
    def _save(self, grp):
        """
        See docstring for "save"
        """
        super()._save(grp)
        
        for key, val in self.dict.items():
            if val is None:
                grp.attrs[key] = np.nan
            else:
                grp.attrs[key] = val
            
      
    def _load(self, grp):
        super()._load(grp)
        
        for key in self.dict.keys():
            val = grp.attrs[key]
            # Interpret NaN as None
            if np.isnan(val):
                self.dict[key] = None
            # Otherwise directly read in the value
            else:
                self.dict[key]=val
            
        

    def __getattr__(self, key):
        
        if key in self.dict.keys():
            if self.dict[key] is None:
                return self.defaults[key]
            else:
                return self.dict[key]
        else:
            raise ValueError(f"Unknown attribute for Cut: {key}")
            
        
    # These range properties are used to set the range for plotting
    @property
    def xrange(self):
        return [self.dict['xmin'], self.dict['xmax']]
    @property
    def yrange(self):
        return [self.dict['ymin'], self.dict['ymax']]
    @property
    def drange(self):
        return [self.dict['dmin'], self.dict['dmax']]
    @property
    def crange(self):
        return [self.dict['cmin'], self.dict['cmax']]
    @property
    def erange(self):
        return [self.dict['emin'], self.dict['emax']]
                
            
    def __str__(self):
        s = [f"{key}:{val}" for key, val in self.dict.items() if val is not None ]
        s = ', '.join(s)
        if s == '':
            return "[Empty cut]"
        else:
            return s
    
    
    def test (self, trackdata):
        """
        Given tracks, return a boolean array representing which tracks
        fall within this cut
        """
        ntracks, _ = trackdata.shape
        keep = np.ones(ntracks).astype('bool')

        for key in self.dict.keys():
            if self.dict[key] is not None:
                i = self.indices[key]
                if 'min' in key:
                    keep *= np.greater(trackdata[:, i], getattr(self, key))
                else:
                    keep *= np.less(trackdata[:, i], getattr(self, key))      
        
        # Return a 1 for every track that is in the cut
        return keep.astype(bool)
    
    
    
if __name__ == '__main__':
    import os
    from panoptes.detector.cr39.cuts import Cut
    domain = Cut(xmin=-5, xmax=0)
    c1 = Cut(dmax=10)
    c2 = Cut(cmax=40)
    s = Subset(domain=domain)
    s.set_ndslices(5)
    s.set_current_dslice(2)
    s.add_cut(c1)
    s.add_cut(c2)
    
    
    path = os.path.join(os.getcwd(), 'testsubset.h5')
    print(path)
    s.save(path)
    
    s2 = Subset(path)
    print(s2.domain.dict)
    print(s2.cuts[0].dict)
    print(s2.cuts[1].dict)