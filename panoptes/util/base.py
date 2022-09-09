# -*- coding: utf-8 -*-
"""
Define the base saveable object class
"""
import os
import h5py
from abc import ABC

class BaseObject(ABC):
    """
    An object with properties that are savable to an HDF5 file or a group
    within an HDF5 file.
    
    The _save and _load methods need to be extended in child classes
    to define exactly what is being loaded and/or saved.
    
    """
    
    def __init__(self, **kwargs):
        # The path to the file
        self.path = None
        # The group within the h5 file where this is stored.
        # Defaults to the root group
        self.group = '/'
    
    
    def _save(self, grp):
        """
        Save this object to an h5 group
        
        
        Subclasses should call this method at the begining of their own
        _save method.
        """
        
        # Empty the group before saving new data there
        for key in grp.keys():
            del grp[key]
        
        grp.attrs['class'] = self.__class__.__name__
        
        
    def _load(self, grp):
        """
        Load an object from an h5 group
        """
        pass
    
    
    def save(self, path, group=None):
        """
        Save this object to an h5 file or group within an h5 file
        """ 
        
        if isinstance(path, h5py.File):
            self.path = path.filename
            self.group = path.name
            self._save(path)
            
        elif isinstance(path, h5py.Group):
            self.path = path.file.filename
            self.group = path.name
            self._save(path)
            
        # If a directory is provided, pass this directly to the _save method
        # to deal with
        elif os.path.isdir(path):
            self.path = path
            self.group = '/'
            self._save(path)
                
        # If a string, assume it is a file path and oepn the file to write
        elif isinstance(path, str):
            self.path = path
            self.group = '/'
            with h5py.File(self.path, 'a') as f:
                if group is not None:
                    grp = f[group]
                else:
                    grp = f['/']
                
                self._save(grp)
        
        else:
            raise ValueError(f"Invalid path of type {type(path)}: {path}")
            


    def load(self, path, group='/'):
        """
        Load this object from a file
        """
        
        if isinstance(path, h5py.File):
            self.path = path.filename
            self.group = path.name
            self._load(path)
            
        elif isinstance(path, h5py.Group):
            self.path = path.file.filename
            self.group = path.name
            self._load(path)
            
            
        # If a directory, then pass that directly to the _load method to deal
        # with.
        elif os.path.isdir(path):
            self.path = path
            self.group = group
            self._load(path)
            
            
        # If a string path, open the file
        elif isinstance(path, str):
            self.path = path
            self.group = group
        
            with h5py.File(self.path, 'r') as f:
                
                if group is not None:
                    grp = f[group]
                else:
                    grp = f['/']
                
                self._load(grp)
        
        
        
        else:
            raise ValueError(f"Invalid path of type {type(path)}: {path}")
            
  