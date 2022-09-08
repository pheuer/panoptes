# -*- coding: utf-8 -*-
"""
@author: Peter

INSTRUCTIONS


* For downloading and running h5toh4convert *
1) Download and install the lastest version of the conversion utility from
   the HDF group: https://portal.hdfgroup.org/display/support/h4h5tools%202.2.5#files
   
2) Add the \bin file to your path. This should be something like 
   C:\Program Files\HDF_Group\H4TOH5\2.2.2\bin

3) Test by running "h4toh5convert -h" in the command line


"""


import os, subprocess, re
import warnings

from panoptes.config import h4toh5convert_path

def get_hdf5(directory, regex='(.*?)'):
    """
    ind an hdf file in a given directory that matches a regex. If the file
    is an HDF4 file, convert to HDF5
    """
    regex += ".[hdf, hdf4, h4, hdf5, h5]"
    
    # Find all the files matching the regex
    regex = re.compile(regex)
    matches = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if regex.match(file):
                path = os.path.join(root, file)
                matches.append(path)
                
    # If no matches are found at all, raise an exception        
    if len(matches) == 0:
        raise ValueError(f"No match found matching {regex} in {directory}")
        
    # Determine if any are HDF5 files
    hdf5 = ['.hdf5', '.h5']
    h5_matches = [m for m in matches if os.path.splitext(m)[1] in hdf5 ]
    
    print(h5_matches)
    
    # If multiple h5 matches are found, warn the user then return only the first one
    if len(h5_matches) > 0:
        if len(h5_matches) > 1:
            warnings.warn(f"Multiple h5 matches found. Only returning the first. {matches}", UserWarning)
        return h5_matches[0]
    
    # # If no h5 file is found, convert HDF4 to HDF5 using hdf4tohdf5 convert
    elif len(matches) > 0: 
        if len(matches) > 1:
            warnings.warn(f"Multiple h4 matches found. Only returning the first. {matches}", UserWarning)
        return ensure_hdf5(matches[0])
    
        
    else: 
        raise ValueError("No h4 or h5 files found.")

def ensure_hdf5(file):
    """
    Convert a provided file to hdf5 if necessary. 
    
    If the provided file path is an hdf5 file, return the filepath. 
    
    If the provided file path is an hdf4 file, use h4toh5convert to convert
    the file and then return the path to the converted file.
    """
    
    
    hdf5 = ['.hdf5', '.h5']
    
    file_dir = os.path.dirname(file)
    name, ext = os.path.splitext(file)
    src = os.path.join(file_dir, name + ext)
    path = os.path.join(file_dir, name + ".h5")  
    
    # If path file already exists, skip the conversion
    if ext in hdf5:
        print(f"File {file} is already an hdf5 file.")
        return path
    elif os.path.exists(path):
        print(f"Matching hdf5 file for {file} already exists.")
        return path
    
    if h4toh5convert_path is None:
        raise ValueError("Must set h4toh5convert_path in the config.py file")
    
    # TODO: Support running on linux or OSX...
    # Quotes necessary in case of spaces etc. in path
    cmd = f"\"{h4toh5convert_path}\" \"{src}\" \"{path}\""

    # Convert the file to h5
    # cwd needs to be set if the path to this module is on a UNC path, because those
    # will raise an error. C should be a drive on most PCs, but this is a hack...
    print(f"Converting file to hdf5: {file}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, cwd='C://')
    
    # TODO: better handling of errors..
    if not os.path.exists(path):
        print(process.communicate()[0])

    return path


