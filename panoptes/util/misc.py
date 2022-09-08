import os
import datetime
import h5py


def identify_filetype(path):
    """
    Try to identify what type of file a file or a group within an hdf5 file 
    is based on  its contents and/or extension.
    
    All of the objects in this package save a 'class' attribute to the root
    directory of their group
    """
    
    # If the provided path is actually within an HDF5 file, identify the
    # type of the group instead of the file.
    if isinstance(path, h5py.Group):
        return identify_hdf_grp(path)
    

    _, ext = os.path.splitext(path)

    if ext.lower() in ['.h5', '.hdf5']:
        with h5py.File(path, 'r') as f:
            return identify_hdf_grp(f)
        
    elif ext.lower() in ['.cpsa']:
        return 'cr39'
    
    else:
        raise ValueError(f"Unrecognized file extension: {ext}")
    

        
def identify_hdf_grp(grp):
    if 'class' in grp.attrs.keys():
        return str(grp.attrs['class'][...])
    
    if 'PSL_per_px' in grp.keys():
        return 'OMEGA IP'
        


def timestamp():
    """
    Creates a timestamp string
    """
    now = datetime.now()
    return now.strftime("%y%m%d_%H%M%S")


def _compressed(xaxis, yaxis, data, chunk=25):
    """
    Returns a sparse sampling of the data
    """
    x_c = xaxis[::chunk]
    y_c = yaxis[::chunk]
    arr = data[::chunk, ::chunk]
    
    return x_c, y_c, arr


def find_file(dir, matchstr):
    # Find all the files in that directory
    files = [x[2] for x in os.walk(dir)][0]
    
    # FInd ones that match the reconstruction h5 pattern
    files = [x for x in files if all(s in x for s in matchstr)]
    
    if len(files) == 0:
        raise ValueError(f"No file found matching {matchstr} in {dir}")
    elif len(files) > 1:
        raise ValueError(f"Multiple files found matching {matchstr} in {dir}")
    else:
        file = files[0]
        
    return os.path.join(dir, file)

def find_folder(dir, matchstr):
        """
        Find subfolder
        """
        
        # Find all subdirectories that match the name
        dirs = [x[0] for x in os.walk(dir) if all(s in x[0] for s in matchstr)]
        
        if len(dirs) == 0:
            raise ValueError(f"No folder found matching {matchstr} in {dir}")
        elif len(dirs) > 1:
            raise ValueError(f"Multiple reconstruction folders found matching {matchstr} in {dir}")
        else:
            folder = dirs[0]
            
        return folder