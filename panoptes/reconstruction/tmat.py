
import os
import astropy.units as u
import h5py
import numpy as np


from multiprocessing import Pool
import tqdm
    
from panoptes.util.base import BaseObject

from cached_property import cached_property

import warnings



class TransferMatrix(BaseObject): 

    def __init__(self, *args, **kwargs):
        """
        Represents a transfer matrix between the object and image plane
        of a pinhole or penumbral imager.
        
        
        Paramters (1)
        -------------
        
        Path to a transfer matrix saved as an h5 file
        
        
        Paramters (8)
        -------------
        
        The parameters (xo, yo, xi, yi, mag, ap_xy, psf, psf_ax), as defined
        below. 
        
        
        
        Paramters (10)
        -------------
        The parameters (xo, yo, xi, yi, mag, ap_xy, psf, psf_ax, R_ap, L_det),
        as defined below. 
        
        
        

        There are three levels of parameters that describe a transfer
        matrix object (all of which are available as attributes to this class
                       if they are defined.)
        
    
        Constants
        ---------
        These (dimensionless)
        
        xo, yo : 1D np.ndarray
            Axes for each dimension of the transfer matrix in the object plane.
            Normalized to R_ap
             
        xi, yi : 1D np.ndarray
            Axes for each dimension of the transfer matrix in the image plane.
            Normalized to R_ap*mag
            
            
        mag : float
            Radiography magnification: 1 + L2/L1. 

            
        ap_xy : np.ndarray (N_ap, 2)
            The locations of the aperture centers in the pinhole plane.
            Normalize to R_ap.
            
        psf : np.ndarray 
            The point-spread function for a single aperture. 
            
        psf_ax : np.ndarray 
            The axis accompanying the pint spread function, normalized to
            R_ap.
            
            
        Calculated Transfer Matrix
        --------------------------
        tmat : np.ndarray (nxo, nyo, nxi, nyi)
            The calculated transfer matrix. 
            
            
        Dimensions
        ----------
        R_ap : u.Quantity
            The radius of the pinhole apertures.
            
        L_det : u.Quantity
            The distance of  the detector from the pinhole array. Together with
            the magnification this defines the detector location.
            
        
        """
        
        self.xo = None
        self.yo = None
        self.xi = None
        self.yi = None
        self.mag = None
        self.ap_xy = None
        self.psf = None
        self.psf_ax = None
        
        self.R_ap = None
        self.L_det = None
        
        super().__init__()
        

        if len(args)==0:
            pass
        elif len(args) == 1:
            self.path = args[0]
            self.load(self.path)
        else:
            raise ValueError(f"Invalid number of parameters: {len(args)}")
            
            
            
        
   
            
            
    @property
    def nxo(self):
        return self.xo.size
    @property
    def nyo(self):
        return self.yo.size
    @property
    def nxi(self):
        return self.xi.size
    @property
    def nyi(self):
        return self.yi.size
    
    
    @property
    def dxo(self):
        return np.mean(np.gradient(self.xo))
    @property
    def dyo(self):
        return np.mean(np.gradient(self.yo))
    @property
    def dxi(self):
        return np.mean(np.gradient(self.xi))
    @property
    def dyi(self):
        return np.mean(np.gradient(self.yi))
    
    
    @property
    def xo_scaled(self):
        self._check_units_set()
        return self.xo*self.R_ap

    @property
    def yo_scaled(self):
        self._check_units_set()
        return self.yo*self.R_ap


    @property
    def xi_scaled(self):
        self._check_units_set()
        return self.xi*self.R_ap*self.mag
 
    @property
    def yi_scaled(self):
        self._check_units_set()
        return self.yi*self.R_ap*self.mag
    
    
    @property
    def oshape(self):
        return (self.xo.size, self.yo.size)
    @property
    def ishape(self):
        return (self.xi.size, self.yi.size)
    @property
    def osize(self):
        return self.xo.size * self.yo.size
    @property
    def isize(self):
        return self.xi.size * self.yi.size
    
    @property
    def shape(self):
        return (self.isize, self.osize)
    

    @cached_property
    def tmat(self):
        with h5py.File(self.path, 'r') as f:
            result = f['tmat'][...]   
        return result
    

    def _save(self, grp):
        """
        Saves the transfer matrix constants (but not the tmat itself) to
        an h5 file. If the path for the file is not already set, it must be
        provided using the path keyword. 
        
        To calculate and save the actual transfer matrix, use the
        calculate_tmat method.

        """
        super()._save(grp)

        if self.xo is not None:
            grp['xo'] = self.xo
            grp['yo'] = self.yo
            grp['xi'] = self.xi
            grp['yi'] = self.yi
            grp['mag'] = self.mag
            grp['ap_xy'] = self.ap_xy
            grp['psf'] = self.psf
            grp['psf_ax'] = self.psf_ax
        
            
        if self.R_ap is not None:
            grp['R_ap'] = self.R_ap.to(u.cm).value
            grp['R_ap'].attrs['unit'] = 'cm'
            grp['L_det'] = self.L_det.to(u.cm).value
            grp['L_det'].attrs['unit'] = 'cm'

        
            
    def _load(self, grp):
        
        super()._load(grp)
        
        if 'xo' in grp.keys():
            self.xo = grp['xo'][...]
            self.yo = grp['yo'][...]
            self.xi = grp['xi'][...]
            self.yi = grp['yi'][...]
            self.mag = grp['mag'][...]
            self.ap_xy = grp['ap_xy'][...]  
            self.psf = grp['psf'][...]
            self.psf_ax = grp['psf_ax'][...]        
            
        if 'R_ap' in grp.keys():
            self.R_ap = grp['R_ap'][...] * u.cm
            self.L_det = grp['L_det'][...] * u.cm


    def set_constants(self, *, xo, yo, xi, yi, 
                      mag, ap_xy, psf=None, psf_ax=None,
                      override_validation=False, **kwargs):
        """
        Set the constants that define the transfer matrix. 
        
        Paramters (all specified as keyword paramters). See TransferMatrix
        docstring. 
        
        
        Required: xo, yo, xi, yi, mag, ap_xy
        
        Optional: 
            psf, psf_ax
                Default: tophat function at R_ap
                
        If xo (or same for yo) is an np.ndarrays of length 2, then they will 
        be interpreted as min(xo), max(xo) and xo will be generated
        as np.arange(min(xo), min(yo), dxo) with dxo = dxi/mag
        
        """
        
        
        # Calculate psf and psf_ax if not provided
        if psf_ax is None:
            psf = np.concatenate((np.ones(50), np.zeros(50)))
            psf_ax = np.linspace(0, 2, num=100)
            
        if psf_ax is None:
            raise ValueError("If psf is provided, psf_ax is required.")
            
            
        error_list = []
        attrs = {'xo':xo, 'yo':yo, 'xi':xi, 'yi':yi,
                   'mag':mag, 'ap_xy':ap_xy, 'psf':psf,
                   'psf_ax':psf_ax}
            
        for key, val in attrs.items() :
            # If a u.Quantity is given, try to convert it to a 
            # dimensionless np.ndarray
            if isinstance(val, u.Quantity):
                try:
                    attrs[key] = val.to(u.dimensionless_unscaled).value
                except u.UnitConversionError:
                    error_list.append(f"{key} (units: {val.unit})")


            elif not isinstance(val, (np.ndarray, float, int)):
                error_list.append(f"{key} ({type(val)})")
                      
        if len(error_list) > 0:
            raise ValueError("All constant arguments must be dimensionless "
                             "u.Quantity arrays,"
                             "np.ndarrays, floats, or ints, "
                             "but the folllowing are not: "
                             f"{error_list}")
            
        else:
            
            for key, val in attrs.items():
                setattr(self, key, val)
                
        # If xo or yo is only size (2), assume they are min, max
        # values and expand
        if self.xo.size == 2:
            dxo = (self.dxi/self.mag)
            if dxo > np.abs(xo[1] - xo[0]):
                raise ValueError(f"Given xo bounds ({self.xo}) are too small "
                                 f" for resolution dxo={dxo}")
            else:
                # Add 1% to dxo so that we satisfy the validity condition
                # without issues with numerical precision.
                self.xo = np.arange(xo[0], xo[1], 1.01*dxo)
            
        if self.yo.size == 2:
            dyo = (self.dyi/self.mag)
            if dyo > np.abs(yo[1] - yo[0]):
                raise ValueError(f"Given yo bounds ({self.yo}) are too small "
                                 f" for resolution dyo={dyo}")
            else:
                self.yo = np.arange(yo[0], yo[1], 1.01*dyo)
            
             
        # Validate that dxo*mag <= dxi
        # If this condition is not satisfied, the TransferMatrix is
        # including spurious resolution in the object plane

        valid = ( (self.dxo*self.mag <= self.dxi) and 
                     (self.dyo*self.mag <= self.dyi) )
         
        if not valid:
            if override_validation:
                 
                warnings.warn("Warning: constants are invalid, violating "
                               "dxo*mag < dxi. Resulting transfer matrix will "
                               "have spurious object plane resolution.")
            else:
                    raise ValueError("Constants are invalid, violating "
                                   "dxo*mag < dxi"
                                   f"({self.dxo*self.mag} > {self.dxi}")
            
            
            
        
    

    
    def set_dimensions(self, R_ap=None, L_det=None, **kwargs):
        """
        Parameters
        ----------
        
        R_ap : u.Quantity
            Radius of the apertures
            
        L_det : u.Quantity
            Source to detector distance 
            
        """
        if isinstance(R_ap, u.Quantity):
            self.R_ap = R_ap.to(u.um)
        elif R_ap is None:
            self.R_ap = None
        else:
            raise ValueError("R_ap must be a u.Quantity with units of length")
            
        if isinstance(L_det, u.Quantity):
            self.L_det = L_det.to(u.cm)
        elif L_det is None:
            self.L_det = None
        else:
            raise ValueError("L_det must be a u.Quantity with units of length")
        

        
    @property
    def scale_set(self):
        return self.R_ap is not None and self.L_det is not None
    
    def _check_units_set(self):
        if not self.scale_set:
            raise ValueError("Must set both R_ap and L_det!")
        
        

    
    
    

    def calculate_tmat(self, path=None):
        """
        Calculates the transfer matrix and stores it to the path
    
        """
        
        if path is not None:
            self.path = path
    
        # Start by saving the paramters of the tmat to the path
        self.save(self.path)

    
        # Create 2D arrays
        xo, yo = np.meshgrid(self.xo, self.yo, indexing='ij')
        xi, yi = np.meshgrid(self.xi, self.yi, indexing='ij')
        xo = xo.flatten()
        yo = yo.flatten()
        xi = xi.flatten()
        yi = yi.flatten()
        
        
        # Calculate chunk size, chunking object plane together
        ideal_size = 10e6 #bytes
        data_bytes = 4 # Float32
        # Ideal chunk size in image plane (each pixel = 1 full object plane)
        chunk_size = int(ideal_size/(self.osize)/data_bytes)
        nchunks = np.ceil(self.isize/chunk_size).astype(np.int32)
        print(f"Chunk size: {chunk_size}")
        print(f"nchunks: {nchunks}")
        
        # Break the image array into chunks
        chunks = []
        for c in range(nchunks):
            a = c*chunk_size
            if c == nchunks-1:
                b = -1
            else:
                b = (c+1)*chunk_size
            
            chunks.append( [xo, yo, xi[a:b], yi[a:b], self.mag, 
                            self.ap_xy, 
                            self.psf, 
                            self.psf_ax, 
                            a, b] )
     
    
    
        with h5py.File(self.path, 'a') as f:
            # Erase the tmat dataset if it already exists
            if 'tmat' in f.keys():
                del f['tmat']
                
            # Create the tmat dataset
            # Extra object plane pixel is the background pixel
            f.create_dataset('tmat', (self.isize, self.osize+1), dtype='float32',
                             compression='gzip', compression_opts=3)
            
            # Set the transfer matrix for the background pixel
            f['tmat'][:, -1] = np.ones(self.isize)
        
            p = Pool()
            
            # Initialize a tqdm progress bar
            with tqdm.tqdm(total=len(chunks)) as pbar:
                
                # Loop through the mapped calculations on the parallel pool
                for i, result in enumerate(p.imap(_calc_tmat, chunks)):
                    
                    # Push up or down values near zero or 1
                    result[np.abs(result)<1e-3] = 0
                    result[np.abs(result-1)<1e-3] = 1
                    
                    # Store the result
                    a = chunks[i][-2]
                    b = chunks[i][-1]
                    f['tmat'][a:b, :-1] = result
                    
                    # Update the progress bar that this iteration is done
                    pbar.update()
                    
                    
            p.close()
            p.join()

def _calc_tmat(arg):
    """
    Calculate a chunk of a transfer matrix 
    """
    
    # Remaining two arguments in 'arg' are the start and stop indices
    # a and b, which are used in the pool to place the chunks properly but are
    # not used in this function.
    xo, yo, xi, yi, mag, ap_xy, psf, psf_ax, _, _ = arg
    
    # Compute the position of each point in the aperture plane
    xa =  xi[:, np.newaxis] + xo[np.newaxis, :]*(mag-1)/mag
    ya =  yi[:, np.newaxis] + yo[np.newaxis, :]*(mag-1)/mag
    
    
    """
    # Create vectors of offsets that cover the entire pixel in the
    # aperture plane
    # "pseudopixels"
    # TODO: how to calculate xa here appropriately...
    dxa = np.median(np.gradient(xa))
    dya = np.median(np.gradient(xa))
    xa_offset = np.linspace(-dxa/2, dxa/2, num=8)
    ya_offset = np.linspace(-dya/2, dya/2, num=8)
    # Add those offsets to the aperture positions in a new axis
    xa = xa[:, :, np.newaxis] + xa_offset[np.newaxis, np.newaxis, :]
    ya = ya[:, :, np.newaxis] + ya_offset[np.newaxis, np.newaxis, :]
    """

    # Compute the distance from this point to every aperture
    r = np.sqrt(  (xa[..., np.newaxis] - ap_xy[:,0])**2 + 
                  (ya[..., np.newaxis] - ap_xy[:,1])**2
                )
    
    # Store the distance to the nearest aperture
    # NOTE: we are ignoring contributions from multiple pinholes
    res = np.min(r, axis=-1)
    
    """
    # Average over the pseudopixels
    res = np.mean(res, axis=-1)
    """
    
    # Interpolate the value of the transfer matrix at this point
    # from the point spread function
    tmat = np.interp(res, psf_ax, psf)
    
    return tmat



        


if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    
    data_dir = os.path.join("C:\\","Users","pvheu","Desktop","data_dir")
    #data_dir = os.path.join('C:\\','Users','pheu','Data','data_dir')
    
    save_path = os.path.join(data_dir, '103955', 'tmat.h5')
   
    isize=101
    osize=101
    
    import numpy as np
    import time
    
    mag = 30
    R_ap = 150*u.um
    


    xi = np.linspace(-1, 1, num=isize)*u.cm / R_ap / mag
    yi= np.linspace(-1, 1, num=isize)*u.cm / R_ap / mag
    
    oscale = 0.01
    xo = np.array([-oscale, oscale])
    yo= np.array([-oscale, oscale])
    
    ap_xy = np.array([[0,0]])*u.cm / R_ap
    
    t = TransferMatrix()
    
    
    t.set_constants(xo=xo, yo=yo, xi=xi, yi=yi, 
                    mag=mag, ap_xy=ap_xy)
    
    t.save(save_path)

    
    print("Calculating tmat")
    t0 = time.time()
    t.calculate_tmat()
    
    print(f"Time: {time.time() - t0:.1f} sec")
    
    import matplotlib.pyplot as plt
    
    with h5py.File(save_path, 'r') as f:
        tmat = f['tmat'][...]

    
    xarr, yarr = np.meshgrid(t.xo, t.yo, indexing='ij')

    
    r = np.sqrt(xarr**2 + yarr**2).flatten()
    i = np.argmin(r)
    print(i)

    
    arr = tmat[:,-1]
    print(arr)
    #arr = np.mean(tmat, axis=0)
    arr = np.reshape(arr, (xi.size, yi.size))
    
    print(arr.shape)
    
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.pcolormesh( (xi*R_ap*mag).to(u.um).value, 
                  (yi*R_ap*mag).to(u.um).value, arr.T)
    

    
    
    