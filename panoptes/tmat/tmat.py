
import os
import astropy.units as u
import h5py
import numpy as np


from multiprocessing import Pool
import tqdm
    
from panoptes.util.misc import timestamp
from panoptes.util.misc import  find_file



class TransferMatrix (h5py.File): 

    def __init__(self, path):
        """
        Represents a transfer matrix between the object and image plane
        of a pinhole or penumbral imager.
        
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
            
        xo_lim, yo_lim, xi_lim, yi_lim : tuple of u.Quantity objects
            Limits on the 
            
            
        


        Paramters
        ---------
        
        arg -> file path or file ID


    
        """
        
        # Constants
        # xo,yo,xi,yi are stored as private attributes because their public
        # attributes will return the subset including any limits. 
        # if no subset is set, _xo = xo etc.
        self._xo = None
        self._yo = None
        self._xi = None
        self._yi = None
        self.mag = None
        self.ap_xy = None
        self.psf = None
        self.psf_ax = None
        
        
        # Dimensions
        self.R_ap = None
        self.L_det = None
        self.xo_lim = None
        self.yo_lim = None
        self.xi_lim = None
        self.yi_lim = None
        
        
        
        self.path = path
        #self.load()

    @property
    def tmat(self, slice=None):
        with h5py.File(self.path, 'r') as f:
            result = f['tmat'][slice]   
        return result


    def save(self, path=None):
    
        with h5py.File(self.path, 'w') as f:
            if self._xo is not None:
                f['xo'] = self._xo.to(u.dimensionless_unscaled).value
                f['yo'] = self._yo.to(u.dimensionless_unscaled).value
                f['xi'] = self._xi.to(u.dimensionless_unscaled).value
                f['yi'] = self._yi.to(u.dimensionless_unscaled).value
                f['mag'] = self.mag
                f['ap_xy'] = self.ap_xy
                f['psf'] = self.psf
                f['psf_ax'] = self.psf_ax
            
            # Tmat is saved from the calc_tmat routine
                
            if self.R_ap is not None:
                f['R_ap'] = self.R_ap.to(u.cm).value
                f['L_det'] = self.L_det.to(u.cm).value
                f['xo_lim'] = self.xo_lim.to(u.cm).value
                f['yo_lim'] = self.yo_lim.to(u.cm).value
                f['xi_lim'] = self.xi_lim.to(u.cm).value
                f['yi_lim'] = self.yi_lim.to(u.cm).value
        
            
    def load(self):
        
        with h5py.File(self.path, 'a') as f:
            if 'xo' in f.keys():
                self._xo = f['xo'][...]
                self._yo = f['yo'][...]
                self._xi = f['xi'][...]
                self._yi = f['yi'][...]
                self.mag = f['mag']
                self.ap_xy = f['ap_xy'][...]  
                self.psf = f['psf'][:]
                self.psf_ax = f['psf_ax'][:]          
                
            # Don't actually load the tmat! Tmat may be huge!
                
            if 'R_ap' in f.keys():
                self.R_ap = f['R_ap'] * u.cm
                self.L_det = f['L_det'] * u.cm
                self.xo_lim = f['xo_lim'] * u.cm
                self.yo_lim = f['yo_lim'] * u.cm
                self.xi_lim = f['xi_lim'] * u.cm
                self.yi_lim = f['yi_lim'] * u.cm

    def set_constants(self, xo, yo, xi, yi, mag, ap_xy, 
                      psf=None, psf_ax=None):
        """
        Set the constants that define the transfer matrix. 
        """
        self._xo = xo
        self._yo = yo
        self._xi = xi
        self._yi = yi
        self.mag = mag
        self.ap_xy = ap_xy
        self.psf = psf
        self.psf_ax = psf_ax

    
    def set_R_ap(self, R_ap):
        """
        Parameters
        ----------
        
        R_ap : u.Quantity
            Radius of the apertures
            
        """
        self.R_ap = R_ap

        
    def set_L_det(self, L_det):
        """
        L_det, along with the magnification which is part of the tmat, 
        defines the geometry along with R_ap.
        
        Parameters
        ----------

        L : u.Quantity
            Source to detector distance 
        """
        self.L_det = L_det
        
    @property
    def scale_set(self):
        return self.R_ap is not None and self.L_det is not None
    
    def _check_units_set(self):
        if not self.scale_set:
            raise ValueError("Must set both R_ap and L_det!")
        
        
    def _calc_limits(self, axis, scaled_axis, limits):
        """
        Given the dimensionless axis, the scaled axis, and some limits, 
        calculate the slice for that axis.
        
 
        """
        if limits is None:
            return slice(0, axis.size, None)
        
        elif isinstance(limits[0], u.Quantity):
            self._check_units_set()
            a = np.argmin(np.abs(scaled_axis - limits[0]))
            b = np.argmin(np.abs(scaled_axis - limits[1]))
            
        else:
            a = np.argmin(np.abs(axis - limits[0]))
            b = np.argmin(np.abs(axis - limits[1]))
        
        return slice(a,b,None)
        
    def set_limits(self, xi_lim=None, yi_lim=None, xo_lim=None, yo_lim=None ):
        """
        Sets the limits in each axis
        
        Each limit keyword must be a tuple of either u.Quantities or floats.
        
        If floats are provided, limits will be calculated using the dimensionless
        axes. 
        If u.Quantities are provided, limits will be calculated using the
        scaled axes.
    
        
        """
        # Set all the slices to cover the whole array for the calculation
        # of new clises
        self.xi_slice = slice(0, None)
        self.yi_slice = slice(0, None)
        self.xo_slice = slice(0, None)
        self.yo_slice = slice(0, None)
        
        # Calculate the new slices
        # If/else handles possiblity that scales have not been set yet
        if self.scale_set:
            self.xi_slice = self._calc_limits(self.xi, self.xi_scaled, xi_lim)
            self.yi_slice = self._calc_limits(self.yi, self.yi_scaled, yi_lim)
            self.xo_slice = self._calc_limits(self.xo, self.xo_scaled, xo_lim)
            self.yo_slice = self._calc_limits(self.yo, self.yo_scaled, yo_lim)
        else:
            self.xi_slice = self._calc_limits(self.xi, None, xi_lim)
            self.yi_slice = self._calc_limits(self.yi, None, yi_lim)
            self.xo_slice = self._calc_limits(self.xo, None, xo_lim)
            self.yo_slice = self._calc_limits(self.yo, None, yo_lim)
            
        # Store the limits
        self.xi_lim = xi_lim
        self.yi_lim = yi_lim
        self.xo_lim = xo_lim
        self.yo_lim = yo_lim
        

    @property
    def xo(self):
        return self._xo[self.xo_slice]
    @property
    def xo_scaled(self):
        self._check_units_set()
        return self._xo*self.R_ap
    
    @property
    def yo(self):
        return self._yo[self.yo_slice]
    @property
    def yo_scaled(self):
        self._check_units_set()
        return self._yo*self.R_ap

    
    @property
    def xi(self):
        return self._xi[self.xi_slice]
    @property
    def xi_scaled(self):
        self._check_units_set()
        return self._xi*self.R_ap*self.mag
    
    @property
    def yi(self):
        return self._yi[self.yi_slice]
    @property
    def yi_scaled(self):
        self._check_units_set()
        return self._yi*self.R_ap*self.mag

    @property
    def tmat_slice(self):
            return self._tmat[self.xi_slice, self.yi_slice, 
                              self.xo_slice, self.yo_slice]
        

    def calc_tmat(self):
        """
        Calculates the transfer matrix based on the set constants.

        """
        
        # Save the sizes 
        nxo = self._xo.size
        nyo = self._yo.size
        nxi = self._xi.size
        nyi = self._yi.size
        
        # Create 2D arrays
        xo, yo = np.meshgrid(self._xo.to(u.dimensionless_unscaled).value, 
                             self._yo.to(u.dimensionless_unscaled).value, indexing='ij')
        xi, yi = np.meshgrid(self._xi.to(u.dimensionless_unscaled).value, 
                             self._yi.to(u.dimensionless_unscaled).value, indexing='ij')
        xo = xo.flatten()
        yo = yo.flatten()
        xi = xi.flatten()
        yi = yi.flatten()
        
        
        # Calculate chunk size, chunking object plane together
        ideal_size = 10e6 #bytes
        data_bytes = 4 # Float32
        # Ideal chunk size in image plane (each pixel = 1 full object plane)
        chunk_size = int(ideal_size/(nxo*nyo)/data_bytes)
        nchunks = np.ceil(nxi*nyi/chunk_size).astype(np.int32)
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
                            self.ap_xy.to(u.dimensionless_unscaled).value, 
                            self.psf, 
                            self.psf_ax.to(u.dimensionless_unscaled).value, 
                            a, b] )
 
    

        with h5py.File(self.path, 'a') as f:
            # Erase the tmat dataset if it already exists
            if 'tmat' in f.keys():
                del f['tmat']
                
            # Create the tmat dataset
            f.create_dataset('tmat', (nxo*nyo, nxi*nyi), dtype='float32',
                             compression='gzip', compression_opts=3)
        
            with Pool() as p:
                
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
                        f['tmat'][:, a:b] = result
                        
                        # Update the progress bar that this iteration is done
                        pbar.update()




def _calc_tmat(arg):
    xo, yo, xi, yi, mag, ap_xy, psf, psf_ax, _, _ = arg
    
    # Compute the position of each point in the aperture plane
    xa =  xi[np.newaxis, :] + xo[:, np.newaxis]*(mag-1)/mag
    ya =  yi[np.newaxis, :] + yo[:, np.newaxis]*(mag-1)/mag
    
    # Compute the distance from this point to every aperture
    r = np.sqrt(  (xa[..., np.newaxis] - ap_xy[:,0])**2 + 
                  (ya[..., np.newaxis] - ap_xy[:,1])**2
                )
    
    # Store the distance to the nearest aperture
    # NOTE: we are ignoring contributions from multiple pinholes for now
    res = np.min(r, axis=-1)
    
    # Interpolate the value of the transfer matrix at this point
    # from the point spread function
    tmat = np.interp(res, psf_ax, psf)
    
    return tmat



        







if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    
    data_dir = os.path.join("C:\\","Users","pvheu","Desktop","data_dir")
    #data_dir = os.path.join('C:\\','Users','pheu','Data','data_dir')
    
    save_path = os.path.join(data_dir, '103955', 'tmat.h5')
   
    
    
    isize=100
    osize=20
    
    import numpy as np
    import time
    
    mag = 30
    R_ap = 150*u.um
    
    xo = np.linspace(-0.015, 0.015, num=osize)*u.cm / R_ap
    yo=np.linspace(-0.015, 0.015, num=osize)*u.cm / R_ap
    xi = np.linspace(-0.6,0.6, num=isize)*u.cm / R_ap / mag
    yi=np.linspace(-0.6,0.6, num=isize)*u.cm / R_ap / mag
    
    ap_xy = np.array([[0,0]])*u.cm / R_ap
    
    psf = np.concatenate((np.ones(50), np.zeros(50)))
    psf_ax = np.linspace(0, 2*R_ap, num=100) / R_ap
    
    
    t = TransferMatrix(save_path)
    
    
    t.set_constants(xo, yo, xi, yi, mag, ap_xy, psf=psf, psf_ax=psf_ax)
    
    t.save()

    
    print("Calculating tmat")
    t0 = time.time()
    t.calc_tmat()
    
    print(f"Time: {time.time() - t0:.1f} sec")
    
    import matplotlib.pyplot as plt
    
    with h5py.File(save_path, 'r') as f:
        tmat = f['tmat'][...]
        
    print(tmat.shape)
    
    
    xarr, yarr = np.meshgrid(xo.value, yo.value, indexing='ij')
    print(xarr.flatten())
    print(yarr.flatten())
    
    r = np.sqrt(xarr**2 + yarr**2).flatten()
    i = np.argmin(r)
    print(i)

    
    arr = tmat[i,:]
    #arr = np.mean(tmat, axis=0)
    arr = np.reshape(arr, (xi.size, yi.size))
    
    print(arr.shape)
    
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.pcolormesh( (xi*R_ap*mag).to(u.um).value, 
                  (yi*R_ap*mag).to(u.um).value, arr.T)
    

    
    
    