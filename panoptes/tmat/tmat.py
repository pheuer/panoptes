
import os
import astropy.units as u
import h5py
import numpy as np


def calc_tmat(xo, yo, dxo, xi, yi, dxi, mag_r,  xy_ap):
    pass



class TransferMatrix: 
    tmat_dir = os.path.join('//expdiv','kodi','TransferMatrices')


    def __init__(self, *args):
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
            
        dxo : float
            Point separation in the object plane.
            Normalized to R_ap
            
        xi, yi : 1D np.ndarray
            Axes for each dimension of the transfer matrix in the image plane.
            Normalized to R_ap*mag_r
            
        dxi : float
            Point separation in the image plane.
            Normalized to R_ap*mag_r
            
            
        mag_r : float
            Radiography magnification: 1 + L2/L1. 
            The pinhole magnification L2/L1 = mag_r -1 is available as the
            attribute `mag_p`
            
        xy_ap : np.ndarray (N_ap, 2)
            The locations of the aperture centers in the pinhole plane.
            Normalize to R_ap (??)
            
            
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
        self.dxo = None
        self._xi = None
        self._yi = None
        self.dxi = None
        self.mag_r = None
        self.xy_ap = None
        
        # Tmat
        # tmat is also a private attribute because the public version includes
        # any limits that are set.
        self._tmat = None
        
        # Dimensions
        self.R_ap = None
        self.L_det = None
        self.xo_lim = None
        self.yo_lim = None
        self.xi_lim = None
        self.yi_lim = None
        
        
        
        
        
        # Get the file (if an argument was provided)
        if len(args) >= 1: 
            # If it's an id, look for the file in the configured tmat directory
            
            # If it's a filepath, assume that is the filepath
            
            
            # If the file contains the tmat constants, load them and set
            # them for the object
            
            
            # If the file contains a computed tmat, load that too
            
            pass
            

    @property
    def mag_p(self): 
        return self.mag_r - 1
    

        
    def set_constants(self, xo, yo, dxo, xi, yi, dxi, mag_r, xy_ap):
        """
        Set the constants that define the transfer matrix. 
        """
        
        
        self._xo = xo
        self._yo = yo
        self.dxo = dxo
        self._xi = xi
        self._yi = yi
        self.dxi = dxi
        self.mag_r = mag_r
        self.xy_ap = xy_ap
        
    def save_constants(self, path):
        with h5py.File(path, 'w') as f:
            f['xo'] = self._xo
            f['yo'] = self._yo
            f['xi'] = self._xi
            f['yi'] = self._yi
            f['dxo'] = self.dxo
            f['dxi'] = self.dxi
            f['mag_r'] = self.mag_r
            f['xy_ap'] = self.xy_ap
            
    def load_constants(self, path):
        with h5py.File(path, 'r') as f:
            self._xo = f['xo'][...]
            self._yo = f['yo'][...]
            self._xi = f['xi'][...]
            self._yi = f['yi'][...]
            self.dxo = f['dxo']
            self.dxi = f['dxi']
            self.mag_r = f['mag_r']
            self.xy_ap = f['xy_ap'][...]    
            
    
    def save_tmat(self, path):
        with h5py.File(path, 'w') as f: 
            f['tmat'] = self._tmat
            
    def load_tmat(self, path):
        with h5py.File(path, 'w') as f: 
            self._tmat = f['tmat'][...]

    def save_dimensions(self, path):
        with h5py.File(path, 'w') as f:
            f['R_ap'] = self.R_ap.to(u.cm).value
            f['L_det'] = self.L_det.to(u.cm).value
            f['xo_lim'] = self.xo_lim.to(u.cm).value
            f['yo_lim'] = self.yo_lim.to(u.cm).value
            f['xi_lim'] = self.xi_lim.to(u.cm).value
            f['yi_lim'] = self.yi_lim.to(u.cm).value
            
    def load_dimensions(self, path):
        with h5py.File(path, 'r') as f:
            self.R_ap = f['R_ap'] * u.cm
            self.L_det = f['L_det'] * u.cm
            self.xo_lim = f['xo_lim'] * u.cm
            self.yo_lim = f['yo_lim'] * u.cm
            self.xi_lim = f['xi_lim'] * u.cm
            self.yi_lim = f['yi_lim'] * u.cm
    

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
        return self._xi*self.R_ap*self.mag_r
    
    @property
    def yi(self):
        return self._yi[self.yi_slice]
    @property
    def yi_scaled(self):
        self._check_units_set()
        return self._yi*self.R_ap*self.mag_r

    @property
    def tmat(self):
            return self._tmat[self.xi_slice, self.yi_slice, 
                              self.xo_slice, self.yo_slice]
            
