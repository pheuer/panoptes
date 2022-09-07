# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 09:33:39 2022

@author: pheu

import os
from HeuerLib.lle.kodi_analysis import *
data_dir = os.path.join("C:\\","Users","pvheu","Desktop","data_dir")
obj = XrayIP(103955, data_dir=data_dir, pinholes='D-PF-C-055_A')





ALL UNITS CM


"""

import numpy as np

import h5py

import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor

from scipy.optimize import fmin, minimize
from scipy.special import erf


from IPython import get_ipython


from panoptes.util.misc import _compressed


def _penumbra_model(axes, data, radius, amp, xc,yc, mag_r,sigma,background):
    """
    axes -> (xarr, yarr)
    data -> The data array to minimize to
    raius -> the TRUE radius, in cm
    
    the remaining are the quantities to minimize
    """
    xarr, yarr = axes
    r = np.sqrt((xarr-xc)**2 + (yarr-yc)**2)
    arr = amp*(1 - erf((r - mag_r*radius)/((mag_r-1)*sigma*np.sqrt(2))))/2 + background
    
    return np.sum((arr-data)**2/np.std(arr))



def _pinhole_array_model(xy_nom, xy_data, dx, dy, rot, mag_s, skew, skew_angle, 
                   hreflect, vreflect):
    """
    A model for comparing two sets of pinhole locations: the nominal ('nom')
    locations and the ones calculated from the data
    """
    xy2 = _adjust_xy(xy_nom, dx, dy, rot, mag_s, skew, skew_angle, 
                     hreflect, vreflect)
    
    # Minimizew the separation between points
    return np.sum( (xy2[:,0]-xy_data[:,0])**2 + (xy2[:,1]-xy_data[:,1])**2)
    #return np.sum( (xy2-xy_data)**2/np.std(xy2))
    
    


def _adjust_xy(xy, dx, dy, rot, mag_s, skew, skew_angle, hreflect, vreflect):
        rot = np.deg2rad(rot)
        skew_angle = np.deg2rad(skew_angle)
        
        # reflect
        if hreflect:
            xy[:,0] = xy[::-1,0] 
        if vreflect:
            xy[:,1] = xy[::-1,1] 
            
            
        # skew
        xy[:,0] = xy[:,0] * ((2*skew/(1+skew))*np.cos(skew_angle) - 
                            (2/(1+skew))*np.sin(skew_angle))
        xy[:,1] = xy[:,1] * ((2*skew/(1+skew))*np.sin(skew_angle) + 
                            (2/(1+skew))*np.cos(skew_angle))
        
        # magnify
        xy = xy*mag_s
            
        # Rotate
        r = np.sqrt(xy[:, 0]**2 + xy[:, 1]**2)
        theta = np.arctan2(xy[:, 1], xy[:, 0])
        xy[:,0] = r*np.cos(theta + rot)
        xy[:,1] = r*np.sin(theta + rot)
        
        # Translate
        xy[:, 0] = xy[:, 0] + dx
        xy[:, 1] = xy[:, 1] + dy
        
        return xy


class PinholeArray:
    
    def __init__(self, *args, pinhole_name=None, plots=False):
        """
        arg : str
            Either a path to a file from which the pinhole can be loaded 
            or nothing
        
        Notes
        -----
        
        Two magnifications are defined:
        
        mag_s = Magnification determined from aperture separation
        mag_r = Magnification determined from aperture radius
      
        Depending on the type of data, this magnification may correspond to
        the pinhole or radiography magnification. 
     
        """
        # Make plots appear in new windows
        # If statement guards against running outside of ipython
        # TODO: support plots outside of iPython...
        if get_ipython() is not None:
            get_ipython().run_line_magic('matplotlib', 'inline')
        
        
        self.plots = plots
        
        
        # Sto
        self.mag_r = None
        
        # Optimal adjustment for the pinhole array
        self.adjustment = {'dx':0, 'dy':0, 'rot':0, 'mag_s':1, 
                           'skew':1, 'skew_angle':0,
                           'hreflect':0, 'vreflect':0}
        
        
        
        if len(args) == 0:
            pass
        elif len(args) == 1:
            self.path = args[0]
            self.load(self.path)
        else:
            raise ValueError(f"Invalid number of arguments: {len(args)}")
            
        
        
 
        
       
        
        

        
            
        
    # *************************************************************************
    # Basic methods
    # *************************************************************************
            
    def set_pinhole_array(self, pinhole_name):
        self._defined_pinhole(pinhole_name)
        
        self._init_pinhole_variables()
        
        
    def _init_pinhole_variables(self):
        # Indices of pinholes to use for fitting the array model
        self.use_for_fit = np.ones(self.npinholes).astype(bool)
        # Indices of pinholes to stack
        self.use_for_stack = np.ones(self.npinholes).astype(bool)
        
        # The locations of the pinhole centers
        # (determined by fitting each aperture image)
        self.pinhole_centers = self.xy
        
        if self.plots:
            self.plot()
        
    
            
    def __getattr__(self, key):
        if key in self.adjustment.keys():
            return self.adjustment[key]
        else:
            print(list(self.adjustment.keys()))
            raise KeyError(f"PinholeArray has no attribute {key}")
            
            
    def adjust(self, dx=0, dy=0, rot=0, mag_s=1,
                  skew=1, skew_angle=0,
                  hreflect=0, vreflect=0,
                  ):
        """
        Translate, rotate, magnify/demagnify, reflect
        
        dx, dy -> cm
        
        rotate -> degrees
        
        mag_s -> scalar
        
        skew, skew_angle -> scalar, degrees
        
        hreflect, vreflect -> bool
        
        The order of operations is 
        
        - Reflect
        - Skew
        - Magnify
        - Rotate
        - Translate

        """
        xy = np.copy(self.xy_prime)

        self.xy = _adjust_xy(xy, dx, dy, rot, mag_s, skew, skew_angle, hreflect, vreflect)
        
        self.adjustment = {'dx':dx, 'dy':dy, 'rot':rot, 'mag_s':mag_s, 
                           'skew':skew, 'skew_angle':skew_angle,
                       'hreflect':hreflect, 'vreflect':vreflect}
        
       
    # *************************************************************************
    # Methods for loading and saving 
    # *************************************************************************
            
    def save(self, grp):
        """
        Save the data about this subset into an h5 group
          
        grp : h5py.Group or path string
            The location to save the h5 data. This could be the root group
            of its own h5 file, or a group within a larger h5 file.
              
            If a string is provided instead of a group, try to open it as
            an h5 file and create the group there
          
        """
        if isinstance(grp, str):
            with h5py.File(grp, 'w') as f:
                self._save(f)
        else:
            self._save(grp)
        
          
                
    def _save(self, grp):
        """
        See docstring for "save"
        """
    
        # Write pinhole information
        info_grp = grp.create_group('pinhole_info')
        info_grp['id'] = str(self.id)
        info_grp['diameter'] = self.diameter
        info_grp['diameter'].attrs['unit'] = 'cm'
        info_grp['material'] = str(self.material)
        info_grp['thickness'] = self.thickness
        info_grp['thickness'].attrs['unit'] = 'cm'
        info_grp['xy_prime'] = self.xy_prime
         
        # Save the pinhole fit + selection parameters
        adjustment_grp = grp.create_group('adjustment')
        for key, val in self.adjustment.items():
            adjustment_grp[key] = val
            
        grp['xy'] = self.xy
        grp['use_for_fit'] = self.use_for_fit
        grp['use_for_stack'] = self.use_for_stack
        grp['pinhole_centers'] = self.pinhole_centers
        
        
        if self.mag_r is not None: 
            grp['mag_r'] = float(self.mag_r)

    def load(self, grp):
           """
           Load a PinholeArray object from a file or hdf5 group
           
           grp : h5py.Group 
               The location from which to load the h5 data. This could be the 
               root group of its own h5 file, or a group within a larger h5 file.
           
           """
           if isinstance(grp, str):
               with h5py.File(grp, 'r') as f:
                   self._load(f)
           else:
               self._load(grp)
                
                        
    def _load(self, grp):
        """
        See documentation for 'load'
        """
        
        # Load pinhole information
        info_grp = grp['pinhole_info']
        self.id = str(info_grp['id'][...])
        self.diameter = info_grp['diameter'][...] 
        self.material = str(info_grp['material'][...])
        self.thickness = info_grp['thickness'][...] 
        self.xy_prime = info_grp['xy_prime'][...]
        
        
        # Save the pinhole fit + selection parameters
        adjustment_grp = grp['adjustment']
        for key,val in self.adjustment.items():
            self.adjustment[key] = adjustment_grp[key][...]
            
        self.xy = grp['xy'][...] 
        self.use_for_fit = grp['use_for_fit'][...]
        self.use_for_stack = grp['use_for_stack'][...]
        self.pinhole_centers = grp['pinhole_centers'][...]
         
         
        if 'mag_r' in grp.keys():
            self.mag_r = float(grp['mag_r'][...])
        else:
            self.mag_r = None
            
        self._init_pinhole_variables()
        

        

    
    @property
    def npinholes(self):
        return self.xy.shape[0]
            

    def _defined_pinhole(self, id):
        """
        Load data about a defined pinhole array from its id number
        """
        self.id = id
        
        self.xy, self.spacing, self.diameter, self.material, self.thickness = pinhole_array_info(id)

        # Store a copy of the unchanged coordinates for reference
        self.xy_prime = np.copy(self.xy) 
        
    
    def fit(self, xaxis, yaxis, data,
                      # Rough adjustment 
                      auto_rough_adjust = None,
                      rough_adjust=None,
                      # Aperture selection
                      use_apertures = None,
                      auto_select_apertures=False,
                      # Aperture fitting
                      aperture_model='penumbra',
                      plots=True):
        
        """
        Fits the pinhole array to data through a multiple step process.
        
        The fitting algorithm goes through the following steps. 
        
        1) Rough fit the pinhole array model to the data. The goal of this step
           is just to get the centers close enough for the subsequent steps
           of the analysis. 
           
        2) Select a subset of aperture images to fit individually to find
           their centers. 
           
        3) Fit the aperture array model to the center locations found in
           step (2). The resulting adjustment will be the best fit of the
           aperture model to the selected apertures.
        
        The optional keywords allow different parts of the
        routine to be programatically bypassed or configured.
        
        Parameters
        ----------
        
        xaxis, yaxis : np.ndarray [nx] [ny]
            The horizontal and vertical axes of the data
            
        data : np.ndarray [nx, ny]
            The data to be fit
            
        rough_adjust : dict, optional
            A dictionary containing a rough adjustment to fit the pinhole
            array model to the data. The dictionary can contain the following
            keys: dx, dy, rot, mag_s, skew, skew_angle, hreflect, vreflect
            
        auto_rough_adjust : bool, optional
            If True, an automatic rough adjustment algorithm will be used.
            If False (default) then user input or the `rough_adjust` keyword 
            will be required.
            
        use_apertures : bool array [napertures,], optional
            A boolean list or array indicating which apertures to include
            in the fit. If False (default) apertures will be selected
            automatically or with user input.
            
        auto_select_apertures : bool, optional
            If True, automatically select the apertures for the fit and skip
            asking for user input.
            
            
        aperture_model : str, optional
            The model with which to fit the aperture images to find their 
            centers (and, if applicable for that model, other paramters like
            the radius of a penumbral image). Default is 'penumbra'. Options
            are:
                - 'penumbra'
                - 'pinhole_supergaussian' # Not yet implemented! 
            
            
        plots : bool, optional
            If True (default) show plots at each stage of the fitting process

        
        
        """
        
        # Perform rough adjustment
        if auto_rough_adjust:
            raise NotImplementedError("Automatic rough adjustment is not yet implemented")
        else:
            # Roughly adjust the pinhole array to the image manually
                if rough_adjust is not None:
                    self.adjust(**rough_adjust)
                else:
                    self._manual_rough_adjust(xaxis, yaxis, data)
                
        if plots:
            self.plot_with_data(xaxis, yaxis, data)
        

        
        # Select pinholes to include in the remainder of the analysis
        if use_apertures is not None:
            self.use_for_fit = np.array(use_apertures)
        else:
            
            # TODO: this diameter calculation won't work as well for pinhole images
            # where the image is not the same size as the projected aperture. 
            # Add a tuning parameter for the size here for that? Or 
            # somehow extract that first? 
            
            # Compute the distance from the edge of the domain that an aperture needs
            # to be to be auto-selected
            border = -0.25 # cm
            
            if auto_select_apertures:
                self._auto_select_apertures(xaxis, yaxis, data, 
                                                   variable='fit',
                                                   border=border)
            else:
                self._manual_select_apertures(xaxis, yaxis, data,
                                                     variable='fit',
                                                     border=border)
    
        
        # Fit each aperture subimage with a model function to find its 
        # center
        if aperture_model == 'penumbra':
            self._fit_penumbra(xaxis, yaxis, data, plots=plots)
        elif aperture_model == 'pinhole_supergaussian':
            raise NotImplementedError("The pinhole supergaussian aperture model is not yet implemented")
        
        # Fit the centers with the array model to get a final global fit
        # to determine the optimal adjustment
        self._fit_array()
        
        if plots:
            self.plot_with_data(xaxis, yaxis, data)
        
        
    def _manual_rough_adjust(self, xaxis, yaxis, data):
        """
        Allow user to manually set a rough adjustment using a cli interface.
        """
        
        # Ensure inline plotting
        if get_ipython() is not None:
            get_ipython().run_line_magic('matplotlib', 'inline')
        
               
        state = {'dx':0, 'dy':0, 'rot':0, 'mag_s':1, 
                 'skew':1, 'skew_angle':0,
                 'hreflect':0, 'vreflect':0}

        print("Enter commands in format key=value.")
        print("enter 'help' for a list of commands")
        self.plot_with_data(xaxis, yaxis, data)
        
        while True:
            state_str = ', '.join([f"{k}:{v}" for k,v in state.items()])
            print(f"Current state: {state_str}")
            x = input(">")
            split = x.split('=')
            if x == 'help':
                print("Enter commands in format key=value."
                      " ** Commands ** "
                      "'help' -> print this documentation\n"
                      "'end' -> accept the current values\n"
                      "'dx' -> Translate the pinhole array horizontally (cm)\n"
                      "'dy' -> Translate the pinhole array vertically (cm)\n"
                      "'rot' -> Rotate the pinholearray (degrees)\n"
                      "'mag_s' -> Magnify or demagnify the pinhole array\n"
                      "'skew' -> Skew amplitude\n"
                      "'skew_angle' -> Skew angle (degrees)\n"
                      "'hreflect' -> Reflect the pinhole array horizontally (1/0 = True/False)\n"
                      "'vreflect' -> Reflect the pinhole array vertically (1/0 = True/False)\n"
                      )
                
            elif x == 'end':
                break
            
            elif len(split)==2 and str(split[0]) in state.keys():
                for key in state.keys():
                    if str(split[0])==key:
                        state[key] = float(split[1])
                        
                self.adjust(**state)
                self.plot_with_data(xaxis, yaxis, data)
                        
            else:
                print(f"Invalid input: {x}")
                
        print("Finished pinhole array rough adjustment")
        
        
        
    def _auto_select_apertures(self, xaxis, yaxis, data, variable='fit', 
                               border = 0):
        """
        Automatically select apertures to include in the fit.
        
        Currently the only available algorithm is to select all apertures that
        do not clip the edges of the image (based on their pinhole diameter)
        
        Parameters
        ----------
        
        xaxis, yaxis, data : np.ndarray
            Data and axes
            
            
        variable : str 
            Changes which list of apertures is changed. 
            
            - 'fit' : the list of apertures used to fit the array
            - 'stack' : the list of apertures to stack
        
        
        border : float
            Ignore apertures within this distance (in axis units) of the
            edge. If negative, include apertures within that region outside
            of the current image.
        
        """
        if variable == 'fit':
            use = self.use_for_fit
        elif variable == 'stack':
            use = self.use_for_stack

        # Auto-exclude apertures that are not within the current bounds
        use = (use *
                (self.xy[:,0] > np.min(xaxis) + border ) *
                (self.xy[:,0] < np.max(xaxis) - border ) *
                (self.xy[:,1] > np.min(yaxis) + border ) *
                (self.xy[:,1] < np.max(yaxis) - border)
               ).astype(bool)
        
        if variable == 'fit':
            self.use_for_fit = use
        elif variable == 'stack':
            self.use_for_stack = use
        
        
        
    def _manual_select_apertures(self, xaxis, yaxis, data,
                                 variable='fit',
                                 border = 0):
        """
        Take user input to select apertures to include or exclude in the fit
        using a graphical point and click interface.
        
        
        Parameters
        ----------
        
        xaxis, yaxis, data : np.ndarray
            Data and axes
            
            
        variable : str 
            Changes which list of apertures is changed. 
            
            - 'fit' : the list of apertures used to fit the array
            - 'stack' : the list of apertures to stack
        
        
        border : float
            Ignore apertures within this distance (in axis units) of the
            edge. If negative, include apertures within that region outside
            of the current image.

        """
        
        if variable == 'fit':
            use = self.use_for_fit
        elif variable == 'stack':
            use = self.use_for_stack
        
        
        # run auto-select to at least get an intelligently chosen baseline
        self._auto_select_apertures_to_fit(xaxis, yaxis, data, variable=variable,
                                           border=border)
        
        
        # Switch to qt plotting for interactive plots
        get_ipython().run_line_magic('matplotlib', 'qt')
        
        fig, ax = self.plot_with_data(xaxis, yaxis, data)
        print("Select the apertures")
        print("Green/red = Include/exclude")
        print("Left click on apertures to toggle")
        print("Middle mouse button click (or 'Enter' key) to re-plot or finish")
        while True:
            cursor = Cursor(ax, color='red', linewidth=1)
            cor = plt.ginput(-1, timeout=-1)  
            for c in cor:
                # Identify the closest index pinhole to this coordinate
                dist = np.sqrt((self.xy[:,0] - c[0])**2 +
                                (self.xy[:,1] - c[1])**2 )
                

                # Switch the value in the 'pinholes_use' array
                ind = np.argmin(dist)
                use[ind] = ~use[ind] 
                fig, ax = fig, ax = self.plot_with_data(xaxis, yaxis, 
                                                        data, fig, ax)
            
            if len(cor) == 0:
                break
            
        print("Finished selecting apertures")
        
        if variable == 'fit':
            self.use_for_fit = use
        elif variable == 'stack':
            self.use_for_stack = use
    
        # Switch back to inline plotting so as to not disturb the console 
        # plots
        if get_ipython() is not None:
            get_ipython().run_line_magic('matplotlib', 'inline')


    def _fit_penumbra(self, xaxis, yaxis, data, plots=True):
        """
        Fit the selected aperture images with the penumbra model to find
        their centers and estimate the magnification based on their radius.
        
        
        The algorithm works by taking a square region ~1.3x the expected
        diameter of the each aperture image and fitting it with a penumbra
        model (`_penumbral_model`). The fit is conducted in stages to speed
        up the computation. All but the final fit are performed on a thinned
        copy of the array to speed up the fits.
        
        1) First just the amplitude of the model is fit
        2) The amplitude + the center are fit
        3) The amplitude + the magnification are fit
        4) All the parameters are fit 
        5) A final fit is conducted using the full resolution sub arrays to
           achieve the maximum possible accuracy in the fit results.
        
        
        """
        
        
        width = 1.3*self.mag_s*0.5*self.diameter
        radius = self.diameter/2
        
        
        use_ind = [i for i,v in enumerate(self.use_for_fit) if v==1]
        
        # Save the center coordinates and magnification from each fit
        self.pinhole_centers = np.zeros([len(use_ind),2])
        mag_r = np.zeros([len(use_ind)])
        
        

        for i, ind in enumerate(use_ind):
            print(f"Fitting aperture {i+1}/{len(use_ind)} (# {ind})")
        
          
            # Find the indices that bound the subregion around this aperture
            xa = np.argmin(np.abs(xaxis - (self.xy[ind,0] - width)))
            xb = np.argmin(np.abs(xaxis - (self.xy[ind,0] + width)))
            ya = np.argmin(np.abs(yaxis - (self.xy[ind,1] - width)))
            yb = np.argmin(np.abs(yaxis - (self.xy[ind,1] + width)))

                    
            # Cut out the subregion, make an array of axes for the points
            arr = data[xa:xb, ya:yb]
            x = xaxis[xa:xb]
            y = yaxis[ya:yb]
            
            axes = np.meshgrid(x, y, indexing='ij')
            
            
            # Reduce the resolution of the image to speed up the initial fits
            # We want to reduce it to be ~30 px
            if np.max(arr.shape) > 60:
                chunk = int(arr.shape[0]/30)
                print(f"Using chunk: {chunk}")
                x_c, y_c, arr_c,  = _compressed(x, y, arr, chunk=chunk)
            else:
                x_c, y_c, arr_c = x, y,  arr
                
            axes_c = np.meshgrid(x_c, y_c, indexing='ij')
            
            # Follow an interative fitting procedure to zero in on the best fit
            # 1) Fit just the amplitude
            # 2) Fit the amplitude and the center
            # 3) Fit the amplitude and the magnification
            # 4) Fit everything at once
            
            # Defaults for all parameters
            # args = axes, data, radius amp, xc,yc, mag_r,sigma,background
            p = [1,
                  self.xy[ind,0],
                  self.xy[ind,1],
                  self.mag_s, # Guess that mag_r = mag_s
                  1e-4, np.mean(data)]
     

            # AMPLITUDE FIT
            print("...amplitude fit")
            model = lambda args: _penumbra_model(axes_c, arr_c, radius, args[0], *p[1:])
            guess = [1]
            p[0] = fmin(model, guess, disp=False)
            
            
            # AMPLITUDE + CENTER FIT
            print("...amplitude + center fit")
            model = lambda args: _penumbra_model(axes_c, arr_c, radius, *args, *p[3:])
            guess = p[:3]
            p[:3] = fmin(model, guess, disp=False)
            
            # AMPLITUDE + MAGNFICATION FIT
            print("...amplitude + magnification fit")
            model = lambda args: _penumbra_model(axes_c, arr_c, radius, 
                                                  args[0], p[1], p[2], 
                                                  args[1], *p[4:])
            guess = [p[0], p[3]]
            res = fmin(model, guess, disp=False)
            p[0]=res[0]
            p[3]=res[1]
            
            # FIT EVERYTHING
            print("...final fit w/ all variables")
            model = lambda args: _penumbra_model(axes_c, arr_c, radius, *args)
            p = fmin(model, p, disp=False)
            
            # TODO: uncomment for full resolution!
            # FIT EVERYTHING
            #print("...final fit w/ all variables (full resolution)")
            #model = lambda args: _penumbra_model(axes, arr, radius, *args)
            #p = fmin(model, p, disp=False)
            
            # xc, yc, mag_r
            self.pinhole_centers[i,0] = p[1]
            self.pinhole_centers[i,1] = p[2]
            mag_r[i] = p[3]
            
            if plots:
                fig, ax = plt.subplots()
                ax.set_aspect('equal')
                ax.pcolormesh(x, y, arr.T)
            
                ax.scatter(self.xy[ind,0], self.xy[ind,1], 
                           color='red', label='Old center')
                
                circle = plt.Circle((p[1], p[2]),  radius*p[3], color='black',
                                    fill=False, linestyle='dashed', label='Diameter')
                ax.add_patch(circle)
                
                ax.scatter(p[1], p[2], color='black', label='New center')
                ax.set_title(f"Aperture # {ind}")
                ax.legend()
                plt.show()
            
        self.mag_r = float(np.mean(mag_r))
        

    
    def _fit_array(self):
        """
        Fit the aperture array model to the centers of the aperatures
        (identified in previous steps.) The fit is conducted in several steps
        to speed it up: 
            
            
        1) The translation, rotation, and magnification are fit
        2) The skew is fit
            
            
        Finally, the best result adjustment is applied to the PinholeArray
        """
        
        if self.pinhole_centers is None:
            raise ValueError("pinhole_centers not defined. Individual "
                             "apertures need to be fit first.")
        
        # Nominal locations for the pinholes being fit
        xy_nom = self.xy_prime[self.use_for_fit,:]
        
        # Fit the centers found with the pinhole array model
        print("Fitting pinhole array model to points")

        # Fit magnification, rotation, x, y
        p = [self.dx,self.dy,self.rot,self.mag_s,
             self.skew, self.skew_angle,self.hreflect,self.vreflect]
        
        
        ds = self.spacing*self.mag_s
        bounds = [ (-ds+self.dx, ds+self.dx),
                  (-ds+self.dy, ds+self.dy),
                  (self.rot-10, self.rot+10),
                  (max([self.mag_s-5, 0.1]), self.mag_s+5),
                  (max([self.skew-0.2, 0.1]), self.skew+0.2),
                  (self.skew_angle-90, self.skew_angle+90),
                  None, None]
         
        # TODO
        # Try using fmin_bound: guess realistic bounds based on the rough fit
        # and physical constraints, eg mag > 0
        # 
        # Try running these fits in a loop to iterate on skew vs rotation+mag?

        print("...Fitting translation, rotation, and magnification")
        model = lambda args: _pinhole_array_model(xy_nom, 
                                                  self.pinhole_centers,
                                                  *args, *p[4:])
        guess = p[0:4]
        res = minimize(model, guess, bounds=bounds[0:4])
        p[0:4] = res.x
        print(p)
        
        print("...Fitting skew and skew angle")
        model = lambda args: _pinhole_array_model(xy_nom, 
                                                  self.pinhole_centers,
                                                  *p[0:4], *args, *p[6:])
        guess = p[4:6]
        res = minimize(model, guess, bounds=bounds[4:6])
        p[4:6] = res.x
        print(p)


        self.adjust(dx=p[0], dy=p[1], rot=p[2], mag_s=p[3],
                             skew=p[4], skew_angle=p[5], hreflect=p[6],
                             vreflect=p[7])
        
        
        error = np.sqrt( (self.xy[self.use_for_fit, 0]- 
                          self.pinhole_centers[:,0])**2 +
                                (self.xy[self.use_for_fit, 1]- 
                                 self.pinhole_centers[:,1])**2 )
        
        print(f"Pinhole fit max error: {np.max(error)*1e4:.2} um")
        print(f"Pinhole fit mean error: {np.mean(error)*1e4:.2} um")
        
        print("Done with fine adjustment")
        
        

        
    def stack(self, xaxis, yaxis, data, width=None, use_apertures=None,
              auto_select_apertures=True):
        """
        Stack the data from the selected pinholes
        
        
        Paramters
        ---------
        width : float
            The width and height of the stacked image
        
        
        use_apertures : bool array [napertures,], optional
            A boolean list or array indicating which apertures to include
            in the fit. If False (default) apertures will be selected
            automatically or with user input.
            
        auto_select_apertures : bool, optional
            If True, automatically select the apertures for the fit and skip
            asking for user input.
        
        """
        
        if self.pinhole_centers is None:
            raise ValueError("pinhole_centers not defined. Individual "
                             "apertures need to be fit first.")
            
            
        # Select pinholes to include in the remainder of the analysis
        if use_apertures is not None:
            self.use_for_stack = np.array(use_apertures)
        else:
            
            # TODO: this diameter calculation won't work as well for pinhole images
            # where the image is not the same size as the projected aperture. 
            # Add a tuning parameter for the size here for that? Or 
            # somehow extract that first? 
            
            # Compute the distance from the edge of the domain that an aperture needs
            # to be to be auto-selected
            border = -0.25 # cm
            
            if auto_select_apertures:
                self._auto_select_apertures(xaxis, yaxis, data, 
                                                   variable='stack',
                                                   border=border)
            else:
                self._manual_select_apertures(xaxis, yaxis, data,
                                                     variable='stack',
                                                     border=border)
            
        
        if width is None:
            width = 1.5*(self.mag_r*self.diameter)
            
        # Calculate the half-width in pixels
        dx = np.mean(np.gradient(xaxis))
        dy = np.mean(np.gradient(yaxis))
        wx = int(width/2/dx)
        wy = int(width/2/dy)
        
        # Indices of the apetures to include
        use_ind = [i for i,v in enumerate(self.use_for_stack) if v==1]
        
        
        
    
        pad = 3*np.max([wx, wy])
        print(f"pad: {pad}")
        
        data =  np.pad(data, pad_width=pad,
                       mode='constant',
                       constant_values=np.nan)
    
        
        # Pad the axes with linear extrapolation 
        xaxis = np.pad(xaxis, pad_width=pad, 
                       mode='linear_ramp',
                       end_values=(np.min(xaxis)-pad*dx, np.max(xaxis)+pad*dx) )

        yaxis = np.pad(yaxis, pad_width=pad, 
                       mode='linear_ramp',
                       end_values=(np.min(yaxis)-pad*dy, np.max(yaxis)+pad*dy) )
        
        
        # For each aperture, select the data region around each aperture
        # and sum them
        output = np.zeros([2*wx, 2*wy, len(use_ind)])
        
        for i, ind in enumerate(use_ind):
            x0 = np.argmin(np.abs(xaxis - (self.pinhole_centers[i,0])))
            y0 = np.argmin(np.abs(yaxis - (self.pinhole_centers[i,1])))
            
            data_subset = data[x0-wx:x0+wx, y0-wy:y0+wy]

            output[...,i] = data_subset
            
        output = np.nanmean(output, axis=-1)

        # Calculate new x and y axes centered on this image
        xaxis = np.linspace(-width/2, width/2, 2*wx)
        yaxis = np.linspace(-width/2, width/2, 2*wy)
        
        return xaxis, yaxis, output
        
    
    
    def plot_with_data(self, xaxis, yaxis, data, *args):
        # Clear figure if it already exists

        if len(args) == 0:
            fig = plt.figure(figsize=(10,3))
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        else:
            fig, ax = args
            ax.clear()
        
        # TODO: only compress if necessary!
        if np.max(data.shape) > 400:
            chunk = int(np.max(data.shape)/200)
            xaxis, yaxis, data = _compressed(xaxis, yaxis, data, chunk=chunk)
        
        
        ax.set_aspect('equal')
        ax.pcolormesh(xaxis, yaxis,
                      data.T, vmax=10*np.median(data))
        
        ax.set_xlabel("X (cm)")
        ax.set_ylabel("Y (cm)")
        
        ax.set_xlim(np.min(xaxis), np.max(xaxis))
        ax.set_ylim(np.min(yaxis), np.max(yaxis))
        

        if self.xy is not None:
            ax.scatter(self.xy[self.use_for_fit,0], 
                       self.xy[self.use_for_fit,1], color='green')
            
            ax.scatter(self.xy[~self.use_for_fit,0], 
                       self.xy[~self.use_for_fit,1], color='red')
            
            
        if self.mag_r is not None:
            ax.scatter(self.pinhole_centers[:,0], 
                       self.pinhole_centers[:,1],  color='black',
                       marker='x')
            
            radius = self.diameter/2
            for i in range(self.pinhole_centers.shape[0]):
                circle = plt.Circle((self.pinhole_centers[i,0], 
                                     self.pinhole_centers[i,1]),  
                                    radius*self.mag_r, color='black',
                                    fill=False, linestyle='dashed')
                ax.add_patch(circle)
                

        plt.show()
            
        return  fig, ax
            
            
    def plot_array(self):
        """
        Plot the pinhole array with the current adjustments applied
        """
        
        fig, ax = plt.subplots()
        
        ax.set_aspect('equal')
        ax.scatter(self.xy[:,0], self.xy[:,1])
        ax.set_title(self.id)
        
        plt.show()
        
        
        
def pinhole_array_info(name):
    
    """
    Given a pinhole ID as a string, returns information about that pinhole 
    
    
    Returns
    -------
    
    results : tuple: 
        
        Contains: 
            - xy : array of pinhole locations in cm in the pinhole 
              plane [Npinholes, 2]
            - spacing : pinhole horizontal spacing
            - diameter : pinhole diameter
            - material : pinhole material
            - thickness : pinhole thickness
            
    
    """
    
    
    if name == 'D-PF-C-055_A':
        # LLE 210x Ta array, 0.3 mm diameter, 0.6 mm spacing
        spacing = 0.06 # cm, horiz separation
        spacing_vert = spacing*np.cos(np.deg2rad(30)) #mm, 0.182 row separation
        
        # Define the number of pinholes per row
        N_row = np.array([7, 10, 11, 12, 13, 14, 15, 16, 15, 16, 15, 14, 13, 12, 11, 10, 7])
        # Create a cumulative array, starting with a zero, for indexing
        ncum = np.cumsum(N_row)
        nslice = np.concatenate((np.zeros(1), ncum)).astype(np.int32)
        
        npinholes = np.sum(N_row)
        xy = np.zeros([npinholes,2])
    
        for i in range(N_row.size):#i = 1:length(N_row)
            x_ind = np.arange(N_row[i]) - (N_row[i]-1)/2
            y_ind = i - (N_row.size-1)/2
            
            xloc = x_ind*spacing*np.ones(x_ind.size)
            yloc = y_ind*spacing_vert*np.ones(x_ind.size)
            xy[nslice[i]:nslice[i+1], :] = np.array([xloc, yloc]).T

        # Remove 'blank' position
        # index row/column to skip
        # Both 1 indexed!
        blanks = [ (5, 10) ] 
        keep = np.ones(npinholes, dtype=bool)
        for i, blank in enumerate(blanks):
            row = blank[0]-1
            ind = ncum[row] + blank[1]
            keep[ind] = False
            
        xy = xy[keep, :]
        
        diameter = 0.03 # cm
        material = 'Ta';
        thickness = 0.02 # cm
        
        
        return  xy, spacing, diameter, material, thickness