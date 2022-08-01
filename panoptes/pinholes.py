# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 09:33:39 2022

@author: pheu

import os
from HeuerLib.lle.kodi_analysis import *
data_dir = os.path.join("C:\\","Users","pvheu","Desktop","data_dir")
obj = XrayIP(103955, data_dir=data_dir, pinholes='D-PF-C-055_A')


"""

import numpy as np

import astropy.units as u

import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor

from scipy.optimize import fmin, minimize
from scipy.special import erf

# Make plots appear in new windows
from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'qt')
get_ipython().run_line_magic('matplotlib', 'inline')

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
    
    def __init__(self, arg, plots=False):
        """
        
        
        Notes
        -----
        
        Two magnifications are defined:
        
        mag_s = Magnification determined from aperture separation
        mag_r = Magnification determined from aperture radius
      
        Depending on the type of data, this magnification may correspond to
        the pinhole or radiography magnification. 
     
        """
        
        
        
        self.plots = plots
        
        # First we will check to see if the pinhole is defined
        self._defined_pinhole(arg)
        
        # TODO Make a save format for pinholes
        
        # If not, assume the argument is a path to a text file with
        # the pinhole info in it.
        
        
        # Indices of pinholes to use in analysis
        self.use = np.ones(self.npinholes).astype(bool)
        
        # Store the locations of the pinhole centers
        # (determined by fitting each aperture image)
        self.pinhole_centers = None
        
        # Optimal adjustment for the pinhole array
        self.adjustment = {'dx':0, 'dy':0, 'rot':0, 'mag_s':1, 
                           'skew':1, 'skew_angle':0,
                           'hreflect':0, 'vreflect':0}

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
        
        
    def _defined_pinhole(self, name):
        self.name = name
        
        if name == 'D-PF-C-055_A':
            # LLE 210x Ta array, 0.3 mm diameter, 0.6 mm spacing
            self.spacing = 0.06 # cm, horiz separation
            spacing_vert = self.spacing*np.cos(np.deg2rad(30)) #mm, 0.182 row separation
            
            # Define the number of pinholes per row
            N_row = np.array([7, 10, 11, 12, 13, 14, 15, 16, 15, 16, 15, 14, 13, 12, 11, 10, 7])
            # Create a cumulative array, starting with a zero, for indexing
            ncum = np.cumsum(N_row)
            nslice = np.concatenate((np.zeros(1), ncum)).astype(np.int32)
            
            self.npinholes = np.sum(N_row)
            xy = np.zeros([self.npinholes,2])
        
            for i in range(N_row.size):#i = 1:length(N_row)
                x_ind = np.arange(N_row[i]) - (N_row[i]-1)/2
                y_ind = i - (N_row.size-1)/2
                
                xloc = x_ind*self.spacing*np.ones(x_ind.size)
                yloc = y_ind*spacing_vert*np.ones(x_ind.size)
                xy[nslice[i]:nslice[i+1], :] = np.array([xloc, yloc]).T

            # Remove 'blank' position
            # index row/column to skip
            # Both 1 indexed!
            blanks = [ (5, 10) ] 
            keep = np.ones(self.npinholes, dtype=bool)
            for i, blank in enumerate(blanks):
                row = blank[0]-1
                ind = ncum[row] + blank[1]
                keep[ind] = False
                
            xy = xy[keep, :]
            self.xy = xy
            # Store a copy of the unchanged coordinates for reference
            self.xy_prime = np.copy(xy) 
            
            self.npinholes = np.sum(keep)
            
            self.diameter = 0.03*u.cm
            self.material = 'Ta';
            self.thickness = 0.02*u.cm
        
    
    def fit(self, xaxis, yaxis, data,
                      rough_adjust=None,
                      auto_select_apertures=False,
                      fit_pinholes=None):
        
        # Roughly adjust the pinhole array to the image manually
        if rough_adjust is not None:
            self.adjust(**rough_adjust)
            self.plot_with_data(xaxis, yaxis, data)
        else:
            self._rough_adjust(xaxis, yaxis, data)
        
        # Select pinholes to include in the remainder of the analysis
        self._select_apertures(xaxis, yaxis, data, 
                               auto_select=auto_select_apertures)
    
        # Fit each aperture subimage with a model function to find its 
        # center and guess at mag_r
        self.fit_penumbra(xaxis, yaxis, data)
        
        # Fit the centers with the array model to get a final global fit
        # to determine the optimal adjustment
        self.fit_array()
        
        self.plot_with_data(xaxis, yaxis, data)
        
        
    def _rough_adjust(self, xaxis, yaxis, data):
        
        # Ensure inline plotting
        get_ipython().run_line_magic('matplotlib', 'inline')
        
        #self.plot_with_data(xaxis, yaxis, data)
               
        state = {'dx':0, 'dy':0, 'rot':0, 'mag_s':1, 
                 'skew':1, 'skew_angle':0,
                 'hreflect':0, 'vreflect':0}

        print("Enter commands in format key=value.")
        print("enter 'help' for a list of commands")
        self.plot_with_data(xaxis, yaxis, data)
        
        for i in range(1000):
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
        
        
    def _select_apertures(self, xaxis, yaxis, data, auto_select=False):
        
        # Compute the distance from the edge of the domain that an aperture needs
        # to be to be auto-selected
        offset = 1.5*(0.5*self.diameter.to(u.cm).value)*self.mag_s
        
        # Auto-exclude apertures that are not within the current bounds
        self.use = (self.use *
                (self.xy[:,0] > np.min(xaxis) + offset ) *
                (self.xy[:,0] < np.max(xaxis) - offset ) *
                (self.xy[:,1] > np.min(yaxis) + offset ) *
                (self.xy[:,1] < np.max(yaxis) - offset)
               ).astype(bool)
        
        if not auto_select:
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
                    self.use[ind] = ~self.use[ind] 
                    fig, ax = fig, ax = self.plot_with_data(xaxis, yaxis, 
                                                            data, fig, ax)
                
                if len(cor) == 0:
                    break
                
            print(f"Finished selecting apertures")
        
        # Switch back to inline plotting so as to not disturb the console 
        # plots
        get_ipython().run_line_magic('matplotlib', 'inline')
        self.plot_with_data(xaxis, yaxis, data)


    def fit_penumbra(self, xaxis, yaxis, data):
        w = 1.3*(self.mag_s*0.5*self.diameter.to(u.cm).value)
        radius = self.diameter.to(u.cm).value/2
        
        
        use_ind = [i for i,v in enumerate(self.use) if v==1]
        
        # Save the center coordinates and magnification from each fit
        self.pinhole_centers = np.zeros([len(use_ind),2])
        mag_r = np.zeros([len(use_ind)])

        for i, ind in enumerate(use_ind):
            print(f"Fitting aperture {i+1}/{len(use_ind)}")
        
            # Find the indices that bound the subregion around this aperture
            xa = np.argmin(np.abs(xaxis - (self.xy[ind,0]-w)))
            xb = np.argmin(np.abs(xaxis - (self.xy[ind,0]+w)))
            ya = np.argmin(np.abs(yaxis - (self.xy[ind,1]-w)))
            yb = np.argmin(np.abs(yaxis - (self.xy[ind,1]+w)))
            
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
                  1e-3, np.mean(data)]
     

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
            
            # FIT EVERYTHING
            print("...final fit w/ all variables (full resolution)")
            model = lambda args: _penumbra_model(axes, arr, radius, *args)
            p = fmin(model, p, disp=False)
            
            # xc, yc, mag_r
            self.pinhole_centers[i,0] = p[1]
            self.pinhole_centers[i,1] = p[2]
            mag_r[i] = p[3]
            

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
            
        self.mag_r = np.mean(mag_r)
        

    
    def fit_array(self):
        
        if self.pinhole_centers is None:
            raise ValueError("pinhole_centers not defined. Individual "
                             "apertures need to be fit first.")
        
        # Nominal locations for the pinholes being fit
        xy_nom = self.xy_prime[self.use,:]
        
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
        model = lambda args: _pinhole_array_model(xy_nom, self.pinhole_centers,
                                                  *args, *p[4:])
        guess = p[0:4]
        res = minimize(model, guess, bounds=bounds[0:4])
        p[0:4] = res.x
        print(p)
        
        
        
        

        print("...Fitting skew and skew angle")
        model = lambda args: _pinhole_array_model(xy_nom, self.pinhole_centers,
                                                  *p[0:4], *args, *p[6:])
        guess = p[4:6]
        res = minimize(model, guess, bounds=bounds[4:6])
        p[4:6] = res.x
        print(p)

        
        
        
        self.adjust(dx=p[0], dy=p[1], rot=p[2], mag_s=p[3],
                             skew=p[4], skew_angle=p[5], hreflect=p[6],
                             vreflect=p[7])
        
        
        error = np.sqrt( (self.xy[self.use, 0]- self.pinhole_centers[:,0])**2 +
                                (self.xy[self.use, 1]- self.pinhole_centers[:,1])**2 )
        
        print(f"Pinhole fit max error: {np.max(error)*1e4:.2} um")
        print(f"Pinhole fit mean error: {np.mean(error)*1e4:.2} um")
        
        

        print("Done with fine adjustment")
        
        
        
    def stack(self, xaxis, yaxis, data, width=None):
        """
        Stack the data from the selected pinholes
        """
        
        if self.pinhole_centers is None:
            raise ValueError("pinhole_centers not defined. Individual "
                             "apertures need to be fit first.")
        
        if width is None:
            width = 1.3*(self.mag_s*self.diameter.to(u.cm).value)
            
        # Calculate the half-width in pixels
        dx = np.mean(np.gradient(xaxis))
        w = int(width/2/dx)
        
        # Indices of the apetures to include
        use_ind = [i for i,v in enumerate(self.use) if v==1]
        
        output = np.zeros([2*w, 2*w])
        
        # For each aperture, select the data region around each aperture
        # and sum them
        for i, ind in enumerate(use_ind):
            x0 = np.argmin(np.abs(xaxis - (self.pinhole_centers[i,0])))
            y0 = np.argmin(np.abs(yaxis - (self.pinhole_centers[i,1])))
            
            data_subset = data[x0-w:x0+w, y0-w:y0+w]

            output += data_subset
                
            
        # Calculate new x and y axes centered on this image
        xaxis = np.linspace(-width/2, width/2, 2*w)
        yaxis = np.linspace(-width/2, width/2, 2*w)
        
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
        
        ax.set_xlim(np.min(xaxis), np.max(xaxis))
        ax.set_ylim(np.min(yaxis), np.max(yaxis))
        

        if self.xy is not None:
            ax.scatter(self.xy[self.use,0], 
                       self.xy[self.use,1], color='green')
            
            ax.scatter(self.xy[~self.use,0], 
                       self.xy[~self.use,1], color='red')
            
            
        if self.pinhole_centers is not None:
            ax.scatter(self.pinhole_centers[:,0], 
                       self.pinhole_centers[:,1],  color='black',
                       marker='x')
            
            radius = self.diameter.to(u.cm).value/2
            for i in range(self.pinhole_centers.shape[0]):
                circle = plt.Circle((self.pinhole_centers[i,0], 
                                     self.pinhole_centers[i,1]),  
                                    radius*self.mag_r, color='black',
                                    fill=False, linestyle='dashed')
                ax.add_patch(circle)
                
        
        
            
        plt.show()
            
        return  fig, ax
            
            
    def plot_array(self):
        fig, ax = plt.subplots()
        
        ax.set_aspect('equal')
        ax.scatter(self.xy[:,0], self.xy[:,1])
        ax.set_title(self.name)
        
        plt.show()