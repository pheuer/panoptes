# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 11:54:35 2022

@author: pvheu
"""

import os
import h5py
import matplotlib.pyplot as plt
import numpy as np

from panoptes import _root_dir

savepath_1d_efield_semianalytic = os.path.join(_root_dir, 'tmat', '1d_efield_semianaytic.h5')

def make_1d_efield_semianalytic():
    """
    Assess PSF using numerical differential:
    The fluence that goes through the radial bin d(ra) in the aperture plane 
    is deposited over the radias bin d(rd) in the detector plane.  We have
    the following relation to connect them:

    rd = ra*M*(1 + 0.5*D_det*(M-1)/ra/M^2*E(ra)*dZ/E_d)
    
    lets use the scaling:
       ra_scaled = ra/R_ap;
       rd_scaled = rd/(R_ap*M) = ra_scaled*(1 + 0.5*D_det*(M-1)/M^2/R_ap/ra_scaled*e*E(ra)*dZ/E_d);
    
     The normalized point-spread function PSF is then:
     PSF = (drd_scaled/dra_scaled)^-1
         = [1 + (0.5*(D_det/R_ap)*(M-1)/M^2*dZ/E_d)*e*dE(ra_scaled)/dra_scaled)]^(-1);
    
     Here, E(ra_scaled) = E(a)*(alpha/(2*pi*epsilon_0)) for E(a) calculated
     above.  The (alpha) coefficient is in coulombs/cm^2 (we call it Q
     elsewhere).  Total charge is: Q_tot = alpha*pi*R_ap^2
    
     So to get the PSF we just need the derivative dE/da:
    
     Limiting values:  
        a-->0,  dEda --> pi/2
        a-->1,  dEda --> ??
        
    
    """
    print(f"Creating 1d efield semianalytic psf")
    
    da = 0.0001
    a = np.arange(0, 2, da)
    db = 0.0001
    b = np.arange(-1, 1, db)
    inta = np.zeros([a.size, b.size])
    intb = np.zeros([a.size, b.size])
    
    
    # Numerical integration ?
    # Calculates E-field...
    for i in range(a.size):
        reg1 = (b < (2*a[i]-1))
        inta[i,reg1] = np.sqrt(1-b[reg1]**2)/(a[i]-b[reg1])/np.sqrt(1 - b[reg1]**2 + (a[i]-b[reg1])**2)
        
        thing1 = ( (np.sqrt(1-b**2)/np.sqrt((a[i]-b)**2 + 1 - b**2) -
		np.sqrt(1 - (2*a[i]-b)**2)/np.sqrt((a[i]-b)**2 + 1 - (2*a[i]-b)**2))/
		(a[i]-b))
        reg2 = (b >= (2*a[i]-1))*(b < a[i])
        intb[i,reg2] = thing1[reg2]
        
    E = np.sum(inta,1)*db + np.sum(intb,1)*db
    
    
    
    fig, ax = plt.subplots()
    ax.plot(a, E, color='black', linewidth=0.5)
    ax.plot(a, np.sum(inta,1)*db, color='red', linewidth=0.5)
    ax.plot(a, np.sum(intb,1)*db, color='blue', linewidth=0.5)
    ax.set_xlim(1e-4, 1)
    ax.set_ylim(1e-6, 10)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("radius (r/R)")
    ax.set_ylabel("E/(\alpha / 2\pi \epsilon_0")
    ax.set_title("Electric field vs. radius on a uniformly charged disk")
    
    
    dEda = np.concatenate( (np.array([np.pi/2,]),
                            (E[2:] - E[:-2])/(2*da),
                            np.array( [(E[-2] - E[-1])/da,]) )
                          )
    
    
    fig, ax = plt.subplots()
    ax.plot(a, dEda, color='blue', linewidth=0.5)
    ax.set_xlim(1e-4, 1)
    ax.set_ylim(1, 1e4)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title("dEda")
    ax.set_xlabel("radius (r/R)")
    
    ax.plot([0.001, 1], np.array([1,1])*np.pi/2, color ='red', linewidth=0.5, 
            linestyle='--')
    ax.plot(np.arange(1, np.max(a)), np.pi*np.arange(1, np.max(a))**-3, 
            color='red', linewidth=0.5, linestyle='--')

    
    with h5py.File(savepath_1d_efield_semianalytic, 'w') as f:
        f['a'] = a
        f['E'] = E
        f['dEda'] = dEda
    
    
    
    
    
    
                 

def calc_1d_efield_semianalytic_psf(V, dz=0.02, # cm
                            L_det=350, # cm
                            R_ap=0.0150, #cm
                            ):
    
    V = np.array(V)
    
    if not os.path.isfile(savepath_1d_efield_semianalytic):
        make_1d_efield_semianalytic()
        
    with h5py.File(savepath_1d_efield_semianalytic, 'r') as f:
        a = f['a'][:]
        E = f['E'][:]
        dEda = f['dEda'][:]
    
    consts = 898755.179*dz
    region = a < 1 # Region inside the aperture, e.g. a=r/R_ap < 1
    
    PSF = np.zeros([V.size, dEda.size])
    
    # V = (D_det/R_ap)*Q(M-1)/M^2/E_d
    PSF = (1 + consts*V[:, np.newaxis]*dEda)**-1
    
    # Q in Cou/cm^2, E_d in MeV.
    #rd = lambda V :  a[region] + consts*V*E[region]
    
    return a,V, PSF
    
    
    
        
        
if __name__ == '__main__':
    
    
    V = np.linspace(-8, -3, num=500)
    V = 10**V
    
    a, V, PSF = calc_1d_efield_semianalytic_psf(V)
    
    print(a)
    
    
    fig, ax = plt.subplots()
    ax.pcolormesh(a, V, PSF)
    ax.set_yscale('log')
    
    
    
    
    
    
    
    