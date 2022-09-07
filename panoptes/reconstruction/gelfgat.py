import numpy as np

import h5py
import tqdm

from scipy.stats import chi2


from cached_property import cached_property

def gelfgat_poisson(P, tmat, niter, h_friction=3):
    """
    Parameters
    ----------
    P : ndarray
        Data as a 1D vector length J*K
        
    T : TransferMatrix object
        The transfer matrix
        
    niter : int
        Number of interations to calculate
        
    h_friction: float
        Amount to divide the step size by. If h_friction=1, the solution
        will oscillate wildly and the logL can decrease.
        
    """

    print("Loading transfer matrix")
    T = tmat.tmat
    
    print("Performing reconstruction")


    # Initial guess is uniform
    # Add one pixel to the object plane for the background pixel
    Bnew = np.ones(tmat.osize+1)
    Bnew = Bnew/np.sum(Bnew) # Normalize Bnew

    # Normalize the transfer matrix to the image plane, not including any
    # zero values
    nonzero = np.nonzero(np.matmul(T, Bnew))[0]
    Tnorm = T[nonzero, :] / np.sum(T[nonzero, :], axis=0)
    
    # Normalized copy of the data
    Pnorm = P[nonzero]/np.sum(P[nonzero])
    
    B = np.zeros([niter+1, Bnew.size])
    B[0, :] = Bnew # Set first one to Bnew
    h = np.zeros(niter)
    logL = np.zeros(niter)
    chisq = np.zeros(niter)
    
    zeno_step = 2
    
    with tqdm.tqdm(total=niter) as pbar:
        for i in range(niter):
            
            Pguess = np.matmul(Tnorm, B[i,:])
            
            Bdelta = B[i,:] * np.matmul(Pnorm/Pguess -1, Tnorm)
            
            
            # Calculate the step size
            deltaPguess = np.matmul(Tnorm, Bdelta)
            dLdh = np.sum(Bdelta**2 / B[i,:])
            d2Ld2h = np.sum(Pnorm*deltaPguess**2 / Pguess**2)
            h[i] = dLdh/d2Ld2h
            
            # Apply the changes
            B[i+1,:] = B[i,:] + h[i]/h_friction*Bdelta
            
            
            # Apply zenos pixel approach to prevent B from going negative
            if zeno_step > 0:
                B[i+1,:] = np.where(B[i+1,:] <= 0, B[i, :]/zeno_step, B[i+1,:])
                
            # Renormalize on each iteration
            B[i+1, :] *=  1/np.sum(B[i+1, :])
            
            # Compute the current log likelihood 
            
            logL[i] = np.sum(P)*np.sum(Pnorm*np.log(Pguess))
            
            chisq[i] = np.sum(P)*np.sum( (Pguess - Pnorm)**2/Pguess)
            
            pbar.update()
            
            
    # Remove the initial guess, which is stored as the first iteration
    # Remove the backround pixel from this image
    initial_guess = B[0, :-1]
    B = B[1:, :]
    
    # Remove the background pixel 
    background = B[:, -1]
    B = B[:, :-1]
   
    # Reshape B, initial_guess to be 2D in the object plane
    B  = np.reshape(B.T, (*tmat.oshape, niter))
    initial_guess = np.reshape(initial_guess, tmat.oshape)
    
    return GelfgatResult(B, logL, chisq, background, initial_guess)






class GelfgatResult:
    
    def __init__(self, *args):
        """
        The results of a Gelfgat reconstruction

        Parameters (1)
        --------------
        
        path : str
            Path to a reconstruction save file 
            
            
            
        Paramters (5)
        -------------
        
        B : np.ndarray (nxo, nyo, iter)
            Results from each step of the reconstruction
            
        logL : np.ndarray (niter)
            The log likelihood at each step of the reconstruction.
        
        chisq : np.ndarray (niter)
            The chi squared at each step of the reconstruction.
            
            
        background : np.ndarray (niter)
            The value of the background pixel at each step of the 
            reconstruction. The background pixel is a constant value added
            to every pixel in the image plane.
            
            
        initial_guess : np.ndarray (nxo, nyo)
            The initial guess for the object. Usually uniform.
        
        """
        
        # These variables record a string that describes the algorithm used
        # to calculate the DOF and termination condition
        self.termination_algorithm = "None"
        self.DOF_algorithm = "None"
        
        
        if len(args) == 1:
            self.load(args[0])
            
        elif len(args) == 5:
            (self.B, self.logL, self.chisq, 
             self.background, self.initial_guess) = args
            
        else:
            raise ValueError(f"Invalid number of arguments: {len(args)}")
            
            
    def save(self, path):
        """
        Saves the reconstruction object to an h5 file.

        """
        
        with h5py.File(path, 'a') as f:
            f['B'] = self.B
            f['logL'] = self.logL
            f['chisq'] = self.chisq
            f['background'] = self.background
            f['initial_guess'] = self.initial_guess

            f['DOF'] = self.DOF
            f['DOF'].attrs['algorithm'] = self.DOF_algorithm
            f['termination_condition'] = self.termination_condition
            f['termination_condition'].attrs['algorithm'] = self.termination_algorithm
            f['termination_ind'] = self.termination_ind     
            
            
    def load(self, path):
        """
        Loads a reconstruction object from an h5 file. 
        
        Properties are not loaded - they will be re-calculated when called
        """
        
        with h5py.File(path, 'r') as f:
            self.B = f['B'][...]
            self.logL = f['logL'][...]
            self.chisq = f['chisq'][...]
            self.background = f['background'][...]
            self.initial_guess = f['initial_guess'][...]
            
            
    @cached_property
    def solution(self):
        """
        The reconstruction at the termination index
        """
        return self.B[:, :, self.termination_ind]
            
            
    @cached_property
    def DOF(self):
        """
        The estimated degrees of freedom at each step of the reconstruction

        """
        self.DOF_algorithm = 'sum(img / (img + 1/img.size) )'
        
        niter = self.B.shape[0]
        
        # Estimate the DOF
        DOF = np.zeros(niter) 
        for i in range(niter):
            img = self.B[i,:] / np.sum(self.B[i,:])
            DOF[i] = np.sum(img / (img + 1/img.size))
        return DOF
    
    @cached_property
    def DOF_asymptotic(self):
        """
        The DOF based on the last step of the reconstruction.
        """
        img = self.B[-1,:] / np.sum(self.B[-1,:])
        return np.sum(img / (img + 1/img.size))
        
        
            
    @cached_property
    def termination_condition(self):
        """
        The termination condition at each step of the reconstruction.
        """
        self.termination_algorithm = "chi2.cdf(likelihood ratio, DOF_asymptotic)"
        
        # Likelihood ratio at each iteration
        likelihood_ratio = -2*(self.logL - self.logL[-1])
        
        termination_condition = chi2.cdf(likelihood_ratio, self.DOF_asymptotic)
        
        return termination_condition
        
    @cached_property
    def termination_ind(self):
        """
        Calculates the termination index
        """
        
        return np.argmin(np.abs(self.termination_condition - 0.5))
        


    
