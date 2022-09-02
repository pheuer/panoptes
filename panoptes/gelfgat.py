import numpy as np

def gelfgat_poisson(P, T, niter, h_friction=3):
    """
    Parameters
    ----------
    P : ndarray
        Data as a 1D vector length J*K
    T : np.ndarray
        Transfer matrix as a 2D vector, with shape [M*N, J*K]
        or [Image, Object]
    niter : int
        Number of interations to calculate
        
    h_friction: float
        Amount to divide the step size by. If h_friction=1, the solution
        will oscillate wildly and the logL can decrease.
        
    """
    ishape = T.shape[0]
    oshape = T.shape[1]

    # Initial guess is uniform
    Bnew = np.ones(oshape)
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
    DOFs = np.zeros(niter)
    
    zeno_step = 2
    

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
        
        DOFs[i] = np.sum(B[i, :] / (B[i, :] + 1/B[i, :].size))

        
    return B, logL, chisq, DOFs