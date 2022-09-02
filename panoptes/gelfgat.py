import numpy as np

import h5py
from multiprocessing import Pool
import tqdm
"""
def gelfgat_poisson(P, tmat_file, niter, h_friction=3):
    
    
    with h5py.File(tmat_file, 'a') as f:
        
        xo = f['xo'][...]
        yo = f['yo'][...]
        xi = f['xi'][...]
        yi = f['yi'][...]
        
        nxo, nyo = xo.size, yo.size
        nxi, nyi = xi.size, yi.size
        
    # Calculate chunk size, chunking the object plane together
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
            
        print(f"{a}, {b}")
        
        chunks.append( [P[a:b], 
                        tmat_file, 
                        niter, h_friction,
                        a, b] )

    solution = np.zeros([nxo*nyo, niter+1])
    with Pool() as p:
        
        # Initialize a tqdm progress bar
        with tqdm.tqdm(total=len(chunks)) as pbar:
            
            # Loop through the mapped calculations on the parallel pool
            for i, result in enumerate(p.imap(_gelfgat_poisson, chunks)):
                
                # Store the result
                a = chunks[i][-2]
                b = chunks[i][-1]
                solution[ a:b, :] = result
                
                # Update the progress bar that this iteration is done
                pbar.update()
                
    print(solution.shape)
    
    return solution
""" 


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
            
            DOFs[i] = np.sum(B[i, :] / (B[i, :] + 1/B[i, :].size))
            
            pbar.update()

        
    return B, logL, chisq, DOFs