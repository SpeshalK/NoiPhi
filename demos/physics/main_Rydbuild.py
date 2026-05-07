#!/bin/python3
import sys
sys.dont_write_bytecode = True
import scipy.sparse as sp
from scipy.sparse import csr_matrix as csr
import numpy as np

from scipy.sparse import tril, triu
from scipy.sparse import csr_matrix as csr

def buildRydHamil(N,C,λ=0.0,periodic=False,nnI=False,nnI2=False):
    """
    Constructs the sparse matrix components of a 1D Rydberg atom chain.

    Generates the interaction terms based on the C/r^{3,6} (customizable C) 
    Van der Waals or dipol-dipole interaction and local field operators.

    Parameters
    ----------
    N : int
        Number of atoms in the chain.
    C : float
        Interaction strength exponent (e.g., 6 for Van der Waals).
    lambda_val : float, optional
        Strength of the XXZ-type exchange term (sigX*sigX + sigY*sigY).
    periodic : bool, optional
        Whether to apply Periodic Boundary Conditions (PBC).

    Returns
    -------
    A list of matrix components that can be scaled by Hamiltonian parameters:
    tuple (csr_matrix, csr_matrix, csr_matrix, csr_matrix)
        Returns (Hx_upper, Hx_lower, H_detuning, H_interaction).
    """

    # Spin Hamiltonian of dimensions 2^N
    sHxsum=csr((2**N,2**N),dtype=np.float64)
    sHn_ksum=csr((2**N,2**N),dtype=np.float64)
    sHint=csr((2**N,2**N),dtype=np.float64)

    # Generating Schachenmayer Hamiltonian (eq 12) in terms of n_k operators.
    for k in range(N):
        n_k=opMaker(N,k,5)
        sigX_k=opMaker(N,k,1)
        sigY_k=opMaker(N,k,2)

        #First and second terms in Hamiltonian
        sHxsum=sHxsum+sigX_k
        sHn_ksum=sHn_ksum+n_k
            

        # Third term in Hamiltonian (bulk interactions)
        for m in range(N):
            if m==k or m<k:
                continue
            elif nnI and np.abs(m-k)>1:
                continue
            elif nnI2 and np.abs(m-k)>2:
                continue
            else:
                n_m=opMaker(N,m,5)

            if nnI2:
                corr=2**(np.abs(k-m)-1)
            else:
                if periodic:
                    #PERIODIC BOUNDARIES (if |k-m|>N/2 then corr should get smaller)
                    if np.abs(k-m) > int(N/2):
                        corr=(N-np.abs(k-m)+1)**C
                    else:
                        corr=np.abs(k-m)**C
                else:
                    #OPEN BOUNADRIES
                    corr=np.abs(k-m)**C

            if λ!=0.0:
                sigX_m=opMaker(N,m,1)
                sigY_m=opMaker(N,m,2)
                sHxxz=csr.multiply(sigX_k*sigX_m+sigY_k*sigY_m,λ)
                del sigX_m
                del sigY_m

                sHzm=n_k*n_m + sHxxz
            else:
                sHzm=n_k*n_m

            sHzm=csr.multiply(sHzm,1/corr)
            sHint=sHint+sHzm
            del n_m
        del n_k
        del sigX_k
        del sigY_k

            
        # If periodic boundaries 
        # add edge interaction.
        if periodic:
            n_0=opMaker(N,0,5)
            n_N=opMaker(N,N-1,5)

            if k < int(N/2):
                corr_edge=(k+1)**C
            else:
                corr_edge=(N-k)**C

            sHzm_edge=n_0*n_N
            sHzm_edge=csr.multiply(sHzm_edge,1/corr_edge)
            sHint=np.add(sHint,sHzm_edge)

    sHx_upper=csr(triu(sHxsum))
    sHx_lower=csr(tril(sHxsum))

    return (sHx_upper,sHx_lower,sHn_ksum,sHint)

def opMaker(N,s,opindex):
    """Build a single-site operator on an N-site lattice.

    Parameters
    ----------
    N : int
        Number of lattice sites.
    s : int
        Site index to apply the operator on.
    opindex : int
        Operator choice: 0=Identity, 1=sigX, 2=sigY,
        3=sigZ, 4=n_ground, 5=n_excited.

    Returns
    -------
    csr_matrix
        Sparse operator of dimension 2^N x 2^N.
    """

    #Identity matrix                         # opindex=0
    sigx=csr(np.array([[0,1],[1,0]]),dtype=complex)       # opindex=1
    sigy=csr(np.array([[0,-1j],[1j,0]]),dtype=complex)    # opindex=2
    sigz=csr(np.array([[1,0],[0,-1]]),dtype=complex)      # opindex=3
    nz_g=csr(-(sigz-np.eye(2))/2)            # opindex=4
    nz_e=csr((sigz+np.eye(2))/2)             # opindex=5

    # Array of basis operators.
    sig=(csr(np.eye(2)),sigx,sigy,sigz,nz_g,nz_e)

    if N==1:
        op=sig[opindex]
    else:
        index=np.zeros(N,dtype=int)
        index[s]=1

        # loop over basis operators using index[] 
        # which has only one nonzero element (index[l]).
        # Build full lattice operator using kron (csr).
        op=sp.kron(sig[index[0]*opindex],sig[index[1]*opindex],format='csr')
        for i in range(2,N):
            op=sp.kron(op,sig[index[i]*opindex],format='csr')
    
    return op
