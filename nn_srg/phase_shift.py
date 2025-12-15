# Copyright (c) 2025 Matthias Heinz
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# This code is adapted from code written by Yannick Dietz and Kai Hebeler
# with permission of the original authors.

import numpy as np
from .constants import hbarc, nucleon_mass
from .potential import Potential

M = nucleon_mass
eps = 1e-3

def compute_phase_shifts_single_channel(pot: Potential):
    Nrows = len(pot.nodes)

    K = compute_K_matrix(pot.without_weights(), Nrows, np.max(pot.nodes) + 1.0, pot.nodes, pot.weights)

    ps = [compute_phase_shift_single_point(K, i, pot.nodes) for i in range(len(pot.nodes))]
    Es = [Elab(p) for p in pot.nodes]

    return np.array(Es), np.array(ps)


#computation of phase shifts in uncoupled channels
def counterterm(pmax, pole):
    return np.arctanh(pole/pmax)/pole

def compute_phase_shift_single_point(K,i, mesh_points):
    return 180.0/np.pi*np.arctan(-mesh_points[i]*K[i,i])

def delta(i,j):
    if (i==j):
        return 1
    else:
        return 0

def Elab(p):
    return 2*p**2*hbarc**2/M

def mom(E):
    return np.sqrt(M*E/2/hbarc**2)
    
def compute_K_matrix(V, Nrows, pmax, mesh_points, mesh_weights):  
    A = np.zeros([Nrows+1,Nrows+1],float)
    K = np.zeros([Nrows,Nrows],float)
    
    for x in range(Nrows):
        pole = mesh_points[x] + eps
        
        for i in range(Nrows):
            for j in range(Nrows):
                A[i,j] = delta(i,j) - 2.0/np.pi * mesh_weights[j]* V[i,j] * mesh_points[j]**2 / (pole**2 - mesh_points[j]**2)
        
        sum = 0.0
        for i in range(Nrows):
            sum += mesh_weights[i]/(pole**2 - mesh_points[i]**2)

        for i in range(Nrows):
            A[Nrows,i] = - 2.0/np.pi * mesh_weights[i] * mesh_points[i]**2 * V[x,i]/(pole**2 - mesh_points[i]**2)
            A[i,Nrows] = + 2.0/np.pi * V[i,x] * pole**2 * (sum - counterterm(pmax,pole))

        A[Nrows,Nrows] = 1 + 2.0/np.pi * V[x,x] * pole**2 * (sum - counterterm(pmax,pole))

        bvec = np.zeros([Nrows+1],float)
        for i in range(Nrows):
            bvec[i] = V[i,x]
        bvec[Nrows] = V[x,x]
        
        xvec = np.linalg.solve(A, bvec)

        for i in range(Nrows):
            K[i,x] = xvec[i]

    return K
