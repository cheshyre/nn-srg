# Copyright (c) 2025 Matthias Heinz
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
"""Hamiltonian diagonalization.

This module provides utilities to diagonalize momentum-space Hamiltonians
and extract bound state energies for nuclear two-body systems.

The Hamiltonian is constructed from kinetic and potential energy matrices
in momentum space, with proper conversion from natural units (fm⁻¹) to
physical units (MeV).

Functions
---------
diagonalize_hamiltonian(pot, num_eigvals=1)
    Diagonalize Hamiltonian and return lowest eigenvalues.

"""
import numpy as np

from .constants import hbarc, red_mass

# Conversion factor to MeV
convert_to_mev: float = hbarc**2 / (2 * red_mass)  


def diagonalize_hamiltonian(pot, num_eigvals: int = 1) -> np.ndarray:
    """Diagonalize Hamiltonian and return lowest eigenvalues.
    
    Constructs the Hamiltonian H = T + V in momentum space with proper
    integration weights, then diagonalizes to extract bound state energies.
    
    Parameters
    ----------
    pot : Potential or CoupledPotential
        Potential object containing the interaction and kinetic energy.
        Must have methods `with_weights()` and `kinetic_energy()`.
    num_eigvals : int, optional
        Number of lowest eigenvalues to return. Default is 1.
        
    Returns
    -------
    ndarray
        Array of lowest eigenvalues in MeV, sorted in ascending order.
        For bound states, eigenvalues are negative.
        
    Notes
    -----
    The Hamiltonian is constructed in momentum space as:
        H = T + V
    where T is the kinetic energy matrix and V is the potential matrix,
    both with Gaussian quadrature weights included.
    
    The kinetic energy is T = p²/(2μ), where p is momentum and μ is the
    reduced mass. The conversion factor (ℏc)²/(2μ) converts from natural
    units to MeV.
    
    Examples
    --------
    >>> from potential import load_3S1_3D1_potential
    >>> pot = load_3S1_3D1_potential('AV18')
    >>> ground_state_energy = diagonalize_hamiltonian(pot, num_eigvals=1)
    >>> print(f"Deuteron binding energy: {-ground_state_energy[0]:.3f} MeV")
    Deuteron binding energy: 2.224 MeV
    
    >>> # Get first 5 eigenvalues (ground + excited states)
    >>> energies = diagonalize_hamiltonian(pot, num_eigvals=5)
    
    """
    # Construct Hamiltonian in momentum space (with integration weights)
    hamiltonian = convert_to_mev * (pot.with_weights() + pot.kinetic_energy())
    
    # Diagonalize 
    evals, _ = np.linalg.eigh(hamiltonian)
    
    return evals[:num_eigvals]
