# Copyright (c) 2025 Matthias Heinz
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# This code is adapted from code written by Yannick Dietz and Kai Hebeler
# with permission of the original authors.

"""Phase shift computation module for nucleon-nucleon scattering.

This module provides functions to compute nuclear scattering phase shifts from
nucleon-nucleon potentials using the K-matrix formalism. It handles both
uncoupled (single channel) and coupled channel cases.

The phase shifts are computed by solving the Lippmann-Schwinger equation in
momentum space to obtain the K-matrix, from which phase shifts are extracted.
For coupled channels, the mixing angle (bar-epsilon) is also computed.

Functions
-------------
compute_phase_shifts_single_channel(pot)
    Compute phase shifts for an uncoupled partial wave channel.

compute_phase_shifts_coupled_channel(pot)
    Compute phase shifts and mixing angle for a coupled channel system.

convert_p_to_Elab(p)
    Convert momentum to laboratory-frame kinetic energy.

convert_Elab_to_p(E)
    Convert laboratory-frame kinetic energy to momentum.

Notes
-----
The phase shifts are computed in degrees. The module uses a regularization
scheme with a momentum cutoff pmax and a small epsilon parameter to handle
singularities in the Lippmann-Schwinger equation.

References
----------
The K-matrix approach and coupled channel formalism are standard in nuclear
scattering theory. See, for example:
- Taylor, J. R. "Scattering Theory" (Wiley, 1972)
- Newton, R. G. "Scattering Theory of Waves and Particles" (Springer, 1982)

"""
import numpy as np
from typing import Tuple, Union
from .constants import hbarc, nucleon_mass
from .potential import Potential, CoupledPotential

# Nucleon mass constant
M = nucleon_mass

# Small epsilon for pole regularization in K-matrix computation
eps = 1e-3


def compute_phase_shifts_single_channel(pot: Potential) -> Tuple[np.ndarray, np.ndarray]:
    """Compute phase shifts for a single (uncoupled) partial wave channel.

    Solves the Lippmann-Schwinger equation to obtain the K-matrix, then
    extracts phase shifts as a function of energy. Phase shifts are returned
    in degrees with discontinuities removed.

    Parameters
    ----------
    pot : Potential
        Potential object containing the nuclear interaction for a single
        partial wave channel. Must include momentum grid (nodes) and
        integration weights.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - Es : np.ndarray
            Laboratory-frame kinetic energies (MeV) corresponding to each
            momentum grid point.
        - ps : np.ndarray
            Phase shifts (degrees) as a function of energy. Discontinuities
            have been removed to ensure smooth behavior.

    Notes
    -----
    The K-matrix is computed using the method of ref. [Newton, 1982], with
    regularization to handle the singularities in the principal value integral.
    Phase shifts are related to K-matrix elements by: δ(p) = arctan(-p * K(p,p)).

    """
    Nrows = len(pot.nodes)

    # Compute the K-matrix by solving the Lippmann-Schwinger equation
    K = _compute_K_matrix(
        pot.without_weights(), Nrows, np.max(pot.nodes) + 1.0, pot.nodes, pot.weights
    )

    # Extract phase shifts from diagonal K-matrix elements
    ps = [
        _compute_phase_shift_single_point(K, i, pot.nodes) for i in range(len(pot.nodes))
    ]
    
    # Convert momentum grid to laboratory energies
    Es = [convert_p_to_Elab(p) for p in pot.nodes]

    # Remove discontinuities and enforce boundary conditions
    ps = _fix_boundary_conditions_and_discontinuities(ps)

    return np.array(Es), np.array(ps)


def compute_phase_shifts_coupled_channel(pot: CoupledPotential) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute phase shifts and mixing angle for a coupled channel system.

    For coupled channels (e.g., ³S₁-³D₁), computes the two eigenphase shifts
    and the mixing angle (bar-epsilon) by solving the coupled Lippmann-Schwinger
    equations and diagonalizing the resulting K-matrix.

    Parameters
    ----------
    pot : CoupledPotential
        Coupled potential object containing the four blocks of the interaction
        matrix: V₀₀, V₀₁, V₁₀, V₁₁ for the two coupled channels.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing:
        - Es : np.ndarray
            Laboratory-frame kinetic energies (MeV).
        - ps0 : np.ndarray
            First eigenphase shift (degrees) as a function of energy.
        - ps1 : np.ndarray
            Second eigenphase shift (degrees) as a function of energy.
        - angle : np.ndarray
            Mixing angle bar-epsilon (degrees) as a function of energy.
            This quantifies the mixing between the two channels.

    Notes
    -----
    The coupled channel K-matrix is computed as a 2N x 2N matrix where N is
    the dimension of each individual channel. The eigenphase shifts are
    obtained by diagonalizing the 2x2 K-matrix at each energy.

    For the ³S₁-³D₁ system, the mixing angle arises from tensor force
    contributions that couple S-wave and D-wave components.

    """
    # Extract the four blocks of the coupled potential matrix
    V00 = pot.extract_channel_potential(pot._channels[0])
    V01 = pot.extract_channel_potential(pot._channels[1])
    V10 = pot.extract_channel_potential(pot._channels[2])
    V11 = pot.extract_channel_potential(pot._channels[3])

    Nrows = len(V00.nodes)

    # Compute the coupled K-matrix
    K = _compute_K_matrix_coupled(
        V00.without_weights(),
        V01.without_weights(),
        V10.without_weights(),
        V11.without_weights(),
        Nrows,
        np.max(pot.nodes) + 1.0,
        pot.nodes,
        pot.weights,
    )

    # Extract eigenphase shifts and mixing angle at each energy
    ps = [
        _compute_phase_shifts_coupled_single_point(K, i, pot.nodes, Nrows)
        for i in range(Nrows)
    ]
    ps0 = np.array([x[0] for x in ps])
    ps1 = np.array([x[1] for x in ps])
    angle = [x[2] for x in ps]
    Es = [convert_p_to_Elab(p) for p in pot.nodes[:Nrows]]

    # Remove discontinuities from both phase shifts
    ps0 = _fix_boundary_conditions_and_discontinuities(ps0)
    ps1 = _fix_boundary_conditions_and_discontinuities(ps1)

    return np.array(Es), np.array(ps0), np.array(ps1), np.array(angle)


def convert_p_to_Elab(p: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert momentum to laboratory-frame kinetic energy.

    For nucleon-nucleon scattering in the center-of-mass frame, converts
    the relative momentum p to the equivalent laboratory kinetic energy.

    Parameters
    ----------
    p : Union[float, np.ndarray]
        Momentum in the center-of-mass frame (fm⁻¹).

    Returns
    -------
    Union[float, np.ndarray]
        Laboratory-frame kinetic energy (MeV).

    Notes
    -----
    Uses the non-relativistic formula: E_lab = 2p²ℏ²c²/M
    where M is the nucleon mass.

    """
    return 2 * p**2 * hbarc**2 / M


def convert_Elab_to_p(E: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert laboratory-frame kinetic energy to momentum.

    Inverse of convert_p_to_Elab. Converts laboratory kinetic energy to
    center-of-mass relative momentum for nucleon-nucleon scattering.

    Parameters
    ----------
    E : Union[float, np.ndarray]
        Laboratory-frame kinetic energy (MeV).

    Returns
    -------
    Union[float, np.ndarray]
        Momentum in the center-of-mass frame (fm⁻¹).

    Notes
    -----
    Uses the non-relativistic formula: p = sqrt(M * E / (2ℏ²c²))
    where M is the nucleon mass.

    """
    return np.sqrt(M * E / 2 / hbarc**2)


# ------------ Internal methods ---------------- #


def _fix_large_E(vals: np.ndarray) -> np.ndarray:
    """Fix phase shifts at high energy to lie within [-30°, 30°].

    Adjusts phase shifts by multiples of 180° to ensure the high-energy
    (last) phase shift value lies within a reasonable range, as phase
    shifts typically become small at high energies.

    Parameters
    ----------
    vals : np.ndarray
        Array of phase shift values (degrees).

    Returns
    -------
    np.ndarray
        Adjusted phase shift array with high-energy value in range.

    """
    large_E_val = vals[-1]
    while large_E_val > 30 or large_E_val < -30:
        shift = 0
        if large_E_val > 30:
            shift = -180
        else:
            shift = 180
        vals += shift
        large_E_val = vals[-1]
    return vals


def _has_discont(vals: np.ndarray) -> bool:
    """Check if phase shift array contains discontinuities.

    Scans the array for jumps larger than 90°, which indicate artificial
    discontinuities from the arctangent function having a restricted range.

    Parameters
    ----------
    vals : np.ndarray
        Array of phase shift values (degrees).

    Returns
    -------
    bool
        True if a discontinuity is detected, False otherwise.

    """
    for i in reversed(range(1, len(vals))):
        if np.abs(vals[i] - vals[i - 1]) > 90:
            return True
    return False


def _fix_one_disc(vals: np.ndarray) -> np.ndarray:
    """Remove one discontinuity from phase shift array.

    Finds the first (scanning backwards from high energy) discontinuity
    and removes it by shifting all lower-energy phase shifts by ±180°.

    Parameters
    ----------
    vals : np.ndarray
        Array of phase shift values (degrees).

    Returns
    -------
    np.ndarray
        Phase shift array with one discontinuity removed.

    Notes
    -----
    This function should be called iteratively until all discontinuities
    are removed, as fixing one discontinuity may reveal others.

    """
    for i in reversed(range(1, len(vals))):
        if np.abs(vals[i] - vals[i - 1]) > 90:
            if vals[i - 1] < vals[i]:
                shift = 180
            else:
                shift = -180
            for j in range(i - 1, -1, -1):
                vals[j] += shift
            return vals
    return vals


def _fix_boundary_conditions_and_discontinuities(vals: np.ndarray) -> np.ndarray:
    """Remove all discontinuities and enforce boundary conditions.

    Processes phase shift array to ensure smooth behavior by:
    1. Adjusting high-energy values to reasonable range
    2. Iteratively removing discontinuities from arctangent branch cuts

    Parameters
    ----------
    vals : np.ndarray
        Raw phase shift values (degrees) from K-matrix.

    Returns
    -------
    np.ndarray
        Smoothed phase shift array without discontinuities.

    Notes
    -----
    Phase shifts should vary smoothly with energy. Discontinuities arise
    from the limited range of arctangent and don't represent physical
    behavior. This function restores the physical smooth dependence.

    """
    vals = _fix_large_E(vals)

    while _has_discont(vals):
        vals = _fix_one_disc(vals)

    return vals


def _counterterm(pmax: float, pole: float) -> float:
    """Compute regularization counterterm for K-matrix calculation.

    Evaluates the counterterm needed to regularize the principal value
    integral in the Lippmann-Schwinger equation at the pole position.

    Parameters
    ----------
    pmax : float
        Maximum momentum cutoff (fm⁻¹) for the regularization scheme.
    pole : float
        Pole position in momentum space (fm⁻¹), typically on-shell
        momentum plus small epsilon.

    Returns
    -------
    float
        Counterterm value: arctanh(pole/pmax) / pole.

    Notes
    -----
    This counterterm subtracts the divergent part of the principal value
    integral, allowing stable numerical evaluation of the K-matrix.

    """
    return np.arctanh(pole / pmax) / pole


def _compute_phase_shift_single_point(K: np.ndarray, i: int, mesh_points: np.ndarray) -> float:
    """Compute phase shift at a single energy point from K-matrix.

    Extracts the phase shift from the diagonal element of the K-matrix
    using the relation: δ(p) = arctan(-p * K(p,p)).

    Parameters
    ----------
    K : np.ndarray
        K-matrix of dimension (N, N) where N is the number of mesh points.
    i : int
        Index of the momentum mesh point.
    mesh_points : np.ndarray
        Momentum grid points (fm⁻¹).

    Returns
    -------
    float
        Phase shift (degrees) at the energy corresponding to mesh_points[i].

    """
    return 180.0 / np.pi * np.arctan(-mesh_points[i] * K[i, i])


def _compute_phase_shifts_coupled_single_point(
    K: np.ndarray, i: int, mesh_points: np.ndarray, Nrows: int
) -> Tuple[float, float, float]:
    """Compute eigenphase shifts and mixing angle at single energy point.

    For coupled channels, diagonalizes the 2x2 K-matrix at a given energy
    to extract the two eigenphase shifts and the mixing angle.

    Parameters
    ----------
    K : np.ndarray
        Coupled K-matrix of dimension (2N, 2N) where N is the number of
        mesh points per channel.
    i : int
        Index of the momentum mesh point.
    mesh_points : np.ndarray
        Momentum grid points (fm⁻¹).
    Nrows : int
        Number of mesh points in a single channel (N).

    Returns
    -------
    Tuple[float, float, float]
        A tuple containing:
        - delta_1 : float
            First eigenphase shift (degrees).
        - delta_2 : float
            Second eigenphase shift (degrees).
        - epsilonbar : float
            Mixing angle bar-epsilon (degrees), quantifying the mixing
            between the two coupled channels.

    Notes
    -----
    The mixing angle is computed through a two-step diagonalization process:
    1. Initial rotation by epsilon to diagonalize K-matrix structure
    2. Final rotation by epsilonbar accounting for energy dependence

    """
    # Initial mixing angle epsilon from K-matrix structure
    epsilon = np.arctan(2 * K[i, i + Nrows] / (K[i, i] - K[i + Nrows, i + Nrows])) / 2.0
    r_epsilon = (K[i, i] - K[i + Nrows, i + Nrows]) / (np.cos(2 * epsilon))
    
    # Intermediate phase shifts
    delta_a = -np.arctan(
        mesh_points[i] * (K[i, i] + K[i + Nrows, i + Nrows] + r_epsilon) / 2.0
    )
    delta_b = -np.arctan(
        mesh_points[i] * (K[i, i] + K[i + Nrows, i + Nrows] - r_epsilon) / 2.0
    )

    # Final mixing angle accounting for energy dependence
    epsilonbar = np.arcsin(np.sin(2 * epsilon) * np.sin(delta_a - delta_b)) / 2.0
    
    # Eigenphase shifts in the physical basis
    delta_1 = (
        180.0
        / np.pi
        * (
            delta_a
            + delta_b
            + np.arcsin(np.tan(2 * epsilonbar) / (np.tan(2 * epsilon)))
        )
        / 2.0
    )
    delta_2 = (
        180.0
        / np.pi
        * (
            delta_a
            + delta_b
            - np.arcsin(np.tan(2 * epsilonbar) / (np.tan(2 * epsilon)))
        )
        / 2.0
    )

    # Convert mixing angle to degrees
    epsilonbar *= -180.0 / np.pi
    return (delta_1, delta_2, epsilonbar)


def _delta(i: int, j: int) -> int:
    """Kronecker delta function.

    Parameters
    ----------
    i : int
        First index.
    j : int
        Second index.

    Returns
    -------
    int
        1 if i == j, 0 otherwise.

    """
    if i == j:
        return 1
    else:
        return 0


def _compute_K_matrix(
    V: np.ndarray,
    Nrows: int,
    pmax: float,
    mesh_points: np.ndarray,
    mesh_weights: np.ndarray
) -> np.ndarray:
    """Compute K-matrix for single channel by solving Lippmann-Schwinger equation.

    Solves the momentum-space Lippmann-Schwinger equation to obtain the
    K-matrix, which is related to the scattering T-matrix and encodes the
    phase shifts. Uses regularization to handle singularities in the
    principal value integral.

    Parameters
    ----------
    V : np.ndarray
        Potential matrix in momentum space, dimension (Nrows, Nrows).
        Should be the unweighted potential V(p, p').
    Nrows : int
        Number of momentum mesh points.
    pmax : float
        Maximum momentum cutoff (fm⁻¹) for regularization.
    mesh_points : np.ndarray
        Momentum grid points (fm⁻¹).
    mesh_weights : np.ndarray
        Integration weights for momentum grid.

    Returns
    -------
    np.ndarray
        K-matrix of dimension (Nrows, Nrows). The diagonal elements K(p,p)
        directly give phase shifts via δ(p) = arctan(-p * K(p,p)).

    Notes
    -----
    The K-matrix is computed by solving a linear system at each energy:
        (I - V * G₀) K = V
    where G₀ is the free propagator with regularization. The system is
    augmented with an extra equation to handle on-shell contributions.

    """
    # Augmented matrix A and solution K
    A = np.zeros([Nrows + 1, Nrows + 1], float)
    K = np.zeros([Nrows, Nrows], float)

    # Solve for each column of K-matrix
    for x in range(Nrows):
        # Pole position (on-shell momentum plus regularization)
        pole = mesh_points[x] + eps

        # Build matrix A for the linear system (I - V*G₀)
        for i in range(Nrows):
            for j in range(Nrows):
                A[i, j] = _delta(i, j) - 2.0 / np.pi * mesh_weights[j] * V[
                    i, j
                ] * mesh_points[j] ** 2 / (pole**2 - mesh_points[j] ** 2)

        # Compute principal value sum for regularization
        sum = 0.0
        for i in range(Nrows):
            sum += mesh_weights[i] / (pole**2 - mesh_points[i] ** 2)

        # Augmented equations for on-shell elements
        for i in range(Nrows):
            A[Nrows, i] = (
                -2.0
                / np.pi
                * mesh_weights[i]
                * mesh_points[i] ** 2
                * V[x, i]
                / (pole**2 - mesh_points[i] ** 2)
            )
            A[i, Nrows] = (
                +2.0 / np.pi * V[i, x] * pole**2 * (sum - _counterterm(pmax, pole))
            )

        A[Nrows, Nrows] = 1 + 2.0 / np.pi * V[x, x] * pole**2 * (
            sum - _counterterm(pmax, pole)
        )

        # Right-hand side vector (potential column)
        bvec = np.zeros([Nrows + 1], float)
        for i in range(Nrows):
            bvec[i] = V[i, x]
        bvec[Nrows] = V[x, x]

        # Solve linear system A * K = V
        xvec = np.linalg.solve(A, bvec)

        # Extract K-matrix column
        for i in range(Nrows):
            K[i, x] = xvec[i]

    return K


def _compute_K_matrix_coupled(
    V00: np.ndarray,
    V01: np.ndarray,
    V10: np.ndarray,
    V11: np.ndarray,
    Nrows: int,
    pmax: float,
    mesh_points: np.ndarray,
    mesh_weights: np.ndarray
) -> np.ndarray:
    """Compute K-matrix for coupled channels by solving coupled Lippmann-Schwinger equations.

    Solves the coupled momentum-space Lippmann-Schwinger equations to obtain
    the 2N x 2N K-matrix for a system with two coupled channels. The K-matrix
    blocks encode both diagonal (uncoupled) and off-diagonal (mixing) scattering.

    Parameters
    ----------
    V00 : np.ndarray
        Potential matrix for channel 0 → channel 0, dimension (Nrows, Nrows).
    V01 : np.ndarray
        Potential matrix for channel 0 → channel 1, dimension (Nrows, Nrows).
    V10 : np.ndarray
        Potential matrix for channel 1 → channel 0, dimension (Nrows, Nrows).
    V11 : np.ndarray
        Potential matrix for channel 1 → channel 1, dimension (Nrows, Nrows).
    Nrows : int
        Number of momentum mesh points per channel.
    pmax : float
        Maximum momentum cutoff (fm⁻¹) for regularization.
    mesh_points : np.ndarray
        Momentum grid points (fm⁻¹).
    mesh_weights : np.ndarray
        Integration weights for momentum grid.

    Returns
    -------
    np.ndarray
        Coupled K-matrix of dimension (2*Nrows, 2*Nrows). The matrix has
        block structure:
        [ K₀₀  K₀₁ ]
        [ K₁₀  K₁₁ ]
        where each block is Nrows x Nrows.

    Notes
    -----
    The coupled K-matrix is computed by solving the linear system:
        (I - V * G₀) K = V
    where V and K are 2x2 block matrices representing the two-channel system.
    The system is augmented with extra equations (size 2N+2 x 2N+2) to handle
    on-shell contributions with regularization.

    For the ³S₁-³D₁ system: V₀₀ is S-S, V₀₁ is S-D, V₁₀ is D-S, V₁₁ is D-D.

    """
    # Augmented matrix A and solution K for coupled system
    A = np.zeros([2 * Nrows + 2, 2 * Nrows + 2], float)
    K = np.zeros([2 * Nrows, 2 * Nrows], float)

    # Solve for each momentum point
    for x in range(Nrows):
        # Pole position (on-shell momentum plus regularization)
        pole = mesh_points[x] + eps

        # Build coupled matrix A for the linear system (I - V*G₀)
        for i in range(Nrows):
            for j in range(Nrows):
                # Block (0,0): channel 0 → channel 0
                A[i, j] = _delta(i, j) - 2.0 / np.pi * mesh_weights[j] * V00[
                    i, j
                ] * mesh_points[j] ** 2 / (pole**2 - mesh_points[j] ** 2)
                
                # Block (0,1): channel 0 → channel 1 (coupling term)
                A[i, j + Nrows] = (
                    -2.0
                    / np.pi
                    * mesh_weights[j]
                    * V01[i, j]
                    * mesh_points[j] ** 2
                    / (pole**2 - mesh_points[j] ** 2)
                )
                
                # Block (1,0): channel 1 → channel 0 (coupling term)
                A[i + Nrows, j] = (
                    -2.0
                    / np.pi
                    * mesh_weights[j]
                    * V10[i, j]
                    * mesh_points[j] ** 2
                    / (pole**2 - mesh_points[j] ** 2)
                )
                
                # Block (1,1): channel 1 → channel 1
                A[i + Nrows, j + Nrows] = _delta(i, j) - 2.0 / np.pi * mesh_weights[
                    j
                ] * V11[i, j] * mesh_points[j] ** 2 / (pole**2 - mesh_points[j] ** 2)

        # Compute principal value sum for regularization
        sum = 0.0
        for i in range(Nrows):
            sum += mesh_weights[i] / (pole**2 - mesh_points[i] ** 2)

        # Augmented equations for on-shell elements (4 additional equations)
        for i in range(Nrows):
            # Equations for channel 0 on-shell
            A[2 * Nrows, i] = (
                -2.0
                / np.pi
                * mesh_weights[i]
                * mesh_points[i] ** 2
                * V00[x, i]
                / (pole**2 - mesh_points[i] ** 2)
            )
            A[i, 2 * Nrows] = (
                +2.0 / np.pi * V00[i, x] * pole**2 * (sum - _counterterm(pmax, pole))
            )

            # Coupling terms for on-shell equations
            A[2 * Nrows, i + Nrows] = (
                -2.0
                / np.pi
                * mesh_weights[i]
                * mesh_points[i] ** 2
                * V01[x, i]
                / (pole**2 - mesh_points[i] ** 2)
            )
            A[i, 2 * Nrows + 1] = (
                +2.0 / np.pi * V01[i, x] * pole**2 * (sum - _counterterm(pmax, pole))
            )

            # More coupling terms
            A[2 * Nrows + 1, i] = (
                -2.0
                / np.pi
                * mesh_weights[i]
                * mesh_points[i] ** 2
                * V10[x, i]
                / (pole**2 - mesh_points[i] ** 2)
            )
            A[i + Nrows, 2 * Nrows] = (
                +2.0 / np.pi * V10[i, x] * pole**2 * (sum - _counterterm(pmax, pole))
            )

            # Equations for channel 1 on-shell
            A[2 * Nrows + 1, i + Nrows] = (
                -2.0
                / np.pi
                * mesh_weights[i]
                * mesh_points[i] ** 2
                * V11[x, i]
                / (pole**2 - mesh_points[i] ** 2)
            )
            A[i + Nrows, 2 * Nrows + 1] = (
                +2.0 / np.pi * V11[i, x] * pole**2 * (sum - _counterterm(pmax, pole))
            )

        # On-shell diagonal elements
        A[2 * Nrows, 2 * Nrows] = 1 + 2.0 / np.pi * V00[x, x] * pole**2 * (
            sum - _counterterm(pmax, pole)
        )
        A[2 * Nrows, 2 * Nrows + 1] = (
            +2.0 / np.pi * V01[x, x] * pole**2 * (sum - _counterterm(pmax, pole))
        )
        A[2 * Nrows + 1, 2 * Nrows] = (
            +2.0 / np.pi * V10[x, x] * pole**2 * (sum - _counterterm(pmax, pole))
        )
        A[2 * Nrows + 1, 2 * Nrows + 1] = 1 + 2.0 / np.pi * V11[x, x] * pole**2 * (
            sum - _counterterm(pmax, pole)
        )

        # Right-hand side: two columns for the two channels
        bvec = np.zeros([2 * Nrows + 2, 2], float)
        for i in range(Nrows):
            bvec[i, 0] = V00[i, x]
            bvec[i, 1] = V01[i, x]
            bvec[i + Nrows, 0] = V10[i, x]
            bvec[i + Nrows, 1] = V11[i, x]

        bvec[2 * Nrows, 0] = V00[x, x]
        bvec[2 * Nrows, 1] = V01[x, x]
        bvec[2 * Nrows + 1, 0] = V10[x, x]
        bvec[2 * Nrows + 1, 1] = V11[x, x]

        # Solve coupled linear system A * K = V
        xvec = np.linalg.solve(A, bvec)

        # Extract K-matrix blocks
        for i in range(Nrows):
            K[i, x] = xvec[i, 0]
            K[i, x + Nrows] = xvec[i, 1]
            K[i + Nrows, x] = xvec[i + Nrows, 0]
            K[i + Nrows, x + Nrows] = xvec[i + Nrows, 1]

    return K
