# Copyright (c) 2018-2025 Matthias Heinz
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
"""Module for Similarity Renormalization Group (SRG) evolution of nuclear potentials.

The Similarity Renormalization Group is a continuous unitary transformation method
that systematically softens short-range interactions while preserving low-energy
observables. The evolution is governed by the flow equation:

    dH/ds = [η(s), H(s)]

where H is the Hamiltonian, s is the flow parameter, and η is the generator.
For nuclear interactions, the Wegner generator η = [H_d, H] (with H_d diagonal 
in momentum) and the Trel generator η = [Trel, H] are commonly used.

This module implements the SRG flow equation in momentum space,
with support for custom flow operators via masking.

References
----------
For the SRG method, see:

- Wegner, F. (1994). "Flow equations for Hamiltonians."
  Annalen der Physik, 506(2), 77-91.
  https://doi.org/10.1002/andp.19945060203

- Głazek, S. D., & Wilson, K. G. (1993). "Renormalization of Hamiltonians."
  Physical Review D, 48(12), 5863.
  https://doi.org/10.1103/PhysRevD.48.5863

- Bogner, S. K., Furnstahl, R. J., & Perry, R. J. (2007).
  "Similarity renormalization group for nucleon-nucleon interactions."
  Physical Review C, 75(6), 061001.
  https://doi.org/10.1103/PhysRevC.75.061001

Classes
-------
SRG
    An abstraction for the SRG evolution intended to work like a numerical
    integrator. It has the following methods::

        srg = SRG(potential)
        srg.evolve(lam)
        evolved_potential = srg.get_potential()
        srg.replace_potential(new_potential)

"""
from math import pi, sqrt
from typing import Any, Optional

import numpy as np
import scipy.integrate as integ


class SRG:
    """Interface for 3D SRG evolution of momentum-space potentials.

    The SRG evolution is performed using the flow equation with a specified
    generator (flow operator). The default is the Wegner generator, where
    the flow operator is constructed from the diagonal Hamiltonian.

    The evolution uses λ = s^(-1/4) as the flow parameter (in fm⁻¹), where
    s is the standard flow parameter.

    Attributes
    ----------
    _potential : Potential
        Current potential object being evolved.
    _v : ndarray
        Current potential matrix (without integration weights).
    _k : ndarray
        Kinetic energy matrix (with integration weights).
    _lam : float
        Current SRG flow parameter λ (in fm⁻¹).
    _flow_op_mask_v : ndarray
        Mask for potential contribution to flow operator.
    _flow_op_mask_k : ndarray
        Mask for kinetic energy contribution to flow operator.
    """

    def __init__(
        self,
        potential,
        flow_operator_mask_v: Optional[np.ndarray] = None,
        flow_operator_mask_k: Optional[np.ndarray] = None,
    ):
        """Initialize SRG evolution object.

        Parameters
        ----------
        potential : Potential
            Potential object to be evolved. Must provide methods for
            extracting the unweighted potential matrix, kinetic energy,
            and the current λ value.
        flow_operator_mask_v : ndarray, optional
            Matrix mask for potential contribution to flow operator.
            Default is zero (standard Trel generator uses kinetic energy only).
        flow_operator_mask_k : ndarray, optional
            Matrix mask for kinetic energy contribution to flow operator.
            Default is unity (full kinetic energy in Trel generator).

        Notes
        -----
        The flow operator masks allow for customized generators. For the
        standard Wegner generator η = [H_d, H], use the default masks.

        """
        self._potential = potential
        self._v = potential.without_weights()
        self._k = potential.kinetic_energy()
        self._lam = potential.lam
        if flow_operator_mask_v is None:
            flow_operator_mask_v = np.zeros_like(self._v)
        if flow_operator_mask_k is None:
            flow_operator_mask_k = np.ones_like(self._v)
        self._flow_op_mask_v = flow_operator_mask_v
        self._flow_op_mask_k = flow_operator_mask_k
        self._flow = "lambda"

    def evolve(
        self,
        lam: float,
        verbose: bool = False,
        integrator: str = "dopri5",
        **integrator_params: Any,
    ) -> "SRG":
        """Evolve potential to specified λ value.

        Solves the SRG flow equation from the current λ to the target λ
        using numerical integration.

        Parameters
        ----------
        lam : float
            Target SRG flow parameter λ (in fm⁻¹) to which the potential
            should be evolved. Smaller values correspond to softer interactions.
        verbose : bool, optional
            If True, print the current flow parameter during evolution.
            Default is False.
        integrator : str, optional
            Name of scipy ODE integrator to use. Default is 'dopri5'
            (Dormand-Prince Runge-Kutta method of order 4/5).
        **integrator_params : dict, optional
            Additional parameters for the integrator. If none specified,
            uses high-precision defaults: atol=1e-12, rtol=1e-12, nsteps=1e9.

        Returns
        -------
        SRG
            Returns self for method chaining.

        Raises
        ------
        Exception
            If the numerical integration fails.

        See Also
        --------
        scipy.integrate.ode : ODE solver interface with integrator options.

        Notes
        -----
        For numerical stability, we use λ = s^(-1/4) as the evolution parameter
        rather than s directly, as the flow equation is stiff in s. The unevolved
        potential corresponds to s=0 or λ→∞, approximated by λ=50.0 fm⁻¹.

        Evolution to λ < 2.0 fm⁻¹ is typical for nuclear structure calculations,
        balancing convergence in many-body methods with numerical precision.

        Examples
        --------
        >>> srg = SRG(initial_potential)
        >>> srg.evolve(lam=2.0)  # Evolve to λ = 2.0 fm⁻¹
        >>> evolved_pot = srg.get_potential()

        """
        solver = integ.ode(_srg_rhs)

        # High-precision defaults chosen for accurate SRG evolution
        if not integrator_params:
            solver.set_integrator(
                integrator,
                atol=1e-12,  # Absolute tolerance for integration
                rtol=1e-12,  # Relative tolerance for integration
                nsteps=int(1e9),  # Maximum number of integration steps
            )
        else:
            solver.set_integrator(integrator, **integrator_params)

        solver.set_f_params(
            self._k, self._flow_op_mask_v, self._potential.weights, self._flow, verbose
        )

        solver.set_initial_value(_flatten(self._v), self._lam)
        solver.integrate(lam)

        if solver.successful():
            self._v = _unflatten(solver.y)
        else:
            raise Exception("Integration failed.")

        # Update flow parameter
        self._lam = lam

        return self

    def get_potential(self):
        """Return new Potential object at current flow parameter.

        Returns
        -------
        Potential
            New Potential object corresponding to the current state of the
            SRG evolution, with updated potential matrix and λ value.

        """
        return self._potential.copy(self._v, self._lam)

    def replace_potential(
        self,
        new_potential,
        flow_operator_mask_v: Optional[np.ndarray] = None,
        flow_operator_mask_k: Optional[np.ndarray] = None,
    ) -> None:
        """Replace potential being used for SRG evolution.

        Parameters
        ----------
        new_potential : Potential
            New potential to replace current potential. Must have the same
            potential type and be at the same λ value.
        flow_operator_mask_v : ndarray, optional
            New mask for potential with correct dimensions.
        flow_operator_mask_k : ndarray, optional
            New mask for kinetic energy with correct dimensions.

        Raises
        ------
        ValueError
            If new potential has different potential type or is at a
            different λ value (tolerance: 1e-4 fm⁻¹).

        Notes
        -----
        This method is primarily intended to support dimension reduction of the
        momentum grid as the SRG evolution progresses. High-momentum components
        become decoupled during evolution and can be safely truncated.

        For starting a new evolution, create a new SRG object instead.

        """
        # Tolerance for λ comparison (in fm⁻¹)
        eps = 1e-4

        if self._potential.potential_type != new_potential.potential_type:
            raise ValueError("New potential does not have same type.")
        if abs(self._lam - new_potential.lam) > eps:
            raise ValueError(
                f"New potential is not at the same λ: "
                f"current={self._lam:.4f}, new={new_potential.lam:.4f}"
            )

        self._potential = new_potential
        self._v = new_potential.without_weights()
        self._k = new_potential.kinetic_energy()
        self._lam = new_potential.lam
        if flow_operator_mask_v is None:
            flow_operator_mask_v = np.zeros_like(self._v)
        if flow_operator_mask_k is None:
            flow_operator_mask_k = np.ones_like(self._v)
        self._flow_op_mask_v = flow_operator_mask_v
        self._flow_op_mask_k = flow_operator_mask_k


# ---------------------------- Internal Methods ---------------------------- #


def _srg_rhs(
    s: float,
    potential: np.ndarray,
    kinetic: np.ndarray,
    potential_weight: np.ndarray,
    weights: np.ndarray,
    flow: str,
    verbose: bool,
) -> np.ndarray:
    """Compute right-hand side of SRG flow equation.

    Evaluates dH/ds = [η, H] where η is the generator
    and H is the Hamiltonian.

    The flow equation in nested commutator form is:
        dH/ds = [[G, H], H]
    where G is the flow operator constructed from H_d and optionally
    parts of V.

    Parameters
    ----------
    s : float
        Current SRG flow parameter (related to λ by λ = s^(-1/4)).
    potential : ndarray
        Flattened potential matrix V(s).
    kinetic : ndarray
        Kinetic energy matrix T (with integration weights).
    potential_weight : ndarray
        Mask for potential contribution to flow operator.
    weights : ndarray
        Gaussian quadrature weights.
    flow : str
        Flow parameterization: 'lambda' for λ = s^(-1/4) flow.
    verbose : bool
        If True, print current s or λ value.

    Returns
    -------
    ndarray
        Flattened right-hand side dH/ds.

    Notes
    -----
    The conversion factor -4.0/s^5 accounts for the chain rule when
    using λ = s^(-1/4) instead of s as the flow parameter.

    """
    T = kinetic
    V = _unflatten(potential)

    # Compute integration weights for momentum space
    nodes_sq = T.diagonal()
    w_sqrt = [sqrt(2 * w * p_sq / pi) for w, p_sq in zip(weights, nodes_sq)]
    W_matrix = np.diag(w_sqrt)
    W_matrix_inv = np.diag([1 / x for x in w_sqrt])

    # Add integration weights to potential
    V_w = _mm(W_matrix, _mm(V, W_matrix))

    # Construct Hamiltonian H = T + V
    H = T + V_w

    # Construct flow operator G (masked parts of Hamiltonian)
    X = np.multiply(V_w, potential_weight)
    G = T + X

    # Compute nested commutator [[G, H], H]
    rhs = _com(_com(G, H), H)

    # Remove integration weights from result
    rhs = _mm(W_matrix_inv, _mm(rhs, W_matrix_inv))

    if verbose:
        print(f"s = {s:.6e}")

    # Convert from s-flow to λ-flow: dH/dλ = dH/ds * ds/dλ
    # where λ = s^(-1/4) implies ds/dλ = -4λ^5
    if flow == "lambda":
        factor = -4.0 / (s**5)
        rhs *= factor

    return _flatten(rhs)


def _com(matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray:
    """Compute commutator [A, B] = AB - BA.

    Parameters
    ----------
    matrix1 : ndarray
        First matrix A.
    matrix2 : ndarray
        Second matrix B.

    Returns
    -------
    ndarray
        Commutator [A, B].
    """
    return _mm(matrix1, matrix2) - _mm(matrix2, matrix1)


def _mm(matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray:
    """Compute matrix product AB.

    Wrapper for np.dot for consistency with commutator notation.

    Parameters
    ----------
    matrix1 : ndarray
        First matrix A.
    matrix2 : ndarray
        Second matrix B.

    Returns
    -------
    ndarray
        Matrix product AB.
    """
    return np.dot(matrix1, matrix2)


def _flatten(m: np.ndarray) -> np.ndarray:
    """Flatten 2D matrix into 1D array.

    Used for compatibility with scipy ODE solvers which require 1D arrays.

    Parameters
    ----------
    m : ndarray
        2D matrix to flatten.

    Returns
    -------
    ndarray
        1D flattened array.
    """
    return np.reshape(m, m.size)


def _unflatten(m: np.ndarray) -> np.ndarray:
    """Reshape 1D array back to 2D square matrix.

    Inverse operation of _flatten, used to recover matrix structure
    from ODE solver output.

    Parameters
    ----------
    m : ndarray
        1D array to reshape.

    Returns
    -------
    ndarray
        2D square matrix.
    """
    dim = int(m.size**0.5)
    return np.reshape(m, (dim, dim))
