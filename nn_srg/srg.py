"""Module for the SRG evolver.

class SRG
---------
An abstraction for the SRG evolution intended to work like a numerical
integrator. It has the following methods::

    srg = SRG(potential)
    srg.evolve(lam)
    evolved_potential = srg.get_potential()
    srg.replace_potential(new_potential)

Changelog:

2018.11.13
    Changed:
        Made get_potential use potential object copy method instead of
        constructor.

2018.11.08
    Added:
        Initial completion of module. Tested and verified it works.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from math import pi
from math import sqrt

import numpy as np
import scipy.integrate as integ


class SRG:
    """Interface for the 3-D SRG evolution."""

    def __init__(self, potential, flow_operator_mask_v=None, flow_operator_mask_k=None):
        """Initialize SRG evolution object.

        Parameters
        ----------
        potential : nn_srg.potential.Potential
            Potential object to be evolved. Comes with weights, lam, and
            kinetic energy, which is everything you need to do the SRG
            evolution.
        flow_operator_mask_v : matrix of floats
            Matrix which masks parts of the potential when computing the
            flow operator.
        flow_operator_mask_k : matrix of floats
            Matrix which masks parts of the kinetic energy when computing the
            flow operator.

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

    def evolve(self, lam, verbose=False, integrator="dopri5", **integrator_params):
        """Evolve potential to specified lam.

        Evolve the SRG object (more specifically the potential) to the
        specified value of lambda according to the SRG differential equation.

        Parameters
        ----------
        lam : float
            Value of lambda to which the potential should be evolved.
        verbose : bool, optional
            Flag which, if set, will print out the current lam of the evolution
        integrator : string, optional
            Name of integrator used.
        integrator_params :
            Additional parameters for the integrator. If none are specified,
            a default set of parameters will be used. See scipy.integrate.ode
            for details.

        See also
        --------
        scipy.integrate.ode : User-friendly interface to various numerical
            integrators. Options for integrator_params can be found here.

        Notes
        -----
        For numerical reasons, we use lambda = 1/s^(1/4) as our evolution
        parameter, as the differential equation with respect to s is rather
        stiff. Thus, the unevolved potential should be at lambda=Infinity,
        corresponding to s=0. This can be reasonably approximated by
        lambda=50.0. Evolving to lambda larger than this is pointless, but
        possible.

        """
        solver = integ.ode(_srg_rhs)

        # Default parameters for integrator chosen based optimal results while
        # testing
        if not integrator_params:
            solver.set_integrator(
                integrator, atol=10 ** (-12), rtol=10 ** (-12), nsteps=10 ** (9)
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

        # If successful, update value of lambda
        self._lam = lam

        return self

    def get_potential(self):
        """Return new `Potential` object from current potential and lam.

        Returns
        -------
        Potential
            New potential corresponding to the current state of the SRG
            evolution.

        """
        return self._potential.copy(self._v, self._lam)

    def replace_potential(
        self, new_potential, flow_operator_mask_v=None, flow_operator_mask_k=None
    ):
        """Replace potential being used for SRG evolution with another.

        Parameters
        ----------
        new_potential : Potential
            New potential to replace current potential in SRG evolution.
        flow_operator_mask_v : matrix of floats
            New mask for potential with correct dimensions.
        flow_operator_mask_k : matrix of floats
            New mask for kinetic energy with correct dimensions.

        Raises
        ------
        ValueError
            When new potential's potential type or lam is different from
            current potential.

        Notes
        -----
        This is primarily intended to support dimension reduction of the
        potential as the SRG evolution progresses. If you would like to reuse
        the SRG object for a different evolution, please make a new SRG evolver
        instead.

        """
        eps = 10 ** (-4)

        if self._potential.potential_type != new_potential.potential_type:
            raise ValueError("New potential does not have same type.")
        if abs(self._lam - new_potential.lam) > eps:
            raise ValueError("New potential is not at the same lam")

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


# ---------------------------- Internal methods ---------------------------- #


# pylint: disable=too-many-arguments,too-many-locals,invalid-name
def _srg_rhs_old(s, potential, kinetic, potential_weight, weights, flow, verbose):
    """Old implementation of SRG flow equation.

    New implementation is more efficient and reflects actual form better.
    Keeping this for documentation and as reference for analytic evaluation of
    flow equation with general flow operator.

    """
    T = kinetic
    V = _unflatten(potential)
    TT = np.dot(T, T)
    X = np.multiply(V, potential_weight)
    Vdiff = V - X
    VT = np.dot(V, T)
    TV = np.dot(T, V)
    XT = np.dot(X, T)
    TX = np.dot(T, X)
    W = np.diag(weights)

    rhs = (
        -1 * (_mm(Vdiff, TT) + _mm(TT, Vdiff))
        + 2 * _mmm(T, Vdiff, T)
        + 2
        / pi
        * (
            _mmm(TV, W, TV)
            + _mmm(VT, W, VT)
            + _mmm(VT, W, XT)
            + _mmm(TV, W, XT)
            + _mmm(XT, W, VT)
            + _mmm(XT, W, TV)
            - 2 * (_mmm(VT, W, TV) + _mmm(VT, W, TX) + _mmm(TX, W, TV))
        )
        + 4
        / pi**2
        * (
            _mmmmm(VT, W, VT, W, X)
            + _mmmmm(XT, W, VT, W, V)
            - 2 * _mmmmm(VT, W, X, W, TV)
        )
    )

    if verbose:
        print(s)

    # Use commutator defined below
    if flow == "lambda":
        factor = -4.0 / (s**5)
        rhs *= factor
    return _flatten(rhs)


def _srg_rhs(s, potential, kinetic, potential_weight, weights, flow, verbose):
    T = kinetic
    V = _unflatten(potential)

    # Compute integration weights
    nodes_sq = T.diagonal()
    w_sqrt = [sqrt(2 * w * p_sq / pi) for w, p_sq in zip(weights, nodes_sq)]
    W_matrix = np.diag(w_sqrt)
    W_matrix_inv = np.diag([1 / x for x in w_sqrt])

    # Add integration weights
    V_w = _mm(W_matrix, _mm(V, W_matrix))

    # Compute Hamiltonian
    H = T + V_w

    # Get flow operator
    X = np.multiply(V_w, potential_weight)
    G = T + X

    # Compute SRG commutators
    rhs = _com(_com(G, H), H)

    # Remove integration weights
    rhs = _mm(W_matrix_inv, _mm(rhs, W_matrix_inv))

    if verbose:
        print(s)

    # Add factor for lambda flow vs s flow
    if flow == "lambda":
        factor = -4.0 / (s**5)
        rhs *= factor
    return _flatten(rhs)


def _com(matrix1, matrix2):
    return _mm(matrix1, matrix2) - _mm(matrix2, matrix1)


def _mm(matrix1, matrix2):
    return np.dot(matrix1, matrix2)


def _mmm(matrix1, matrix2, matrix3):
    return np.dot(matrix1, np.dot(matrix2, matrix3))


def _mmmmm(matrix1, matrix2, matrix3, matrix4, matrix5):
    return _mm(_mm(matrix1, matrix2), _mmm(matrix3, matrix4, matrix5))


def _flatten(m):
    flattened = np.reshape(m, m.size)
    return flattened


def _unflatten(m):
    unflattened = np.reshape(m, (int(m.size**0.5), int(m.size**0.5)))
    return unflattened
