# Copyright (c) 2025 Matthias Heinz
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# This code is adapted from code written by Yannick Dietz and Kai Hebeler
# with permission of the original authors.

import numpy as np
from .constants import hbarc, nucleon_mass
from .potential import Potential, CoupledPotential

M = nucleon_mass
eps = 1e-3


def compute_phase_shifts_single_channel(pot: Potential):
    Nrows = len(pot.nodes)

    K = _compute_K_matrix(
        pot.without_weights(), Nrows, np.max(pot.nodes) + 1.0, pot.nodes, pot.weights
    )

    ps = [
        _compute_phase_shift_single_point(K, i, pot.nodes) for i in range(len(pot.nodes))
    ]
    Es = [convert_p_to_Elab(p) for p in pot.nodes]

    ps = _fix_boundary_conditions_and_discontinuities(ps)

    return np.array(Es), np.array(ps)


def compute_phase_shifts_coupled_channel(pot: CoupledPotential):

    V00 = pot.extract_channel_potential(pot._channels[0])
    V01 = pot.extract_channel_potential(pot._channels[1])
    V10 = pot.extract_channel_potential(pot._channels[2])
    V11 = pot.extract_channel_potential(pot._channels[3])

    Nrows = len(V00.nodes)

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

    ps = [
        _compute_phase_shifts_coupled_single_point(K, i, pot.nodes, Nrows)
        for i in range(Nrows)
    ]
    ps0 = np.array([x[0] for x in ps])
    ps1 = np.array([x[1] for x in ps])
    angle = [x[2] for x in ps]
    Es = [convert_p_to_Elab(p) for p in pot.nodes[:Nrows]]

    ps0 = _fix_boundary_conditions_and_discontinuities(ps0)
    ps1 = _fix_boundary_conditions_and_discontinuities(ps1)

    return np.array(Es), np.array(ps0), np.array(ps1), np.array(angle)


def convert_p_to_Elab(p):
    return 2 * p**2 * hbarc**2 / M


def convert_Elab_to_p(E):
    return np.sqrt(M * E / 2 / hbarc**2)


# ------------ Internal methods ---------------- #


def _fix_large_E(vals):
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


def _has_discont(vals):
    for i in reversed(range(1, len(vals))):
        if np.abs(vals[i] - vals[i - 1]) > 90:
            return True
    return False


def _fix_one_disc(vals):
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


def _fix_boundary_conditions_and_discontinuities(vals):
    vals = _fix_large_E(vals)

    while _has_discont(vals):
        vals = _fix_one_disc(vals)

    return vals


def _counterterm(pmax, pole):
    return np.arctanh(pole / pmax) / pole


def _compute_phase_shift_single_point(K, i, mesh_points):
    return 180.0 / np.pi * np.arctan(-mesh_points[i] * K[i, i])


def _compute_phase_shifts_coupled_single_point(K, i, mesh_points, Nrows):
    epsilon = np.arctan(2 * K[i, i + Nrows] / (K[i, i] - K[i + Nrows, i + Nrows])) / 2.0
    r_epsilon = (K[i, i] - K[i + Nrows, i + Nrows]) / (np.cos(2 * epsilon))
    delta_a = -np.arctan(
        mesh_points[i] * (K[i, i] + K[i + Nrows, i + Nrows] + r_epsilon) / 2.0
    )
    delta_b = -np.arctan(
        mesh_points[i] * (K[i, i] + K[i + Nrows, i + Nrows] - r_epsilon) / 2.0
    )

    epsilonbar = np.arcsin(np.sin(2 * epsilon) * np.sin(delta_a - delta_b)) / 2.0
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

    epsilonbar *= -180.0 / np.pi
    return (delta_1, delta_2, epsilonbar)


def _delta(i, j):
    if i == j:
        return 1
    else:
        return 0


def _compute_K_matrix(V, Nrows, pmax, mesh_points, mesh_weights):
    A = np.zeros([Nrows + 1, Nrows + 1], float)
    K = np.zeros([Nrows, Nrows], float)

    for x in range(Nrows):
        pole = mesh_points[x] + eps

        for i in range(Nrows):
            for j in range(Nrows):
                A[i, j] = _delta(i, j) - 2.0 / np.pi * mesh_weights[j] * V[
                    i, j
                ] * mesh_points[j] ** 2 / (pole**2 - mesh_points[j] ** 2)

        sum = 0.0
        for i in range(Nrows):
            sum += mesh_weights[i] / (pole**2 - mesh_points[i] ** 2)

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

        bvec = np.zeros([Nrows + 1], float)
        for i in range(Nrows):
            bvec[i] = V[i, x]
        bvec[Nrows] = V[x, x]

        xvec = np.linalg.solve(A, bvec)

        for i in range(Nrows):
            K[i, x] = xvec[i]

    return K


def _compute_K_matrix_coupled(
    V00, V01, V10, V11, Nrows, pmax, mesh_points, mesh_weights
):
    A = np.zeros([2 * Nrows + 2, 2 * Nrows + 2], float)
    K = np.zeros([2 * Nrows, 2 * Nrows], float)

    for x in range(Nrows):
        pole = mesh_points[x] + eps

        for i in range(Nrows):
            for j in range(Nrows):
                A[i, j] = _delta(i, j) - 2.0 / np.pi * mesh_weights[j] * V00[
                    i, j
                ] * mesh_points[j] ** 2 / (pole**2 - mesh_points[j] ** 2)
                A[i, j + Nrows] = (
                    -2.0
                    / np.pi
                    * mesh_weights[j]
                    * V01[i, j]
                    * mesh_points[j] ** 2
                    / (pole**2 - mesh_points[j] ** 2)
                )
                A[i + Nrows, j] = (
                    -2.0
                    / np.pi
                    * mesh_weights[j]
                    * V10[i, j]
                    * mesh_points[j] ** 2
                    / (pole**2 - mesh_points[j] ** 2)
                )
                A[i + Nrows, j + Nrows] = _delta(i, j) - 2.0 / np.pi * mesh_weights[
                    j
                ] * V11[i, j] * mesh_points[j] ** 2 / (pole**2 - mesh_points[j] ** 2)

        sum = 0.0
        for i in range(Nrows):
            sum += mesh_weights[i] / (pole**2 - mesh_points[i] ** 2)

        for i in range(Nrows):
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

        xvec = np.linalg.solve(A, bvec)

        for i in range(Nrows):
            K[i, x] = xvec[i, 0]
            K[i, x + Nrows] = xvec[i, 1]
            K[i + Nrows, x] = xvec[i + Nrows, 0]
            K[i + Nrows, x + Nrows] = xvec[i + Nrows, 1]

    return K
