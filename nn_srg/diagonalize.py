# Copyright (c) 2025 Matthias Heinz
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from .constants import hbarc, red_mass

import numpy as np

convert_to_mev = hbarc**2 / (2 * red_mass)


def diagonalize_hamiltonian(pot, num_eigvals=1):
    hamiltonian = convert_to_mev * (pot.with_weights() + pot.kinetic_energy())
    evals, evecs = np.linalg.eigh(hamiltonian)

    return evals[:num_eigvals]
