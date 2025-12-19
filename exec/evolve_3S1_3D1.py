# Copyright (c) 2025 Matthias Heinz
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from nn_srg.potential import load_3S1_3D1_potential, fast_and_lazy_plot
from nn_srg.srg import SRG
from nn_srg.diagonalize import diagonalize_hamiltonian

# EM500 or AV18
name = "EM500"

# Load potential
pot = load_3S1_3D1_potential(name)
print(diagonalize_hamiltonian(pot))
fast_and_lazy_plot(pot.extract_channel_potential(pot._channels[0]), 2.0)

# SRG evolve potential
srg = SRG(pot)
srg.evolve(1.8, verbose=True)

print(diagonalize_hamiltonian(srg.get_potential()))
fast_and_lazy_plot(srg.get_potential().extract_channel_potential(pot._channels[0]), 2.0)
