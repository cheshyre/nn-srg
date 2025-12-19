# Copyright (c) 2025 Matthias Heinz
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from nn_srg.potential import load_1S0_potential, fast_and_lazy_plot
from nn_srg.srg import SRG

# EM500 or AV18
name = "EM500"

# Load potential
pot = load_1S0_potential(name)
fast_and_lazy_plot(pot, 2.0)

# SRG evolve potential
srg = SRG(pot)
srg.evolve(1.8, verbose=True)

fast_and_lazy_plot(srg.get_potential(), 2.0)
