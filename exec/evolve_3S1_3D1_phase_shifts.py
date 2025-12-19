# Copyright (c) 2025 Matthias Heinz
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from nn_srg.potential import load_3S1_3D1_potential, fast_and_lazy_plot
from nn_srg.srg import SRG
from nn_srg.diagonalize import diagonalize_hamiltonian
from nn_srg.phase_shift import compute_phase_shifts_coupled_channel
import matplotlib.pyplot as plt

# EM500 or AV18
name = "EM500"

# Load potential
pot = load_3S1_3D1_potential(name)
print(diagonalize_hamiltonian(pot))
# fast_and_lazy_plot(pot.extract_channel_potential(pot._channels[0]), 2.0)
Es, ps_3S1, ps_3D1, eps = compute_phase_shifts_coupled_channel(pot)

# SRG evolve potential
srg = SRG(pot)
srg.evolve(1.8, verbose=True)

print(diagonalize_hamiltonian(srg.get_potential()))
Es, ps_3S1_new, ps_3D1_new, eps_new = compute_phase_shifts_coupled_channel(
    srg.get_potential()
)
# fast_and_lazy_plot(srg.get_potential().extract_channel_potential(pot._channels[0]), 2.0)

plt.plot(Es, ps_3S1, label=r"$\lambda = \infty$")
plt.plot(Es, ps_3S1_new, label=r"$\lambda = 1.8\:\mathrm{fm}^{-1}$")
plt.ylim(0, 200)
plt.xlim(0, 200)
plt.legend(loc="best")
plt.ylabel(r"$\delta$ (degrees)")
plt.xlabel(r"$E_\mathrm{lab}$ (MeV)")
plt.show()
