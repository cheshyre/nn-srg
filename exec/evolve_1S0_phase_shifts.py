# Copyright (c) 2025 Matthias Heinz
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from nn_srg.potential import load_1S0_potential, fast_and_lazy_plot
from nn_srg.srg import SRG
from nn_srg.phase_shift import compute_phase_shifts_single_channel
import matplotlib.pyplot as plt

# EM500 or AV18
name = "EM500"

# Load potential
pot = load_1S0_potential(name)
# fast_and_lazy_plot(pot, 2.0)
p_vals, delta_vals = compute_phase_shifts_single_channel(pot)

# SRG evolve potential
srg = SRG(pot)
srg.evolve(1.8, verbose=True)

# fast_and_lazy_plot(srg.get_potential(), 2.0)
p_vals, delta_vals_new = compute_phase_shifts_single_channel(srg.get_potential())

print(p_vals)
print(delta_vals)
print(delta_vals_new)
print(delta_vals - delta_vals_new)

plt.plot(p_vals, delta_vals, label=r"$\lambda = \infty$")
plt.plot(p_vals, delta_vals_new, label=r"$\lambda = 1.8\:\mathrm{fm}^{-1}$")
plt.ylim(0, 80)
plt.xlim(0, 200)
plt.legend(loc="best")
plt.ylabel(r"$\delta$ (degrees)")
plt.xlabel(r"$E_\mathrm{lab}$ (MeV)")
plt.show()
