# Copyright (c) 2025 Matthias Heinz
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import pathlib

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.interpolate import interp1d

from nn_srg.potential import load_1S0_potential
from nn_srg.srg import SRG

# matplotlib font configurations

plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{lmodern}\usepackage{amsmath}"
plt.rcParams["font.family"] = ["Latin Modern Roman"]
plt.rcParams["font.size"] = 10

# Set up initial figure
fig, axs = plt.subplots(1, 1)

fig_width = 3.5
fig_height = 3.5

# Set maximum momentum to show
kmax = 5.0

# Define new even grid for data
i_nodes = np.linspace(2.0, kmax - 1e-3, 200)

# Set lambdas to show
lambdas = [
    # (25.0, r"\infty")
] + [(x, f"{x:0.1f}" + r"\:\mathrm{fm}^{-1}") for x in [4.0, 3.0, 2.5, 2.0]]

line_styles = [":", "--", "-.", "-"]


# EM500 or AV18
name = "EM500"
color = "blue"

# Load potential
pot = load_1S0_potential(name)

# SRG evolve potential
srg = SRG(pot)
nodes = np.array(pot.nodes)
orig = np.array(pot.without_weights()[0])

f = interp1d(nodes, orig, kind="cubic", bounds_error=False, fill_value=0.0)

i_orig = f(i_nodes)

# Iterate over lambdas
for i, lll in enumerate(lambdas):
    lam, lam_disp = lll
    srg.evolve(lam)

    # Read in potential
    new = np.array(srg.get_potential().without_weights()[0])

    # Plot potential matrix on corresponding axis in grid
    ax = axs
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(5))
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(5))
    ax.tick_params(bottom=True, top=True, left=True, right=True, which="both")
    ax.tick_params(axis="x", which="both", direction="in")
    ax.tick_params(axis="y", which="both", direction="in")
    ax.tick_params(which="major", length=5)

    f = interp1d(nodes, new, kind="cubic", bounds_error=False, fill_value=0.0)
    i_new = f(i_nodes)
    i_ratio = i_new / (i_orig + 1e-8)

    ax.plot(
        i_nodes**2,
        i_ratio,
        color=color,
        ls=line_styles[i],
        label=r"$\lambda = " + lam_disp + r"$",
    )

ax.set_xlim((4, 25))
ax.set_yscale("log")
ax.set_ylim((1e-4, 1.0))
ax.legend(framealpha=1.0)
ax.set_xlabel(r"$p^2$ (fm$^{-2}$)")
ax.set_ylabel(r"$V_\lambda(0, p) / V_\infty(0, p)$")

# Adjust plot boundaries
plt.gcf().subplots_adjust(left=0.17)
plt.gcf().subplots_adjust(right=0.97)
plt.gcf().subplots_adjust(top=0.95)
plt.gcf().subplots_adjust(bottom=0.15)

# Save plot
plt.gcf().set_size_inches(fig_width, fig_height)
path = __file__.replace(".py", ".pdf").replace("scripts", "plots")
pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(path, dpi=300)
plt.close(fig)
