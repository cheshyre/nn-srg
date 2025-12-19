# Copyright (c) 2025 Matthias Heinz
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import pathlib

import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import scipy.interpolate

from nn_srg.potential import load_1S0_potential
from nn_srg.srg import SRG

# EM500 or AV18
name = "EM500"

# Load potential
pot = load_1S0_potential(name)

# SRG evolve potential
srg = SRG(pot)

# matplotlib font configurations

plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{lmodern}\usepackage{amsmath}"
plt.rcParams["font.family"] = ["Latin Modern Roman"]
plt.rcParams["font.size"] = 10

fig_width = 6.5
fig_height = 1.60

# Set maximum momentum to show
kmax = 5.0

# Define new even grid for data
nodes = np.linspace(0.0, kmax, 200)

# Set lambdas to show
lambdas = [(25.0, r"\infty")] + [
    (x, f"{x:0.1f}" + r"\:\mathrm{fm}^{-1}") for x in [4.0, 2.5, 2.0, 1.8]
]

# Set up initial figure
fig = plt.figure(1, (fig_width, fig_height))

# Set up grid in figure
grid = ImageGrid(
    fig,
    111,
    nrows_ncols=(1, len(lambdas)),
    axes_pad=0.00,
    share_all=False,
    label_mode="L",
    #  cbar_location="right",
    #  cbar_mode="single",
)

# Iterate over lambdas
for i, lll in enumerate(lambdas):
    lam, lam_disp = lll
    srg.evolve(lam)
    # Get old nodes for data
    data_nodes = np.array(srg.get_potential().nodes)

    # Read in potential
    potential = np.array(srg.get_potential().without_weights())

    # Interpolate potential
    interp = scipy.interpolate.RectBivariateSpline(data_nodes, data_nodes, potential)
    potential = interp(nodes, nodes)

    # Plot potential matrix on corresponding axis in grid
    ax = grid[i]
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(5))
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(5))
    ax.tick_params(bottom=True, top=True, left=True, right=True, which="both")
    ax.tick_params(axis="x", which="both", direction="in")
    ax.tick_params(axis="y", which="both", direction="in")
    ax.tick_params(which="major", length=5)

    im = ax.matshow(
        potential,
        extent=[0.0, kmax, kmax, 0.0],
        vmin=-1,
        vmax=1,
        cmap=plt.cm.RdBu_r,
    )

    # Set ticks for axes for this axis
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_yticks([0, 1, 2, 3, 4])

    # Set xlabel on 3rd grid
    if i == 2:
        ax.set_xlabel(r"$p$ (fm$^{-1}$)", labelpad=8)
        ax.xaxis.set_label_position("top")

    # Set ylabel on first grid
    if i == 0:
        ax.set_ylabel(r"$p'$ (fm$^{-1}$)")

    # Add label for lambda value
    ax.text(0.4, 4.5, "$\\lambda={}$".format(lam_disp))

    # Move xticks to top
    ax.tick_params(axis="x", which="both", bottom=True, top=True, labelbottom=False)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")
    # Disable yticks for grids after the left most
    if i != 0:
        ax.tick_params(labelleft=False)
        # ax.tick_params(axis="y", which="both", left=False)

    if i == len(lambdas) - 1:
        axins = inset_axes(
            ax,
            width="10%",
            height="100%",
            loc="lower left",
            bbox_to_anchor=(1.05, 0, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )

# Set colorbar
#  grid.cbar_axes[0].colorbar(im)
cbar = plt.colorbar(im, cax=axins, ticks=[-1.0, -0.5, 0.0, 0.5, 1.0])
cbar.ax.set_yticklabels(["$-1.0$", "$-0.5$", "0 (fm)", "0.5", "1.0"])
# Set colorbar labels
#  for i, cax in enumerate(grid.cbar_axes):
#      cax.set_yticks([-0.5, 0, 0.5])
#      labels = [item.get_text() for item in cax.get_yticklabels()]
#      labels = ["-0.5", "", "0.5"]
#      labels[1] = r"0 (MeV)"
#      cax.set_yticklabels(labels)

# Set xticks to top
plt.tick_params(axis="x", which="both", bottom=True, top=True, labelbottom=False)


# Adjust plot boundaries
plt.gcf().subplots_adjust(left=0.06)
plt.gcf().subplots_adjust(right=0.90)
plt.gcf().subplots_adjust(top=0.80)
plt.gcf().subplots_adjust(bottom=0.01)

# Save plot
plt.gcf().set_size_inches(fig_width, fig_height)
path = __file__.replace(".py", ".pdf").replace("scripts", "plots")
pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(path, dpi=300)
plt.close(fig)
