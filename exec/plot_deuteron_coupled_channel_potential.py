# pylint: skip-file
"""Plot deuteron s1 3d1 coupled channel."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

# Check if module is installed, otherwise load locally
try:
    import nn_srg.potential as potential
    import nn_srg.srg as srg
except ImportError:
    from context import nn_srg
    potential = nn_srg.potential
    srg = nn_srg.srg


# Load unevolved potential
chan_3s1 = potential.load(2, 3, 'EM420new', '10010', 50, 'np')
chan_3s1_3d1 = potential.load(2, 3, 'EM420new', '10210', 50, 'np')
chan_3d1_3s1 = potential.load(2, 3, 'EM420new', '12010', 50, 'np')
chan_3d1 = potential.load(2, 3, 'EM420new', '12210', 50, 'np')

# Create coupled channel potential from channels
c_potential = potential.CoupledPotential([chan_3s1, chan_3s1_3d1, chan_3d1_3s1,
                                          chan_3d1])

# Plot initial potential
potential.plot(c_potential, -2.0, 2.0)

# Set up SRG evolution
v_mask = np.array([[0 for _ in range(len(c_potential.nodes))] for _ in
                   range(len(c_potential.nodes))])
k_mask = np.array([[1 for _ in range(len(c_potential.nodes))] for _ in
                   range(len(c_potential.nodes))])

srg_obj = srg.SRG(c_potential, v_mask, k_mask)

# Create list of lambdas to which to evolve
lambdas = [25] + list(range(10, 4, -1)) + list(np.arange(4, 3.1, -0.5)) \
    + list(np.arange(3.0, 2.5, -0.2)) + list(np.arange(2.4, 1.38, -0.05))

for l in lambdas:
    # Evolve potential
    srg_obj.evolve(l, verbose=False, integrator='dopri5', atol=10**(-6),
                   rtol=10**(-6), nsteps=10**(5))

    # Extract evolved potential
    c_potential = srg_obj.get_potential()

# Plot evolved potential
potential.plot(c_potential, -2.0, 2.0)
