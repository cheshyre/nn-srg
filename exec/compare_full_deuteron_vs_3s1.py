# pylint: skip-file
"""Compute deuteron lowest energy state in 3s1 3d1 coupled channel and 3s1."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from math import sqrt

import numpy as np
from numpy.linalg import eigh

# Check if module is installed, otherwise load locally
try:
    import nn_srg.potential as potential
    import nn_srg.srg as srg
except ImportError:
    from context import nn_srg
    potential = nn_srg.potential
    srg = nn_srg.srg


# Physical constants
hbarc = 197.327
proton_mass = 938.272
neutron_mass = 939.565
red_mass = proton_mass * neutron_mass / (proton_mass + neutron_mass)

# Load unevolved potential
chan_3s1 = potential.load(2, 3, 'EM420new', '10010', 50, 'np')
chan_3s1_3d1 = potential.load(2, 3, 'EM420new', '10210', 50, 'np')
chan_3d1_3s1 = potential.load(2, 3, 'EM420new', '12010', 50, 'np')
chan_3d1 = potential.load(2, 3, 'EM420new', '12210', 50, 'np')

# Create coupled channel potential from channels
full_potential = potential.CoupledPotential([chan_3s1, chan_3s1_3d1,
                                             chan_3d1_3s1, chan_3d1])


# Compute hamiltonian
hamiltonian_full = full_potential.with_weights() \
    + full_potential.kinetic_energy()
hamiltonian_single = chan_3s1.with_weights() \
    + chan_3s1.kinetic_energy()

# Compute bound state eigenvalues
ev_full = np.amin(eigh(hamiltonian_full)[0])
ev = np.amin(eigh(hamiltonian_single)[0])

# Print unevolved results
print('Unevolved')
print('E_full =  {} MeV'.format(hbarc**2 / (2 * red_mass) * ev_full))
print('E_3s1 = {} MeV\n'.format(hbarc**2 / (2 * red_mass) * ev))

# Set up SRG evolution
v_mask = np.array([[0 for _ in range(len(full_potential.nodes))] for _ in
                   range(len(full_potential.nodes))])
k_mask = np.array([[1 for _ in range(len(full_potential.nodes))] for _ in
                   range(len(full_potential.nodes))])

srg_obj_full = srg.SRG(full_potential, v_mask, k_mask)

# Create list of lambdas to which to evolve
lambdas = [25] + list(range(10, 4, -1)) + list(np.arange(4, 3.1, -0.5)) \
    + list(np.arange(3.0, 2.5, -0.2)) + list(np.arange(2.4, 1.38, -0.05)) \
    + list(np.arange(1.35, 0.39, -0.05))
num_pts = [100]*9 + [83] + [79] + [76] + [73] + [72]*2 + [71] + [70]*2 + [69] \
    + [68]*2 + [67]*2 + [66] + [65]*2 + [64]*2 + [63] + [62]*2 + [61]*2 \
    + [60]*20

potential.plot(full_potential, -2.0, 2.0)

for l, n in zip(lambdas, num_pts):
    # Evolve potential
    srg_obj_full.evolve(l, verbose=False, integrator='dopri5', atol=10**(-6),
                        rtol=10**(-6), nsteps=10**(5))

    # Extract evolved potential
    full_potential = srg_obj_full.get_potential()

    # Reduce dimension if necessary
    if n < full_potential.dim:
        full_potential = full_potential.reduce_dim(n)
        v_mask = np.array([[0 for _ in range(len(full_potential.nodes))]
                           for _ in range(len(full_potential.nodes))])
        k_mask = np.array([[1 for _ in range(len(full_potential.nodes))]
                           for _ in range(len(full_potential.nodes))])
        srg_obj_full.replace_potential(full_potential, v_mask, k_mask)

    # Compute Hamiltonian
    hamiltonian_full = full_potential.with_weights() \
        + full_potential.kinetic_energy()

    # Load reference potential (calculated by different code)
    #  chan_3s1 = potential.load(2, 3, 'EM420new', '10010', l, 'np')
    chan_3s1 = full_potential.extract_channel_potential(
        chan_3s1.potential_type.channel)

    hamiltonian = chan_3s1.with_weights() \
        + chan_3s1.kinetic_energy()

    # Get lowest eigenvalues
    ev_full = np.amin(eigh(hamiltonian_full)[0])
    ev = np.amin(eigh(hamiltonian)[0])

    # Output values
    print('Lambda: {}'.format(l))
    print('E_full = {} MeV'.format(hbarc**2 / (2 * red_mass) * ev_full))
    print('E_3s1 = {} MeV\n'.format(hbarc**2 / (2 * red_mass) * ev))


potential.plot(full_potential, -2.0, 2.0)
