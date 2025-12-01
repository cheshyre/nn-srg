# pylint: skip-file
"""Compute deuteron lowest energy state in 3s1 3d1 coupled channel (bound).

Computes lowest energy eigenvalue for a set of reference SRG evolved potentials
as well as the manually SRG evolved potential to show the implementation is
numerically equivalent to the reference implementation.

"""
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


def _get_coupled_channel_hamiltonian(channels):
    chan_dim = channels[0].dim
    dim = int(sqrt(len(channels)))
    hamiltonian = np.zeros((dim * chan_dim, dim * chan_dim))

    for i in range(dim):
        for j in range(dim):
            chan = channels[i * dim + j]
            if i == j:
                chan_hamiltonian = chan.with_weights() + chan.kinetic_energy()
            else:
                chan_hamiltonian = chan.with_weights()
            r_s = i * chan_dim
            r_e = (i + 1) * chan_dim
            c_s = j * chan_dim
            c_e = (j + 1) * chan_dim
            hamiltonian[r_s:r_e, c_s:c_e] = chan_hamiltonian
    return hamiltonian


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

# Compute reference Hamiltonian
hamiltonian_ref = _get_coupled_channel_hamiltonian([chan_3s1, chan_3s1_3d1,
                                                    chan_3d1_3s1, chan_3d1])

# Create coupled channel potential from channels
c_potential = potential.CoupledPotential([chan_3s1, chan_3s1_3d1, chan_3d1_3s1,
                                          chan_3d1])

# Compute alternate Hamiltonian
hamiltonian_coupled = c_potential.with_weights() + c_potential.kinetic_energy()

# Compute bound state eigenvalues
ev_ref = np.amin(eigh(hamiltonian_ref)[0])
ev = np.amin(eigh(hamiltonian_coupled)[0])

# Print unevolved results
print('Unevolved')
print('E_ref = E_srg = {} MeV'.format(hbarc**2 / (2 * red_mass) * ev_ref))
print('E_alt = {} MeV\n'.format(hbarc**2 / (2 * red_mass) * ev))

# Set up SRG evolution
v_mask = np.array([[0 for _ in range(len(c_potential.nodes))] for _ in
                   range(len(c_potential.nodes))])
k_mask = np.array([[1 for _ in range(len(c_potential.nodes))] for _ in
                   range(len(c_potential.nodes))])

srg_obj = srg.SRG(c_potential, v_mask, k_mask)

# Create list of lambdas to which to evolve
lambdas = [25] + list(range(10, 4, -1)) + list(np.arange(4, 3.1, -0.5)) \
    + list(np.arange(3.0, 2.5, -0.2)) + list(np.arange(2.4, 1.38, -0.05))
num_pts = [100]*9 + [83] + [79] + [76] + [73] + [72]*2 + [71] + [70]*2 + [69] \
    + [68]*2 + [67]*2 + [66] + [65]*2 + [64]*2 + [63] + [62]*2 + [61]*2

for l, n in zip(lambdas, num_pts):
    # Evolve potential
    srg_obj.evolve(l, verbose=False, integrator='dopri5', atol=10**(-6),
                   rtol=10**(-6), nsteps=10**(5))

    # Extract evolved potential
    c_potential = srg_obj.get_potential()

    # Reduce dimension if necessary
    if n < c_potential.dim:
        c_potential = c_potential.reduce_dim(n)
        v_mask = np.array([[0 for _ in range(len(c_potential.nodes))] for _ in
                           range(len(c_potential.nodes))])
        k_mask = np.array([[1 for _ in range(len(c_potential.nodes))] for _ in
                           range(len(c_potential.nodes))])
        srg_obj.replace_potential(c_potential, v_mask, k_mask)

    # Compute Hamiltonian
    hamiltonian = c_potential.with_weights() + c_potential.kinetic_energy()

    # Load reference potential (calculated by different code)
    chan_3s1 = potential.load(2, 3, 'EM420new', '10010', l, 'np')
    chan_3s1_3d1 = potential.load(2, 3, 'EM420new', '10210', l, 'np')
    chan_3d1_3s1 = potential.load(2, 3, 'EM420new', '12010', l, 'np')
    chan_3d1 = potential.load(2, 3, 'EM420new', '12210', l, 'np')

    # Compute reference Hamiltonian
    hamiltonian_ref = _get_coupled_channel_hamiltonian([chan_3s1, chan_3s1_3d1,
                                                        chan_3d1_3s1,
                                                        chan_3d1])

    # Get lowest eigenvalues
    ev_ref = np.amin(eigh(hamiltonian_ref)[0])
    ev = np.amin(eigh(hamiltonian)[0])

    # Output values
    print('Lambda: {}'.format(l))
    print('E_ref = {} MeV'.format(hbarc**2 / (2 * red_mass) * ev_ref))
    print('E_srg = {} MeV\n'.format(hbarc**2 / (2 * red_mass) * ev))
