# pylint: skip-file
"""Save a set of potentials to the standard directory for future use."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import glob
import os

# Check if module is installed, otherwise load locally
try:
    import nn_srg.potential as potential
except ImportError:
    from context import nn_srg
    potential = nn_srg.potential


path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'potentials')
pattern = 'VNN_N3LO_EM550new_*_np.dat'

full_path = os.path.join(path, pattern)

for file in glob.glob(full_path):
    a = potential.load_from_file(file)
    potential.save(a)
