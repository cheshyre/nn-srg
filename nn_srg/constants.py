# Copyright (c) 2025 Matthias Heinz
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
"""Physical constants.

This module defines fundamental physical constants and derived quantities
commonly used in nuclear structure calculations. All values are in natural
units where ℏ = c = 1, with energies in MeV and lengths in fm.

Constants
---------
hbarc : float
    Reduced Planck constant times speed of light (MeV·fm).
proton_mass : float
    Proton rest mass (MeV/c²).
neutron_mass : float
    Neutron rest mass (MeV/c²).
nucleon_mass : float
    Average nucleon mass (MeV/c²).
red_mass : float
    Reduced mass for nucleon-nucleon system (MeV/c²).

References
----------
Physical constants are taken from:

- Particle Data Group (PDG), "Review of Particle Physics" (2024).
  https://pdg.lbl.gov/

- CODATA 2018 recommended values.
  https://doi.org/10.1103/RevModPhys.93.025010

"""

# Conversion constant: ℏc in MeV·fm
# PDG 2024: ℏc = 197.3269804 MeV·fm
hbarc: float = 197.3269804  # MeV·fm

# Nucleon masses in MeV/c²
# PDG 2024: proton mass = 938.27208816 MeV/c²
proton_mass: float = 938.27208816  # MeV/c²

# PDG 2024: neutron mass = 939.56542052 MeV/c²
neutron_mass: float = 939.56542052  # MeV/c²

# Average nucleon mass 
nucleon_mass: float = (proton_mass + neutron_mass) / 2  # MeV/c²

# Reduced mass for nucleon-nucleon system: μ = m₁m₂/(m₁+m₂)
red_mass: float = proton_mass * neutron_mass / (proton_mass + neutron_mass)  # MeV/c²