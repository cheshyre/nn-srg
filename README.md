# nn_srg

Similarity Renormalization Group (SRG) evolution for nucleon-nucleon interactions in momentum space.

## Overview

`nn_srg` provides tools for loading, manipulating, and evolving nucleon-nucleon potentials using the Similarity Renormalization Group method. The package handles both uncoupled and coupled-channel potentials with support for standard momentum-space representations.

**Key Features:**
- Load nucleon-nucleon potentials (AV18, EM500) in partial wave channels
- Evolve potentials using SRG flow equations
- Compute ground-state energies and scattering phase shifts
- Handle coupled channels (e.g., ³S₁-³D₁ channel)

## Installation

```bash
# Using Poetry (recommended)
poetry install

# Or using pip
pip install -e .
```

## Quick Start

```python
from nn_srg.potential import load_1S0_potential, load_3S1_3D1_potential
from nn_srg.srg import SRG
from nn_srg.diagonalize import diagonalize_hamiltonian

# Load potential
pot = load_3S1_3D1_potential('EM500')

# Evolve with SRG to λ = 2.0 fm⁻¹
srg = SRG(pot)
srg.evolve(lam=2.0)
evolved_pot = srg.get_potential()

# Compute deuteron binding energy
energy = diagonalize_hamiltonian(evolved_pot, num_eigvals=1)
print(f"Binding energy: {-energy[0]:.3f} MeV")
```

## Package Structure

```
nn_srg/
├── potential.py     # Potential loading and representation
├── srg.py           # SRG evolution
├── diagonalize.py   # Bound-state calculations
├── phase_shift.py   # Scattering phase shifts
├── constants.py     # Physical constants
└── potentials/      # Potential data files
    └── NN/
        ├── AV18/    # Argonne v18 potential
        └── EM500/   # Entem-Machleidt N³LO potential
```

## Examples

Example scripts for SRG evolution and visualization are provided in the `exec/` directory:
- `evolve_1S0.py` - Evolve ¹S₀ channel potential
- `evolve_3S1_3D1.py` - Evolve coupled ³S₁-³D₁ potential
- `plot_1S0_contour.py` - Visualize potential matrices

## Literature References

**SRG:**
- Bogner, S. K., Furnstahl, R. J., Perry, R. J. (2007). "Similarity renormalization group for nucleon-nucleon interactions." [Physical Review C 75, 061001](https://doi.org/10.1103/PhysRevC.75.061001).

**Nuclear Forces:**
- Machleidt, R., Entem, D. R. (2011). "Chiral effective field theory and nuclear forces." [Physics Reports 503, 1](https://doi.org/10.1016/j.physrep.2011.02.001).
- `EM500`: Entem, D. R., Machleidt, R. (2003). "Accurate charge-dependent nucleon-nucleon potential at fourth order of chiral perturbation theory." [Physical Review C 68, 041001(R)](https://doi.org/10.1103/PhysRevC.68.041001).
- `AV18`: Wiringa, R. B., Stoks, V. G. J., Schiavilla, R. (1995). "Accurate nucleon-nucleon potential with charge-independence breaking." [Physical Review C 51, 38](https://doi.org/10.1103/PhysRevC.51.38).
