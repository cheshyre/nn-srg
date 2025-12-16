# Copyright (c) 2018-2025 Matthias Heinz
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
"""Nuclear potential module.

Module containing representations of 3D nucleon-nucleon potentials.

Also contains logic to read potentials from and save them to files with a standard
naming convention.

Classes
-------
Channel
    A container for the channel information for a potential. It has the following
    method::

        channel = Channel(spin, orb_ang_mom_1, orb_ang_mom_2, tot_ang_mom, isospin)

    These are also commonly read as S, L, L', J, and T.

CoupledChannel
    A container to handle coupled channels. It has the following method::

        channel = CoupledChannel(list_of_channels)

    All channels in coupled channel should have same S, J, and T.

PotentialType
    A container class to hold all the physical information about the potential. It
    has the following method::

        potential_type = PotentialType(name, channel, particles)

Potential
    Abstraction for the representation of a potential. Handles the logic of adding
    and removing weights. Can generate corresponding kinetic energy. It has the
    following methods::

        potential = Potential(potential_type, nodes, weights, potential, lam=50.0,
                              has_weights=False)
        kinetic_energy = potential.kinetic_energy()
        potential_data_wo_weights = potential.without_weights()
        potential_data_w_weights = potential.with_weights()
        new_potential = potential.copy(potential_data, lam)
        reduced_potential = potential.reduce_dim(dim)

CoupledPotential
    Abstraction for representation for potential of coupled channel. Handles logic
    of adding and removing weights. Can generate kinetic energy. It has the
    following methods::

        potential = CoupledPotential([potential1, potential2, potential3,
                                      potential4])
        kinetic_energy = potential.kinetic_energy()
        potential_data_wo_weights = potential.without_weights()
        potential_data_w_weights = potential.with_weights()
        new_potential = potential.copy(potential_data, lam)
        reduced_potential = potential.reduce_dim(dim)
        channel_potential = potential.extract_channel_potential(
            potential1.potential_type.channel
        )

Functions
---------
load_from_file(file_str, name, channel, particles, lam=None)
    Load a potential from a file. Requires that standard file-naming
    conventions have been followed.

load_1S0_potential(name)
    Load the ¹S₀ (spin-singlet, S-wave) nucleon-nucleon potential.

load_3S1_3D1_potential(name)
    Load the coupled ³S₁-³D₁ (spin-triplet, coupled S-D wave) potential.

fast_and_lazy_plot(potential, v_scale=1.0)
    Quick visualization of a potential matrix with colorbar.

"""
from math import pi
from math import sqrt
import os
from typing import List, Tuple, Optional, Union

import numpy as np
import matplotlib.pyplot as plt

# Standard directory path for storing/loading potentials
STANDARD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'potentials')


class Channel:
    """Container for information on channel for potential.
    
    Represents a partial wave channel characterized by quantum numbers that
    describe the two-body system: spin (S), orbital angular momenta (L, L'),
    total angular momentum (J), and isospin (T).
    """

    def __init__(
        self,
        spin: int,
        orb_ang_mom_1: int,
        orb_ang_mom_2: int,
        tot_ang_mom: int,
        isospin: int
    ) -> None:
        """Create Channel object.

        Parameters
        ----------
        spin : int
            Spin quantum number (S).
        orb_ang_mom_1 : int
            Outgoing orbital angular momentum quantum number (L).
        orb_ang_mom_2 : int
            Incoming orbital angular momentum quantum number (L').
        tot_ang_mom : int
            Total angular momentum (J).
        isospin : int
            2-body isospin quantum number (T).

        """
        self._spin = spin
        self._l1 = orb_ang_mom_1
        self._l2 = orb_ang_mom_2
        self._j = tot_ang_mom
        self._isospin = isospin

    def as_5tuple(self) -> Tuple[int, int, int, int, int]:
        """Return 5-tuple representation of channel.

        Returns
        -------
        Tuple[int, int, int, int, int]
            5-tuple with channel quantum numbers in order (S, L, L', J, T).

        """
        return (self._spin, self._l1, self._l2, self._j, self._isospin)

    def __str__(self) -> str:
        """Return string representation of channel.

        Returns
        -------
        str
            String of 5 integers with channel information which are SLLJT.
            For example, "10001" represents S=1, L=0, L'=0, J=0, T=1.

        """
        return '{}{}{}{}{}'.format(self._spin, self._l1, self._l2, self._j,
                                   self._isospin)

    def __eq__(self, other: object) -> bool:
        """Return whether channel is same as another channel object.

        Parameters
        ----------
        other : object
            Another Channel object to compare with.

        Returns
        -------
        bool
            True if self and other have identical quantum numbers, False otherwise.

        """
        if not isinstance(other, Channel):
            return False
        return self.as_5tuple() == other.as_5tuple()

    def __ne__(self, other: object) -> bool:
        """Return whether channel is different from another channel object.

        Parameters
        ----------
        other : object
            Another Channel object to compare with.

        Returns
        -------
        bool
            False if self and other have identical quantum numbers, True otherwise.

        """
        return not self.__eq__(other)


class CoupledChannel(Channel):
    """Container for information about coupled channel.
    
    Represents a coupled channel.
    All channels in a coupled channel must
    share the same spin (S), total angular momentum (J), and isospin (T).
    """

    def __init__(self, list_of_channels: List[Channel]) -> None:
        """Create coupled channel container.

        Parameters
        ----------
        list_of_channels : List[Channel]
            List of Channel objects to be coupled. All channels must have
            identical S, J, and T quantum numbers.

        Raises
        ------
        ValueError
            If the given channels do not have matching S, J, and T values.

        """
        # Extract unique values of S, J, T from all channels
        spins = {x.as_5tuple()[0] for x in list_of_channels}
        tot_ang_moms = {x.as_5tuple()[3] for x in list_of_channels}
        isospins = {x.as_5tuple()[4] for x in list_of_channels}
        
        # Check that S, J, T are consistent across all channels
        if len(spins) * len(isospins) * len(tot_ang_moms) != 1:
            raise ValueError('Given channels cannot be coupled.')
        
        # Initialize parent Channel with '*' for L values (not single-valued in coupled channel)
        super(CoupledChannel, self).__init__(spins.pop(), '*', '*',
                                             tot_ang_moms.pop(),
                                             isospins.pop())
        self._channels = list_of_channels

    @property
    def channels(self) -> List[Channel]:
        """Return list of channels in coupled channel.

        Returns
        -------
        List[Channel]
            List of individual Channel objects that make up this coupled channel.

        """
        return self._channels

    def __eq__(self, other: object) -> bool:
        """Return whether coupled channel object is same as another.

        Parameters
        ----------
        other : object
            Another CoupledChannel object to compare with.

        Returns
        -------
        bool
            True if all constituent channels are equal, False otherwise.

        """
        if not isinstance(other, CoupledChannel):
            return False
        # Check that all channels match pairwise
        return False not in {x == y for x, y in zip(self.channels,
                                                    other.channels)}

    def __ne__(self, other: object) -> bool:
        """Return whether coupled channel object is not same as another.

        Parameters
        ----------
        other : object
            Another CoupledChannel object to compare with.

        Returns
        -------
        bool
            True if any constituent channels differ, False otherwise.

        """
        return not self.__eq__(other)


class PotentialType:
    """Container for metadata information related to a potential.
    
    Stores the information about a potential,
    including its name, channel structure, and particle content.
    """

    def __init__(self, name: str, channel: Union[Channel, CoupledChannel], particles: str) -> None:
        """Construct potential type.

        Parameters
        ----------
        name : str
            Name for potential, may reflect information about its origin
            (e.g., "EM500", "Av18").
        channel : Union[Channel, CoupledChannel]
            Object representing the partial wave channel(s) for the potential.
        particles : str
            String representing constituent particles in the interaction
            (e.g., "np" for neutron-proton, "pp" for proton-proton).

        """
        self._name = name
        self._channel = channel
        self._particles = particles

    @property
    def name(self) -> str:
        """Return name of potential.

        Returns
        -------
        str
            The name for this potential.

        """
        return self._name

    @property
    def channel(self) -> Union[Channel, CoupledChannel]:
        """Return channel of potential.

        Returns
        -------
        Union[Channel, CoupledChannel]
            The channel or coupled channel associated with this potential.

        """
        return self._channel

    @property
    def particles(self) -> str:
        """Return string of particles in potential.

        Returns
        -------
        str
            String identifier for the particles in the interaction.

        """
        return self._particles

    def __str__(self) -> str:
        """Return human-readable string representation of potential type.

        Returns
        -------
        str
            String combining name, channel, and particles information.

        """
        return '{}_{}'.format(self._name, str(self._channel))


class Potential:
    """Abstraction for the representation of a nuclear potential.
    
    This class handles the storage and manipulation of nuclear interaction
    potentials in momentum space. It manages the relationship between weighted
    and unweighted representations, can generate kinetic energy matrices, and
    supports dimension reduction operations.
    
    The potential is represented on a discrete momentum grid defined by nodes
    (momentum values) and weights (integration weights), typically from
    Gauss-Legendre quadrature.
    """

    def __init__(
        self,
        potential_type: PotentialType,
        nodes: Union[List[float], np.ndarray],
        weights: Union[List[float], np.ndarray],
        potential: np.ndarray,
        lam: float = 50.0,
        has_weights: bool = False
    ) -> None:
        """Create Potential object.

        Parameters
        ----------
        potential_type : PotentialType
            Object containing metadata about the potential.
        nodes : Union[List[float], np.ndarray]
            Momentum grid points (fm^-1).
        weights : Union[List[float], np.ndarray]
            Integration weights corresponding to nodes.
        potential : np.ndarray
            Potential matrix data (dimension: len(nodes) × len(nodes)).
        lam : float, optional
            SRG flow parameter lambda (fm^-1), by default 50.0.
            Values of 50.0 indicate unevolved potential.
        has_weights : bool, optional
            Whether the potential matrix already includes weight factors,
            by default False.

        """
        self._potential_type = potential_type
        self._nodes = np.array(nodes)
        self._weights = np.array(weights)
        self._lam = lam
        
        # Store potential in unweighted form internally
        if has_weights:
            self._potential = _rem_w(potential, weights, nodes)
        else:
            self._potential = potential

    @property
    def potential_type(self) -> PotentialType:
        """Return potential type.

        Returns
        -------
        PotentialType
            Metadata object describing this potential.

        """
        return self._potential_type

    @property
    def lam(self) -> float:
        """Return lambda value of the potential.

        Returns
        -------
        float
            SRG flow parameter lambda (fm^-1).

        """
        return self._lam

    @property
    def nodes(self) -> np.ndarray:
        """Return nodes of the potential.

        Returns
        -------
        np.ndarray
            Momentum grid points (fm^-1).

        """
        return self._nodes

    @property
    def weights(self) -> np.ndarray:
        """Return weights of the potential.

        Returns
        -------
        np.ndarray
            Integration weights corresponding to momentum grid.

        """
        return self._weights

    @property
    def dim(self) -> int:
        """Return the dimension of the potential matrix.

        Returns
        -------
        int
            The dimension N of the (N x N) potential matrix.

        """
        return len(self._nodes)

    def kinetic_energy(self) -> np.ndarray:
        """Generate kinetic energy matrix.

        Creates the free two-body kinetic energy operator in momentum space,
        which is diagonal with entries T = p^2/m (reduced mass).

        Returns
        -------
        np.ndarray
            Diagonal kinetic energy matrix (MeV).

        """
        # Kinetic energy in momentum space is p^2/m on the diagonal
        return np.diag([p * p for p in self._nodes])

    def without_weights(self) -> np.ndarray:
        """Return potential without weight factors.

        Returns
        -------
        np.ndarray
            Unweighted potential matrix V(p, p') (MeV).

        """
        return self._potential

    def with_weights(self) -> np.ndarray:
        """Return potential with weight factors included.

        Applies weight factors w(p)^(1/2) * p and w(p')^(1/2) * p' to construct
        the weighted potential matrix used in discretized Lippmann-Schwinger
        equation: (2/π) * sqrt(w*w') * p * p' * V(p, p').

        Returns
        -------
        np.ndarray
            Weighted potential matrix for use in integral equations.

        """
        return _add_w(self._potential, self._weights, self._nodes)

    def copy(self, potential: np.ndarray, lam: float) -> 'Potential':
        """Create new potential from current one with updated data and lambda.

        Parameters
        ----------
        potential : np.ndarray
            New potential matrix data (unweighted).
        lam : float
            New SRG lambda value (fm^-1).

        Returns
        -------
        Potential
            New Potential object with updated matrix and lambda but same
            grid points, weights, and metadata.

        """
        return Potential(self._potential_type, self._nodes, self._weights,
                        potential, lam)

    def reduce_dim(self, dim: int) -> 'Potential':
        """Return new potential with reduced momentum space dimension.

        Creates a new potential by keeping only the lowest `dim` momentum
        states, effectively truncating the high-momentum content.

        Parameters
        ----------
        dim : int
            Target dimension (must be less than current dimension and positive).

        Returns
        -------
        Potential
            New reduced-dimension potential.

        Raises
        ------
        ValueError
            When dim is not smaller than current dimension or is non-positive.

        """
        if dim >= len(self._nodes):
            raise ValueError('Value of dim is not smaller than current dim.')
        if dim <= 0:
            raise ValueError('Zero or negative dim is not allowed.')
        
        # Extract sub-blocks of the potential, nodes, and weights
        return Potential(self._potential_type, self._nodes[:dim],
                        self._weights[:dim], self._potential[:dim, :dim],
                        self._lam)


class CoupledPotential(Potential):
    """Representation for potential of coupled channel system.
    
    Manages potentials for coupled channels, where multiple partial waves
    are mixed due to tensor forces. The coupled potential is stored as a
    block matrix where each block corresponds to a channel-channel coupling.
    
    For example, the ^3S_1-^3D_1 system has a 2x2 block structure:
    [ V(^3S_1, ^3S_1)   V(^3S_1, ^3D_1) ]
    [ V(^3D_1, ^3S_1)   V(^3D_1, ^3D_1) ]
    """

    def __init__(self, list_of_potentials: List[Potential]) -> None:
        """Create coupled potential from individual channel potentials.

        Parameters
        ----------
        list_of_potentials : List[Potential]
            List of Potential objects to couple. Must be provided in row-major
            order for the block matrix. The number of potentials must be a
            perfect square (e.g., 4 for 2x2 coupling)

        Raises
        ------
        ValueError
            If potentials cannot be coupled (different names/particles/lambda),
            have inconsistent dimensions, or number is not a perfect square.

        """
        # Store construction potentials for later use in copy() and reduce_dim()
        self._construction = list_of_potentials
        
        # Extract and validate channel structure
        channels = [x.potential_type.channel for x in list_of_potentials]
        name = {x.potential_type.name for x in list_of_potentials}
        particles = {x.potential_type.particles for x in list_of_potentials}
        
        # Ensure all potentials share same name and particle type
        if len(name) * len(particles) != 1:
            raise ValueError('Given potentials cannot be coupled.')
        
        # Create coupled channel from individual channels
        coupled_channel = CoupledChannel(channels)
        potential_type = PotentialType(name.pop(),
                                       coupled_channel, particles.pop())
        
        # Validate that all potentials have same lambda value
        lam = {x.lam for x in list_of_potentials}
        if len(lam) != 1:
            raise ValueError('Not all given potentials are at the same lam.')
        lam = lam.pop()
        
        # Validate that all potentials have same dimension
        dim = {x.dim for x in list_of_potentials}
        if len(dim) != 1:
            raise ValueError('Not all given potentials have same dim.')
        dim = dim.pop()
        
        # Determine coupling structure (c_dim × c_dim blocks of dim × dim)
        c_dim = int(sqrt(len(list_of_potentials)))
        if c_dim**2 != len(list_of_potentials):
            raise ValueError('Non-square number of potentials given.')
        
        # Extract nodes and weights from first row of potentials
        # (all should be identical, so we just use the first c_dim)
        nodes = []
        weights = []
        for pot in list_of_potentials[:c_dim]:
            nodes += pot.nodes.tolist()
            weights += pot.weights.tolist()
        nodes = np.array(nodes)
        weights = np.array(weights)
        
        # Assemble block matrix from individual channel potentials
        potential_data = np.zeros((c_dim * dim, c_dim * dim))
        self._channel_indexes = []
        
        # Fill each block in row-major order
        for i in range(c_dim):
            for j in range(c_dim):
                # Calculate row and column indices for this block
                r_s = i * dim  # row start
                r_e = (i + 1) * dim  # row end
                c_s = j * dim  # column start
                c_e = (j + 1) * dim  # column end
                
                # Extract unweighted potential and place in block
                data = list_of_potentials[i * c_dim + j].without_weights()
                potential_data[r_s:r_e, c_s:c_e] = data
                
                # Store indices for later extraction
                self._channel_indexes.append((r_s, r_e, c_s, c_e))
        
        # Initialize parent Potential class
        super(CoupledPotential, self).__init__(potential_type, nodes, weights,
                                               potential_data, lam)
        self._c_dim = c_dim  # number of channels
        self._w_dim = dim  # dimension within each channel
        self._channels = channels

    def copy(self, potential: np.ndarray, lam: float) -> 'CoupledPotential':
        """Create coupled potential from current one with new data and lambda.

        Parameters
        ----------
        potential : np.ndarray
            New potential matrix data (full block matrix, unweighted).
        lam : float
            New SRG lambda value (fm^-1).

        Returns
        -------
        CoupledPotential
            New CoupledPotential object with updated matrix and lambda.

        """
        new_potentials = []
        # Extract each block and create new Potential object
        for pot, ranges in zip(self._construction, self._channel_indexes):
            sub_matrix = _submatrix(potential, ranges)
            new_potentials.append(pot.copy(sub_matrix, lam))
        return CoupledPotential(new_potentials)

    def reduce_dim(self, dim: int) -> 'CoupledPotential':
        """Return new potential with reduced dimension in each channel.

        Creates a new coupled potential by reducing the dimension within each
        individual channel block to `dim` lowest momentum states.

        Parameters
        ----------
        dim : int
            Target dimension for each channel (must be less than current
            channel dimension and positive).

        Returns
        -------
        CoupledPotential
            New reduced-dimension coupled potential.

        Raises
        ------
        ValueError
            When dim is not smaller than current channel dimension or is
            non-positive.

        """
        if dim >= self._w_dim:
            raise ValueError('Value of dim is not smaller than current dim.')
        if dim <= 0:
            raise ValueError('Zero or negative dim is not allowed.')
        
        new_potentials = []
        # Reduce dimension of each channel block
        for pot, ranges in zip(self._construction, self._channel_indexes):
            sub_matrix = _submatrix(self._potential, ranges)
            new_potentials.append(pot.copy(sub_matrix,
                                           self._lam).reduce_dim(dim))
        return CoupledPotential(new_potentials)

    def extract_channel_potential(self, channel: Channel) -> Potential:
        """Return potential corresponding to a specific channel.

        Extracts the diagonal block corresponding to the requested channel
        from the coupled potential matrix.

        Parameters
        ----------
        channel : Channel
            Channel to extract (must be one of the coupled channels).

        Returns
        -------
        Potential
            Potential object for the specified channel.

        Raises
        ------
        ValueError
            If the requested channel is not found in the coupled system.

        """
        # Search for matching channel and extract its diagonal block
        for chan, potential, ranges in zip(self._channels, self._construction,
                                           self._channel_indexes):
            if channel == chan:
                sub_matrix = _submatrix(self._potential, ranges)
                return potential.copy(sub_matrix, self._lam)
        raise ValueError('Channel not found.')

    @property
    def dim(self) -> int:
        """Return the dimension of single channel in the potential matrix.

        Returns
        -------
        int
            The dimension of a single channel block in the coupled potential.
            The full matrix dimension is c_dim * dim where c_dim is the
            number of coupled channels.

        """
        return self._w_dim


def load_from_file(
    file_str: str,
    name: str,
    channel: Channel,
    particles: str,
    lam: Optional[float] = None
) -> Potential:
    """Load potential from file.

    Reads potential data from a file with the standard format:
    - First N lines: weight and node pairs
    - Remaining NxN lines: potential matrix elements

    Parameters
    ----------
    file_str : str
        Path to file with potential data.
    name : str
        Name for potential, may reflect something about origin.
    channel : Channel
        Object representing the partial wave channel for the potential.
    particles : str
        String representing constituent particles in the interaction.
    lam : Optional[float], optional
        SRG lambda value of potential (fm^-1), by default None.
        None indicates unevolved potential (treated as 50.0).

    Returns
    -------
    Potential
        Potential object created from file data.

    """
    # Parse info about potential from filename
    # Set default lambda for unevolved potentials
    if lam is None:
        lam = 50.0
    
    # First pass: count number of grid points
    with open(file_str) as file:
        num_points = 0
        for line in file:
            if len(line.strip().split()) == 2:
                num_points += 1
    
    # Second pass: read weights, nodes, and potential matrix
    with open(file_str) as file:
        nodes = []
        weights = []
        # Read weight-node pairs
        for _ in range(num_points):
            vals = file.readline().split()
            weights.append(float(vals[0]))
            nodes.append(float(vals[1]))
        
        # Read potential matrix (last element of each line is the matrix entry)
        potential = np.array([[float(file.readline().split()[-1]) for _ in
                               range(num_points)] for _ in range(num_points)])

    # Create potential_type object with metadata
    potential_type = PotentialType(name, channel, particles)

    # Return potential
    return Potential(potential_type, nodes, weights, potential, lam)


def load_1S0_potential(name: str) -> Potential:
    """Load singlet S-wave (^1S_0) neutron-proton potential.

    Convenience function for loading the uncoupled ^1S_0 channel, which is
    important for neutron-proton scattering and nuclear matter.

    Parameters
    ----------
    name : str
        Name of the potential to load (must exist in STANDARD_PATH/NN/name/).

    Returns
    -------
    Potential
        The ^1S_0 neutron-proton potential.

    """
    # ^1S_0 channel: S=0, L=0, L'=0, J=0, T=1
    chan = Channel(spin=0, orb_ang_mom_1=0, orb_ang_mom_2=0, tot_ang_mom=0, isospin=1)
    chan_str = str(chan) + "_np"
    path = os.path.join(STANDARD_PATH, "NN", name, f"SLLJT_{chan_str}.dat")

    return load_from_file(path, name, chan, "np")


def load_3S1_3D1_potential(name: str) -> CoupledPotential:
    """Load coupled ^3S_1-^3D_1 neutron-proton potential.

    Loads the four components of the coupled channel system:
    - ^3S_1-^3S_1: S-wave to S-wave
    - ^3S_1-^3D_1: S-wave to D-wave (tensor coupling)
    - ^3D_1-^3S_1: D-wave to S-wave (tensor coupling)
    - ^3D_1-^3D_1: D-wave to D-wave

    This coupled channel is crucial for deuteron binding and triplet np scattering.

    Parameters
    ----------
    name : str
        Name of the potential to load (must exist in STANDARD_PATH/NN/name/).

    Returns
    -------
    CoupledPotential
        The ^3S_1-^3D_1 coupled channel potential.

    """
    pot_list = []

    # Load all four blocks of the coupled channel potential
    # (S=1, L, L', J=1, T=0) for all combinations of L, L' ∈ {0, 2}
    for l1, l2 in [
        (0, 0),  # ^3S_1-^3S_1
        (0, 2),  # ^3S_1-^3D_1
        (2, 0),  # ^3D_1-^3S_1
        (2, 2),  # ^3D_1-^3D_1
    ]:
        chan = Channel(spin=1, orb_ang_mom_1=l1, orb_ang_mom_2=l2, tot_ang_mom=1, isospin=0)
        chan_str = str(chan) + "_np"
        path = os.path.join(STANDARD_PATH, "NN", name, f"SLLJT_{chan_str}.dat")

        pot_list.append(
            load_from_file(path, name, chan, "np")
        )
    
    return CoupledPotential(pot_list)


def fast_and_lazy_plot(potential: Potential, v_scale: float = 1.0) -> None:
    """Plot potential matrix with colorbar.

    Creates a quick visualization of the potential matrix useful for
    debugging and qualitative analysis. Shows momentum dependence V(p, p').

    Parameters
    ----------
    potential : Potential
        Potential object to be plotted.
    v_scale : float, optional
        Maximum magnitude for the symmetric colorbar scale (MeV),
        by default 1.0.

    """
    _, ax = plt.subplots()
    # Plot matrix with symmetric colorbar centered at zero
    im = ax.matshow(potential.without_weights(), vmin=-1 * v_scale, vmax=v_scale, cmap=plt.cm.RdBu_r,)
    
    # Set up axis labels and ticks
    nodes = potential.nodes
    steps = 20  # show tick every 20 points
    plt.xticks([x for x in range(0, len(nodes), steps)])
    ax.set_xlabel(r"p (fm$^{-1}$)")
    ax.xaxis.set_label_position('top')
    ax.set_xticklabels(["{:.2f}".format(nodes[x]) for x in range(0, len(nodes), steps)])
    plt.ylabel(r"p' (fm$^{-1}$)")
    plt.yticks([x for x in range(0, len(nodes), steps)])
    ax.set_yticklabels(["{:.2f}".format(nodes[x]) for x in range(0, len(nodes), steps)])
    plt.colorbar(im)
    plt.show()
    plt.close()


# ------------------- Internal Methods ------------------------------------- #


def _add_w(matrix: np.ndarray, weights: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Add weight factors to potential matrix.

    Transforms unweighted potential V(p, p') to weighted form used in
    discretized integral equations: (2/π) * sqrt(w*w') * p * p' * V(p, p').

    Parameters
    ----------
    matrix : np.ndarray
        Unweighted potential matrix.
    weights : np.ndarray
        Integration weights.
    nodes : np.ndarray
        Momentum grid points.

    Returns
    -------
    np.ndarray
        Weighted potential matrix.

    """
    # Construct diagonal matrix of sqrt(w) * p factors
    factor_vector = [sqrt(w) * p for w, p in zip(weights, nodes)]
    
    # Apply: weighted = sqrt(w) * p * V * sqrt(w') * p'
    weighted_matrix = np.dot(np.dot(np.diag(factor_vector), matrix),
                             np.diag(factor_vector))
    
    # Include (2/π) normalization factor
    return 2 / pi * weighted_matrix


def _rem_w(matrix: np.ndarray, weights: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Remove weight factors from potential matrix.

    Transforms weighted potential back to unweighted form V(p, p').

    Parameters
    ----------
    matrix : np.ndarray
        Weighted potential matrix.
    weights : np.ndarray
        Integration weights.
    nodes : np.ndarray
        Momentum grid points.

    Returns
    -------
    np.ndarray
        Unweighted potential matrix.

    """
    # Construct diagonal matrix of 1/(sqrt(w) * p) factors
    factor_vector = [1/(sqrt(w) * p) for w, p in zip(weights, nodes)]
    
    # Apply inverse transformation: V = (1/(sqrt(w)*p)) * weighted * (1/(sqrt(w')*p'))
    unweighted_matrix = np.dot(np.dot(np.diag(factor_vector), pi / 2 * matrix),
                               np.diag(factor_vector))
    
    return unweighted_matrix


def _submatrix(potential: np.ndarray, ranges: Tuple[int, int, int, int]) -> np.ndarray:
    """Extract submatrix from potential using index ranges.

    Helper function to extract block submatrices from coupled potentials.

    Parameters
    ----------
    potential : np.ndarray
        Full potential matrix.
    ranges : Tuple[int, int, int, int]
        Index ranges (row_start, row_end, col_start, col_end) for extraction.

    Returns
    -------
    np.ndarray
        Extracted submatrix potential[row_start:row_end, col_start:col_end].

    """
    return potential[np.ix_(list(range(ranges[0], ranges[1])),
                            list(range(ranges[2], ranges[3])))]
