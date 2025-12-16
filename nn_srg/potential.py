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
from math import pi, sqrt
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

STANDARD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "potentials")


class Channel:
    """Container for information on channel for potential.

    A channel is defined by the quantum numbers S (spin),
    L (outgoing relative orbital angular momentum),
    L' (incoming relative orbital angular momentum),
    J (total angular momentum), and T (isospin).
    """

    def __init__(
        self,
        spin: int,
        orb_ang_mom_1: int,
        orb_ang_mom_2: int,
        tot_ang_mom: int,
        isospin: int,
    ):
        """Create Channel object.

        Parameters
        ----------
        spin : int
            Spin quantum number S.
        orb_ang_mom_1 : int
            Outgoing relative orbital angular momentum quantum number L.
        orb_ang_mom_2 : int
            Incoming relative orbital angular momentum quantum number L'.
        tot_ang_mom : int
            Total angular momentum J.
        isospin : int
            2-body isospin quantum number T.

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
        tuple of int
            5-tuple with channel quantum numbers (S, L, L', J, T).

        """
        return (self._spin, self._l1, self._l2, self._j, self._isospin)

    def __str__(self) -> str:
        """Return string representation of channel.

        Returns
        -------
        str
            String of 5 integers with channel information in format SLLJT.

        """
        return "{}{}{}{}{}".format(
            self._spin, self._l1, self._l2, self._j, self._isospin
        )

    def __eq__(self, other) -> bool:
        """Return whether channel is same as another channel object.

        Parameters
        ----------
        other : Channel
            Channel object to compare with.

        Returns
        -------
        bool
            True if self and other are the same, False otherwise.

        """
        return self.as_5tuple() == other.as_5tuple()

    def __ne__(self, other) -> bool:
        """Return whether channel is different from another channel object.

        Parameters
        ----------
        other : Channel
            Channel object to compare with.

        Returns
        -------
        bool
            True if channels are different, False otherwise.

        """
        return self.as_5tuple() != other.as_5tuple()


class CoupledChannel(Channel):
    """Container for information about coupled channel.

    A coupled channel consists of multiple channels that are coupled together,
    typically arising from tensor forces. All channels must share the same
    spin S, total angular momentum J, and isospin T.
    """

    def __init__(self, list_of_channels: List[Channel]):
        """Create coupled channel container.

        Parameters
        ----------
        list_of_channels : list of Channel
            List of Channel objects in coupled channel.

        Raises
        ------
        ValueError
            If channels cannot be coupled (different S, J, or T values).

        """
        spins = {x.as_5tuple()[0] for x in list_of_channels}
        tot_ang_moms = {x.as_5tuple()[3] for x in list_of_channels}
        isospins = {x.as_5tuple()[4] for x in list_of_channels}
        if len(spins) * len(isospins) * len(tot_ang_moms) != 1:
            raise ValueError("Given channels cannot be coupled.")
        super(CoupledChannel, self).__init__(
            spins.pop(), "*", "*", tot_ang_moms.pop(), isospins.pop()
        )
        self._channels = list_of_channels

    @property
    def channels(self) -> List[Channel]:
        """Return list of channels in coupled channel.

        Returns
        -------
        list of Channel
            List of Channel objects comprising the coupled channel.

        """
        return self._channels

    def __eq__(self, other) -> bool:
        """Return whether coupled channel object is same as another.

        Parameters
        ----------
        other : CoupledChannel
            CoupledChannel object to compare with.

        Returns
        -------
        bool
            True if coupled channels are equal, False otherwise.

        """
        return False not in {x == y for x, y in zip(self.channels, other.channels)}

    def __ne__(self, other) -> bool:
        """Return whether coupled channel object is not same as another.

        Parameters
        ----------
        other : CoupledChannel
            CoupledChannel object to compare with.

        Returns
        -------
        bool
            True if coupled channels are not equal, False otherwise.

        """
        return False in {x == y for x, y in zip(self.channels, other.channels)}


class PotentialType:
    """Container for information related to potential.

    Stores metadata about a potential including its name, channel information,
    and constituent particles.
    """

    def __init__(
        self, name: str, channel: Union[Channel, CoupledChannel], particles: str
    ):
        """Construct potential type.

        Parameters
        ----------
        name : str
            Name for potential, may reflect something about origin (e.g., 'EM500', 'AV18').
        channel : Channel or CoupledChannel
            Object representing the partial wave channel for the potential.
        particles : str
            String representing constituent particles in the interaction (e.g., 'np', 'nn', 'pp').

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
            Name of potential.

        """
        return self._name

    @property
    def channel(self) -> Union[Channel, CoupledChannel]:
        """Return channel of potential.

        Returns
        -------
        Channel or CoupledChannel
            Channel information for the potential.

        """
        return self._channel

    @property
    def particles(self) -> str:
        """Return particles in potential.

        Returns
        -------
        str
            String representing constituent particles.

        """
        return self._particles


class Potential:
    """Abstraction for representation of a potential.

    Handles momentum-space potentials with Gaussian quadrature weights.
    The potential can be stored in weighted or unweighted form, where
    weighted means the Gaussian quadrature weights and momentum factors
    are included in the matrix elements.

    Attributes
    ----------
    potential_type : PotentialType
        Metadata about the potential.
    nodes : array_like
        Momentum grid points (in fm⁻¹).
    weights : array_like
        Gaussian quadrature weights.
    lam : float
        SRG flow parameter λ (in fm⁻¹). Default 50.0 indicates unevolved potential.
    """

    def __init__(
        self,
        potential_type: PotentialType,
        nodes: np.ndarray,
        weights: np.ndarray,
        potential: np.ndarray,
        lam: float = 50.0,
        has_weights: bool = False,
    ):
        """Create potential.

        Parameters
        ----------
        potential_type : PotentialType
            Object with information about potential.
        nodes : array_like
            Momentum nodes for Gaussian quadrature (in fm⁻¹).
        weights : array_like
            Weights for Gaussian quadrature.
        potential : array_like
            2D array with potential matrix elements.
        lam : float, optional
            SRG flow parameter λ (in fm⁻¹). Default is 50.0, representing
            an unevolved potential.
        has_weights : bool, optional
            Whether potential matrix includes quadrature weights. Default is False.

        """
        self._potential_type = potential_type
        self._nodes = nodes
        self._weights = weights
        self._lam = lam
        if has_weights:
            self._w_potential = potential
        else:
            self._w_potential = _add_w(potential, weights, nodes)

        self._w_dim = len(weights)

    @property
    def potential_type(self) -> PotentialType:
        """Return potential type.

        Returns
        -------
        PotentialType
            Metadata about the potential.

        """
        return self._potential_type

    @property
    def nodes(self) -> np.ndarray:
        """Return momentum nodes.

        Returns
        -------
        ndarray
            Momentum grid points (in fm⁻¹).

        """
        return self._nodes

    @property
    def weights(self) -> np.ndarray:
        """Return quadrature weights.

        Returns
        -------
        ndarray
            Gaussian quadrature weights.

        """
        return self._weights

    @property
    def lam(self) -> float:
        """Return SRG flow parameter.

        Returns
        -------
        float
            SRG flow parameter λ (in fm⁻¹).

        """
        return self._lam

    def kinetic_energy(self) -> np.ndarray:
        """Generate kinetic energy matrix.

        Returns
        -------
        ndarray
            Diagonal matrix with kinetic energy values T = p²/(2μ),
            where μ is the reduced mass (assuming m_nucleon/2).

        """
        kinetic = np.diag([node**2 / 2 for node in self.nodes])
        return _add_w(kinetic, self.weights, self.nodes)

    def without_weights(self) -> np.ndarray:
        """Return potential matrix without quadrature weights.

        Returns
        -------
        ndarray
            Unweighted potential matrix V(p, p').

        """
        return _rem_w(self._w_potential, self.weights, self.nodes)

    def with_weights(self) -> np.ndarray:
        """Return potential matrix with quadrature weights.

        Returns
        -------
        ndarray
            Weighted potential matrix for direct use in integration.

        """
        return self._w_potential

    def copy(self, potential: np.ndarray, lam: float) -> "Potential":
        """Create copy of potential with new matrix and flow parameter.

        Parameters
        ----------
        potential : array_like
            New potential matrix (without weights).
        lam : float
            New SRG flow parameter λ (in fm⁻¹).

        Returns
        -------
        Potential
            New Potential object with updated matrix and λ.

        """
        return Potential(
            self.potential_type, self.nodes, self.weights, potential, lam, False
        )

    def reduce_dim(self, dim: int) -> "Potential":
        """Reduce dimensionality of potential.

        Parameters
        ----------
        dim : int
            New dimension (must be less than current dimension).

        Returns
        -------
        Potential
            Potential with reduced momentum grid.

        """
        return Potential(
            self.potential_type,
            self.nodes[:dim],
            self.weights[:dim],
            self.without_weights()[:dim, :dim],
            self.lam,
            False,
        )

    @property
    def w_dim(self) -> int:
        """Return weighted dimension of potential matrix.

        The weighted dimension accounts for the Gaussian quadrature weights
        used in momentum-space integration.

        Returns
        -------
        int
            Matrix dimension.

        """
        return self._w_dim


class CoupledPotential:
    """Abstraction for representation of coupled-channel potential.

    Handles potentials in coupled channels.
    The potential is represented as a block matrix with diagonal
    and off-diagonal blocks.

    For a coupled ³S₁-³D₁ channel, the structure is:
        [ V_SS  V_SD ]
        [ V_DS  V_DD ]
    """

    def __init__(self, potentials: List[Potential]):
        """Create coupled potential.

        Parameters
        ----------
        potentials : list of Potential
            List of Potential objects representing different blocks.
            For coupled S-D partial wave, order is [V_SS, V_SD, V_DS, V_DD].

        Raises
        ------
        ValueError
            If potentials cannot be coupled (incompatible channels).

        """
        self._potentials = potentials
        potential_type = potentials[0].potential_type
        if not isinstance(potential_type.channel, CoupledChannel):
            raise ValueError("Potentials are not in coupled channels.")

    def kinetic_energy(self) -> np.ndarray:
        """Generate kinetic energy matrix for coupled channel.

        Returns
        -------
        ndarray
            Block-diagonal kinetic energy matrix.

        """
        kinetics = [pot.kinetic_energy() for pot in self._potentials[::3]]
        return np.block(
            [[kinetics[0], 0 * kinetics[0]], [0 * kinetics[1], kinetics[1]]]
        )

    def without_weights(self) -> np.ndarray:
        """Return potential matrix without quadrature weights.

        Returns
        -------
        ndarray
            Unweighted coupled potential matrix.

        """
        unweighted_pots = [pot.without_weights() for pot in self._potentials]
        return np.block(
            [
                [unweighted_pots[0], unweighted_pots[1]],
                [unweighted_pots[2], unweighted_pots[3]],
            ]
        )

    def with_weights(self) -> np.ndarray:
        """Return potential matrix with quadrature weights.

        Returns
        -------
        ndarray
            Weighted coupled potential matrix.

        """
        weighted_pots = [pot.with_weights() for pot in self._potentials]
        return np.block(
            [[weighted_pots[0], weighted_pots[1]], [weighted_pots[2], weighted_pots[3]]]
        )

    def copy(self, potential: np.ndarray, lam: float) -> "CoupledPotential":
        """Create copy of coupled potential with new matrix and flow parameter.

        Parameters
        ----------
        potential : array_like
            New coupled potential matrix (without weights).
        lam : float
            New SRG flow parameter λ (in fm⁻¹).

        Returns
        -------
        CoupledPotential
            New CoupledPotential object with updated matrix and λ.

        """
        dim = len(self._potentials[0].nodes)
        pots = [
            self._potentials[i].copy(_submatrix(potential, ranges), lam)
            for i, ranges in enumerate(
                [
                    (0, dim, 0, dim),
                    (0, dim, dim, 2 * dim),
                    (dim, 2 * dim, 0, dim),
                    (dim, 2 * dim, dim, 2 * dim),
                ]
            )
        ]
        return CoupledPotential(pots)

    def reduce_dim(self, dim: int) -> "CoupledPotential":
        """Reduce dimensionality of coupled potential.

        Parameters
        ----------
        dim : int
            New dimension for each channel block.

        Returns
        -------
        CoupledPotential
            Coupled potential with reduced momentum grid.

        """
        return CoupledPotential([pot.reduce_dim(dim) for pot in self._potentials])

    def extract_channel_potential(self, channel: Channel) -> Potential:
        """Extract a specific channel potential from coupled potential.

        Parameters
        ----------
        channel : Channel
            Channel to extract.

        Returns
        -------
        Potential
            Potential for the specified channel.

        Raises
        ------
        ValueError
            If requested channel is not in coupled channel.

        """
        for pot in self._potentials:
            if pot.potential_type.channel == channel:
                return pot
        raise ValueError("Could not find potential for channel.")

    @property
    def w_dim(self) -> int:
        """Return weighted dimension of coupled potential matrix.

        Returns
        -------
        int
            Total matrix dimension (sum of individual channel dimensions).

        """
        return self._potentials[0].w_dim * 2


def load_from_file(
    file_str: str,
    name: str,
    channel: Channel,
    particles: str,
    lam: Optional[float] = None,
) -> Potential:
    """Load potential from file.

    The file format should have nodes and weights listed first (one per line),
    followed by the potential matrix elements.

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
    lam : float, optional
        SRG flow parameter λ (in fm⁻¹). Default is None, which sets λ = 50.0
        (effectively unevolved).

    Returns
    -------
    Potential
        Potential created from extracted information and data from file.

    """
    # Default lambda for unevolved potential
    if lam is None:
        lam = 50.0

    # Count number of nodes
    with open(file_str) as file:
        num_points = 0
        for line in file:
            if len(line.strip().split()) == 2:
                num_points += 1

    # Read nodes, weights, and potential matrix
    with open(file_str) as file:
        nodes = []
        weights = []
        for _ in range(num_points):
            vals = file.readline().split()
            weights.append(float(vals[0]))
            nodes.append(float(vals[1]))
        potential = np.array(
            [
                [float(file.readline().split()[-1]) for _ in range(num_points)]
                for _ in range(num_points)
            ]
        )

    # Create potential_type
    potential_type = PotentialType(name, channel, particles)

    # Return potential
    return Potential(potential_type, nodes, weights, potential, lam)


def load_1S0_potential(name: str) -> Potential:
    """Load the ¹S₀ nucleon-nucleon potential.

    The ¹S₀ channel is the spin-singlet S-wave channel (S=0, L=0, J=0, T=1).

    Parameters
    ----------
    name : str
        Name of the potential (e.g., 'EM500', 'AV18').

    Returns
    -------
    Potential
        The loaded ¹S₀ potential object.

    Examples
    --------
    >>> pot = load_1S0_potential('EM500')
    >>> print(pot.lam)
    50.0

    """
    chan = Channel(spin=0, orb_ang_mom_1=0, orb_ang_mom_2=0, tot_ang_mom=0, isospin=1)
    chan_str = str(chan) + "_np"
    path = os.path.join(STANDARD_PATH, "NN", name, f"SLLJT_{chan_str}.dat")

    return load_from_file(path, name, chan, "np")


def load_3S1_3D1_potential(name: str) -> CoupledPotential:
    """Load the coupled ³S₁-³D₁ nucleon-nucleon potential.

    The ³S₁-³D₁ channel is the spin-triplet coupled S-D wave channel
    (S=1, L=0,2, J=1, T=0). This coupling arises from the tensor force.

    Parameters
    ----------
    name : str
        Name of the potential (e.g., 'EM500', 'AV18').

    Returns
    -------
    CoupledPotential
        The loaded coupled ³S₁-³D₁ potential with blocks [V_SS, V_SD, V_DS, V_DD].

    Examples
    --------
    >>> pot = load_3S1_3D1_potential('EM500')
    >>> v_matrix = pot.without_weights()
    >>> print(v_matrix.shape)
    (200, 200)  # For 100 points per channel

    """
    pot_list = []

    for l1, l2 in [(0, 0), (0, 2), (2, 0), (2, 2)]:
        chan = Channel(
            spin=1, orb_ang_mom_1=l1, orb_ang_mom_2=l2, tot_ang_mom=1, isospin=0
        )
        chan_str = str(chan) + "_np"
        path = os.path.join(STANDARD_PATH, "NN", name, f"SLLJT_{chan_str}.dat")

        pot_list.append(load_from_file(path, name, chan, "np"))
    return CoupledPotential(pot_list)


def fast_and_lazy_plot(potential: Potential, v_scale: float = 1.0) -> None:
    """Plot potential with colorbar.

    Creates a quick visualization of the potential matrix.

    Parameters
    ----------
    potential : Potential
        Potential to be plotted.
    v_scale : float, optional
        Maximum magnitude to be reflected on the colorbar scale. Default is 1.0.

    """
    _, ax = plt.subplots()
    im = ax.matshow(
        potential.without_weights(),
        vmin=-1 * v_scale,
        vmax=v_scale,
        cmap=plt.cm.RdBu_r,
    )
    nodes = potential.nodes
    steps = 20
    plt.xticks([x for x in range(0, len(nodes), steps)])
    ax.set_xlabel(r"p (fm$^{-1}$)")
    ax.xaxis.set_label_position("top")
    ax.set_xticklabels(["{:.2f}".format(nodes[x]) for x in range(0, len(nodes), steps)])
    plt.ylabel(r"p' (fm$^{-1}$)")
    plt.yticks([x for x in range(0, len(nodes), steps)])
    ax.set_yticklabels(["{:.2f}".format(nodes[x]) for x in range(0, len(nodes), steps)])
    plt.colorbar(im)
    plt.show()
    plt.close()


# ------------------- Internal Methods ------------------------------------- #


def _add_w(matrix: np.ndarray, weights: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Add Gaussian quadrature weights to potential matrix.

    Transforms bare potential V(p,p') to weighted form for numerical integration.
    The factor (2/π) comes from momentum-space normalization.

    Parameters
    ----------
    matrix : ndarray
        Unweighted potential matrix.
    weights : ndarray
        Gaussian quadrature weights.
    nodes : ndarray
        Momentum grid points.

    Returns
    -------
    ndarray
        Weighted potential matrix.
    """
    factor_vector = [sqrt(w) * p for w, p in zip(weights, nodes)]
    weighted_matrix = np.dot(
        np.dot(np.diag(factor_vector), matrix), np.diag(factor_vector)
    )
    return 2 / pi * weighted_matrix


def _rem_w(matrix: np.ndarray, weights: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Remove Gaussian quadrature weights from potential matrix.

    Transforms weighted potential back to bare form V(p,p').
    The factor (π/2) inverts the normalization from _add_w.

    Parameters
    ----------
    matrix : ndarray
        Weighted potential matrix.
    weights : ndarray
        Gaussian quadrature weights.
    nodes : ndarray
        Momentum grid points.

    Returns
    -------
    ndarray
        Unweighted potential matrix.
    """
    factor_vector = [1 / (sqrt(w) * p) for w, p in zip(weights, nodes)]
    unweighted_matrix = np.dot(
        np.dot(np.diag(factor_vector), pi / 2 * matrix), np.diag(factor_vector)
    )
    return unweighted_matrix


def _submatrix(potential: np.ndarray, ranges: Tuple[int, int, int, int]) -> np.ndarray:
    """Extract submatrix from potential matrix.

    Parameters
    ----------
    potential : ndarray
        Full potential matrix.
    ranges : tuple of int
        Row and column ranges (row_start, row_end, col_start, col_end).

    Returns
    -------
    ndarray
        Extracted submatrix.
    """
    return potential[
        np.ix_(list(range(ranges[0], ranges[1])), list(range(ranges[2], ranges[3])))
    ]
