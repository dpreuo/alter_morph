import numpy as np
from koala.lattice import Lattice
import tinyarray as ta
import kwant
from .hamiltonians import _hopping_matrix


class Amorphous(kwant.builder.SiteFamily):
    def __init__(self, norbs: int, lattice: Lattice):
        """Create a site family object for a kwant lattice.

        Args:
            norbs (int): The number of orbitals per site
            lattice (Lattice): A koala lattice object

        Raises:
            ValueError: If norbs is not an integer > 0
        """

        positions = lattice.vertices.positions
        if norbs is not None:
            if int(norbs) != norbs or norbs <= 0:
                raise ValueError("The norbs parameter must be an integer > 0.")
            norbs = int(norbs)
        self.norbs = norbs
        self.positions = positions
        self.canonical_repr = (
            f"lattice {int(self.__hash__()%1e4)}, norbs {norbs}, {len(positions)} sites"
        )
        # self.canonical_repr = '1'

    def pos(self, tag):
        """Return the real-space position of the site with a given tag."""
        return self.positions[tag[0], :]

    def normalize_tag(self, tag):
        """Return a normalized version of the tag.

        Raises TypeError or ValueError if the tag is not acceptable.
        """
        if tag[0] >= len(self.positions):
            raise ValueError
        return ta.array(tag)

    def __hash__(self):
        return hash(ta.array(self.positions))
        # return 1


def kwant_altermagnetic_hamiltonian(
    lattice: Lattice,
    t1: float,
    t2: float,
    J: float,
    m_values: np.ndarray,
    return_lattice=False,
):
    """This is the kwant implementation of the function alt_hamiltonian in the
    hamiltonians module. Works the same but creates a kwant.Builder object.


    Args:
        lattice (Lattice): The lattice to generate the Hamiltonian for
        t1 (float): Strong hopping parameter where direction matches orbital
        t2 (float): Weak hopping parameter where direction opposes orbital
        J (float): Interacting coupling parameter
        m_values (np.ndarray): Mean field values for the magnetic moments per site
        return_lattice (bool, optional): Whether to return the kwant lattice
            object for the lattice. Defaults to False.

    Returns:
        kwant.Builder: The kwant.Builder object with the Hamiltonian
        Amorphous: The site family object for the lattice (if return_lattice is True)
    """

    n_orbs = 4
    syst = kwant.Builder()
    k_lattice = Amorphous(n_orbs, lattice)

    for n in range(lattice.n_vertices):
        vertex_neighbours = lattice.vertices.adjacent_vertices[n]
        neighbour_magnetisations = m_values[vertex_neighbours]
        total_magnetisation = np.sum(neighbour_magnetisations)
        syst[k_lattice(n)] = J * total_magnetisation * np.diag(np.array([-1, 1, 1, -1]))

    for n_edge in range(lattice.n_edges):
        vector = lattice.edges.vectors[n_edge]
        e0, e1 = lattice.edges.indices[n_edge]
        theta = np.arctan2(vector[0], vector[1])
        h0_term = np.kron(np.eye(2), _hopping_matrix(t1, t2, theta))

        syst[k_lattice(e0), k_lattice(e1)] = h0_term
        # syst[k_lattice(e1), k_lattice(e0)] = h0_term.T.conj()

    if return_lattice:
        return syst, k_lattice
    return syst


def cumulative_density(dos: kwant.kpm.SpectralDensity, density=True):
    """Given a density of states, kwant.kpm.SpectralDensity, return the cumulative
    density of states. This is useful for finding the the fermi level for a given
    filling fraction.

    Args:
        dos (kwant.kpm.SpectralDensity): The density of states object
        density (bool, optional): Whether to normalise the cumulative density such that
            the maximum value is 1. Defaults to True.

    Returns:
        energies (np.ndarray): The energies at which the density is calculated
        cumulative_density (np.ndarray): The cumulative density of states
    """

    factor = dos._a / (2 * dos.num_moments)
    rho = np.transpose(dos._gammas.transpose()).real
    if density:
        total = factor * np.sum(rho, axis=0)
    else:
        total = 1

    return dos.energies, factor * np.cumsum(rho, axis=0) / total


def lattice_ham_to_kwant(lattice: Lattice, hamiltonian: np.ndarray):
    """Converts a lattice and a hamiltonian to a kwant lattice and system.

    Args:
        lattice (Lattice): A koala lattice.
        hamiltonian (np.ndarray): A hamiltonian, must be stored as a numpy array with the
        orbital indices orbital degrees of freedom ordered before the spatial degrees of freedom.

    Returns:
        kwant_lattice (Amorphous): A kwant lattice.
        kwant_syst (kwant.Builder): A kwant system.
    """

    n_orbs = hamiltonian.shape[0] // lattice.n_vertices

    kwant_syst = kwant.Builder()
    kwant_lattice = Amorphous(n_orbs, lattice)

    on_site_terms = [
        hamiltonian[i * n_orbs : (i + 1) * n_orbs, i * n_orbs : (i + 1) * n_orbs]
        for i in range(lattice.n_vertices)
    ]

    for i in range(lattice.n_vertices):
        if np.any(on_site_terms[i]):
            kwant_syst[kwant_lattice(i)] = on_site_terms[i]

    for i in range(lattice.n_vertices):
        for j in range(i):
            element = hamiltonian[
                i * n_orbs : (i + 1) * n_orbs, j * n_orbs : (j + 1) * n_orbs
            ]
            if np.any(element):
                kwant_syst[kwant_lattice(i), kwant_lattice(j)] = element

    return kwant_lattice, kwant_syst
