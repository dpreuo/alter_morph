import numpy as np
from koala.lattice import Lattice
import tinyarray as ta
import kwant
from .hamiltonians import _hopping_matrix


class Amorphous(kwant.builder.SiteFamily):
    def __init__(self, norbs: int, lattice: Lattice):

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

    factor = dos._a / (2 * dos.num_moments)
    rho = np.transpose(dos._gammas.transpose()).real
    if density:
        total = factor * np.sum(rho, axis=0)
    else:
        total = 1

    return dos.energies, factor * np.cumsum(rho, axis=0) / total
