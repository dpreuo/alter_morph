import numpy as np
from koala.lattice import Lattice
import tinyarray as ta
import kwant
from .hamiltonians import _hopping_matrix
from .lattice_utilities import add_contacts


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
    U: float,
    m_values: np.ndarray,
    n_values: np.ndarray,
    theta_offset=0.0,
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

        neighbour_densities = n_values[vertex_neighbours]
        total_density = np.sum(neighbour_densities)

        syst[k_lattice(n)] = (
            J * total_magnetisation * np.diag(np.array([-1, 1, 1, -1]))
            + U * total_density * np.eye(n_orbs)
        )

    for n_edge in range(lattice.n_edges):
        vector = lattice.edges.vectors[n_edge]
        e0, e1 = lattice.edges.indices[n_edge]
        theta = np.arctan2(vector[0], vector[1])
        h0_term = np.kron(np.eye(2), _hopping_matrix(t1, t2, theta, theta_offset))

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


def crack_hamiltonian_for_contacts_kwant(
    lattice: Lattice, hamiltonian: np.ndarray, cross_edges=False
):
    """Given a lattice with periodic x and open y, we crack the lattice upen and add
    contacts, doubling every site on an edge crossing bond and placing the copy on the
    other side as a contact. Takes the Hamiltonian and copies the relevant edges so
    that the bonds connecting you to the contact have the right couplings.

    Args:
        lattice (Lattice): The lattice
        hamiltonian (np.ndarray): The hamiltonian
        cross_edges (bool, optional): Whether to add edges that cross the contacts. Defaults to False.

    Returns:
        Lattice: The new lattice with contacts
        Amorphous: The kwant lattice describing the same lattice with contacts
        kwant.Builder: The kwant system with the new Hamiltonian
        tuple: The new vertices that have been added corresponding to the left and right contacts

    """

    n_orbitals = int(hamiltonian.shape[0] / lattice.n_vertices)

    # check the lattice has the right boundaries
    total_boundaries = np.sum(np.abs(lattice.edges.crossing), axis=0)
    assert total_boundaries[0] != 0, "lattice has must be periodic in x"
    assert total_boundaries[1] == 0, "lattice must have open boundaries in y"

    # find the version of the lattice with contact zones added
    contact_lattice, new_vertices, original_vertices = add_contacts(
        lattice,
        return_added_indices=True,
        cross_edges=cross_edges,
        make_uniform=True,
    )

    # figure out all the onsite terms in the new lattice
    original_onsite_matrices = np.array(
        [
            hamiltonian[
                i * n_orbitals : (i + 1) * n_orbitals,
                i * n_orbitals : (i + 1) * n_orbitals,
            ]
            for i in range(lattice.n_vertices)
        ]
    )
    new_onsite_matrices = np.concatenate(
        [
            original_onsite_matrices,  # sites in the original lattice,
            original_onsite_matrices[
                original_vertices[0][0]
            ],  # sites on the left contact
            original_onsite_matrices[
                original_vertices[0][1]
            ],  # sites on the right contact
        ],
        axis=0,
    )

    # now we have to deal with which new vertex corresponds to which old vertex
    new = np.concatenate(new_vertices[0])
    old = np.concatenate(original_vertices[0])
    new_vertex_dict = {new[i]: old[i] for i in range(len(old))}
    old_vertex_indices = np.arange(lattice.n_vertices)
    old_vertex_dict = {
        old_vertex_indices[i]: old_vertex_indices[i] for i in range(lattice.n_vertices)
    }
    vertex_remapping_dict = old_vertex_dict | new_vertex_dict
    assert len(vertex_remapping_dict) == len(old_vertex_dict) + len(
        new_vertex_dict
    ), "something has gone wrong when checking all the new vertices"

    new_coupling_terms = []
    for edge in contact_lattice.edges.indices:

        # if it's an original edge then we can just use the coupling term
        if np.sort(edge).tolist() in np.sort(lattice.edges.indices, axis=1).tolist():
            i1 = edge[0]
            i2 = edge[1]
            coupling = hamiltonian[
                i1 * n_orbitals : (i1 + 1) * n_orbitals,
                i2 * n_orbitals : (i2 + 1) * n_orbitals,
            ]
            new_coupling_terms.append(coupling)

        # otherwise you have to find the original edge that
        # this corresponds to, and use the original coupling
        else:
            og_edge = np.array(
                [vertex_remapping_dict[edge[0]], vertex_remapping_dict[edge[1]]]
            )
            i1 = og_edge[0]
            i2 = og_edge[1]
            coupling = hamiltonian[
                i1 * n_orbitals : (i1 + 1) * n_orbitals,
                i2 * n_orbitals : (i2 + 1) * n_orbitals,
            ]

            new_coupling_terms.append(coupling)

    # check you got a coupling for every edge and every onsite:
    assert len(new_onsite_matrices) == contact_lattice.n_vertices
    assert len(new_coupling_terms) == len(contact_lattice.edges.indices)

    # now we can build the new hamiltonian
    kwant_system = kwant.Builder()
    kwant_lattice = Amorphous(n_orbitals, contact_lattice)

    # add the onsite terms
    for i, onsite in enumerate(new_onsite_matrices):
        kwant_system[kwant_lattice(i)] = onsite

    # add the coupling terms
    for edge, coupling in zip(contact_lattice.edges.indices, new_coupling_terms):
        i1, i2 = edge
        kwant_system[kwant_lattice(i1), kwant_lattice(i2)] = coupling

    return contact_lattice, kwant_lattice, kwant_system, new_vertices[0]


def attach_leads_to_cracked(
    contact_lattice: Lattice,
    kwant_lattice: Amorphous,
    kwant_system: kwant.builder.Builder,
    contact_vertices: np.ndarray,
    lead_onsite: np.ndarray,
    lead_coupling: np.ndarray,
    lead_sym=None,
):
    """Given a kwant system with contacts, attach leads to the contacts. The leads
    are assumed to be square lattices with the same spacing as the contacts. The
    contacts are assumed to be on the left and right of the system. The leads are
    created with specified onsite and coupling terms.

    Args:
        contact_lattice (Lattice): The koala lattice with contacts
        kwant_lattice (Amorphous): The same kwant lattice with contacts
        kwant_system (kwant.builder.Builder): The kwant system, created with the crack_hamiltonian_for_contacts_kwant function
        contact_vertices (np.ndarray): The indices of the vertices that in the left and right contacts
        lead_onsite (np.ndarray): The onsite terms for the leads
        lead_coupling (np.ndarray): The coupling terms for the leads
        lead_sym (np.ndarray): Specify lead symmetries. A matrix which commutes with the lead Hamiltonian (default value is none)

    Returns:
        kwant.builder.Builder: The kwant system with leads attached
        kwant.builder.Builder: The left lead
        kwant.builder.Builder: The right lead
    """

    n_orbitals = kwant_lattice.norbs
    assert np.all(
        lead_onsite.shape == (n_orbitals, n_orbitals)
    ), "lead_onsite has wrong shape"
    assert np.all(
        lead_coupling.shape == (n_orbitals, n_orbitals)
    ), "lead_coupling has wrong shape"

    # find sites for the leads, and positions of the contacts
    left_vertices = contact_vertices[0]
    left_positions = contact_lattice.vertices.positions[left_vertices]
    left_y_range = (np.min(left_positions[:, 1]), np.max(left_positions[:, 1]))
    left_a = (left_y_range[1] - left_y_range[0]) / (len(left_vertices) - 1)
    # left_shift = left_y_range[0] / left_a

    right_vertices = contact_vertices[1]
    right_positions = contact_lattice.vertices.positions[right_vertices]
    right_y_range = (np.min(right_positions[:, 1]), np.max(right_positions[:, 1]))
    right_a = (right_y_range[1] - right_y_range[0]) / (len(right_vertices) - 1)
    # right_shift = right_y_range[0] / right_a

    # create left lead
    left_lead_lattice = kwant.lattice.Monatomic(
        ((left_a, 0), (0, left_a)),
        norbs=n_orbitals,
        name="left_lead",
        offset=(0, left_y_range[0]),
    )
    left_lead_symmetry = kwant.TranslationalSymmetry(left_lead_lattice.vec((-1, 0)))

    left_lead = kwant.Builder(left_lead_symmetry, conservation_law=lead_sym)
    for i in range(len(left_vertices)):
        left_lead[left_lead_lattice(0, i)] = lead_onsite
    left_lead[left_lead_lattice.neighbors()] = lead_coupling
    for i in range(len(left_vertices)):
        kwant_system[(left_lead_lattice(0, i))] = lead_onsite
        kwant_system[left_lead_lattice(0, i), kwant_lattice(left_vertices[i])] = (
            lead_coupling
        )
    kwant_system[left_lead_lattice.neighbors()] = lead_coupling
    kwant_system.attach_lead(left_lead)

    # # create right lead
    right_lead_lattice = kwant.lattice.Monatomic(
        ((right_a, 0), (0, right_a)),
        norbs=n_orbitals,
        name="right_lead",
        offset=(1, right_y_range[0]),
    )
    right_lead_symmetry = kwant.TranslationalSymmetry(right_lead_lattice.vec((1, 0)))
    right_lead = kwant.Builder(right_lead_symmetry, conservation_law=lead_sym)
    for i in range(len(right_vertices)):
        right_lead[right_lead_lattice(0, i)] = lead_onsite
    right_lead[right_lead_lattice.neighbors()] = lead_coupling
    for i in range(len(right_vertices)):
        kwant_system[right_lead_lattice(0, i)] = lead_onsite
        kwant_system[right_lead_lattice(0, i), kwant_lattice(right_vertices[i])] = (
            lead_coupling
        )
    kwant_system[right_lead_lattice.neighbors()] = lead_coupling
    kwant_system.attach_lead(right_lead)

    # kwant_system.eradicate_dangling()

    return kwant_system, left_lead, right_lead
