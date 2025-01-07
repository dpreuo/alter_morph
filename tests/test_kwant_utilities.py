from alter_morph.lattice_utilities import alter_lattice_maker, add_contacts
import koala
import numpy as np
from alter_morph.kwant_utilities import (
    crack_hamiltonian_for_contacts_kwant,
    attach_leads_to_cracked,
    kwant_altermagnetic_hamiltonian,
    lattice_ham_to_kwant,
)
from alter_morph.hamiltonians import alt_hamiltonian


def test_crack_hamiltonian_leads():

    lattice = alter_lattice_maker(5, "voronoi")
    lattice = koala.graph_utils.cut_boundaries(lattice, [False, True])

    t1 = 1  # orbital parallel neighbor hopping
    t2 = 0.5  # orbital perpendicular neighbor hopping
    J = 0.4  # Interaction strength
    initial_m = np.full(lattice.n_vertices, 1)  # initial magnetization

    lead_onsite = np.diag([1, 1, 1, 1])  # lead onsite
    lead_coupling= np.diag([1, 1, 1, 1])  # lead coupling
    lead_symmetry = np.diag([-1, -1, 1, 1])  # lead symmetry

    hamiltonian = alt_hamiltonian(lattice, t1, t2, J, initial_m)

    contact_lattice, k_lattice, k_system, contact_vertices = (
        crack_hamiltonian_for_contacts_kwant(lattice, hamiltonian)
    )
    kwant_system_with_leads, left_lead, right_lead = attach_leads_to_cracked(
        contact_lattice, k_lattice, k_system, contact_vertices, lead_onsite, lead_coupling, lead_symmetry
    )

def test_kwant_altermagnetic_hamiltonian():

    lattice = alter_lattice_maker(5, "voronoi")
    lattice = koala.graph_utils.cut_boundaries(lattice, [False, True])

    t1 = 1  # orbital parallel neighbor hopping
    t2 = 0.5  # orbital perpendicular neighbor hopping
    J = 0.4  # Interaction strength
    initial_m = np.full(lattice.n_vertices, 1)  # initial magnetization
    theta = 0.572

    hamiltonian = alt_hamiltonian(lattice, t1, t2, J, initial_m, theta)
    k_hamiltonian = kwant_altermagnetic_hamiltonian(lattice, t1, t2, J, initial_m, theta).finalized()

    assert np.allclose(hamiltonian, k_hamiltonian.hamiltonian_submatrix())

def test_lattice_ham_to_kwant():
    lattice = alter_lattice_maker(5, "voronoi")
    lattice = koala.graph_utils.cut_boundaries(lattice, [False, True])

    t1 = 1  # orbital parallel neighbor hopping
    t2 = 0.5  # orbital perpendicular neighbor hopping
    J = 0.4  # Interaction strength
    initial_m = np.full(lattice.n_vertices, 1)  # initial magnetization
    theta = 0.572

    hamiltonian = alt_hamiltonian(lattice, t1, t2, J, initial_m, theta)
    k_hamiltonian = lattice_ham_to_kwant(lattice, hamiltonian)[1].finalized()

    assert np.allclose(hamiltonian, k_hamiltonian.hamiltonian_submatrix())