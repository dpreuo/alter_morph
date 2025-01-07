import numpy as np
from scipy import linalg as la
from koala import plotting as pl
from alter_morph.hamiltonians import (
    alt_hamiltonian,
    find_m_per_state,
    find_spin_per_state,
    spectral_function,
)
from alter_morph.lattice_utilities import alter_lattice_maker
from alter_morph.mean_field import hartree_fock, hartree_fock_with_boundary_twisting


def test_mean_field():
    system_length = 5
    lattice_type = "voronoi"
    lattice = alter_lattice_maker(system_length, lattice_type)
    # lattice = koala.graph_utils.cut_boundaries(lattice)

    t1 = 1  # orbital parallel neighbor hopping
    t2 = 0.5  # orbital perpendicular neighbor hopping
    J = 0.5  # Interaction strength
    filling = 0.2  # filling of the lattice
    initial_m = np.full(lattice.n_vertices, 1)  # initial magnetization
    theta_offset = 0.5  # offset of the staggered magnetization

    initial_parameters = {
        "t1": t1,
        "t2": t2,
        "J": J,
        "filling": filling,
        "initial_m": initial_m,
        "theta_offset": theta_offset,
    }
    m_values = hartree_fock(lattice, initial_parameters, 20)
    m_values = hartree_fock_with_boundary_twisting(lattice, initial_parameters, 5, 4)
