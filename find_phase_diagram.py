import numpy as np
from scipy import linalg as la
from alter_morph.hamiltonians import (
    alt_hamiltonian,
)
from multiprocessing import Pool as WorkerPool

from alter_morph.lattice_utilities import alter_lattice_maker
from alter_morph.mean_field import hartree_fock


def main():

    system_length = 20
    lattice_type = "voronoi"
    lattice = alter_lattice_maker(system_length, lattice_type)

    J = 0.5
    filling = 0.5
    run_length = 80

    initial_parameters = {
        "t1": 1,
        "t2": 0.5,
        "J": J,
        "filling": filling,
        "initial_m": np.full(lattice.n_vertices, 1),
        "theta_offset": 0,
    }

    m_values = hartree_fock(
        lattice,
        initial_parameters,
        run_length,
        mixing_proportion=0.5,
        verbose=False,
    )

    hamiltonian = alt_hamiltonian(
        lattice,
        initial_parameters["t1"],
        initial_parameters["t2"],
        initial_parameters["J"],
        m_values[-1],
        theta_offset=initial_parameters["theta_offset"],
    )
    energies, states = la.eigh(hamiltonian)

    stuff_to_save = {
        

if __name__ == "__main__":
    main()