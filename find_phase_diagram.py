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
from alter_morph.mean_field import hartree_fock
# plotting
import matplotlib
import matplotlib.pyplot as plt


# generate the lattice and set the initial conditions

import koala.graph_utils


system_length = 20
lattice_type = "voronoi"
lattice = alter_lattice_maker(system_length, lattice_type)
# lattice = koala.graph_utils.cut_boundaries(lattice)


t1 = 1  # orbital parallel neighbor hopping
t2 = 0.5  # orbital perpendicular neighbor hopping
J_lims = [0, 1]  # Interaction strength
filling_lims = [0,1]  # filling of the lattice
initial_m = np.full(lattice.n_vertices, 1)  # initial magnetization
theta_offset = 0.  # offset of the staggered magnetization
n_runs = 4


for J in np.linspace(*J_lims, n_runs):
    for filling in np.linspace(*filling_lims, n_runs):
        

        initial_parameters = {
            "t1": t1,
            "t2": t2,
            "J": J,
            "filling": filling,
            "initial_m": initial_m,
            "theta_offset": theta_offset,
        }

        m_values = hartree_fock(lattice, initial_parameters, 100, mixing_proportion=0.5)

