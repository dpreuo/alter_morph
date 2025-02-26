import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from koala.voronization import generate_lattice
from koala.pointsets import uniform, bluenoise
from koala import graph_utils as gu
from koala import example_graphs as eg
from koala.lattice import Lattice
from koala.graph_color import color_lattice
from koala import plotting as pl
from tqdm import tqdm
from copy import copy

from alter_morph.hamiltonians import (
    alt_hamiltonian,
    find_m_per_state,
    find_m_and_n_values,
    find_spin_per_state,
    spectral_function,
)

# generate the lattice for all the tests
n_vertices = 6**2
points = uniform(n_vertices)
vor_lat = generate_lattice(uniform(n_vertices // 2))

def test_hamiltonian():

    t1 = 1
    t2 = 0.3
    J = 0.4
    U = 0.4
    filling = 0.5
    initial_m = np.zeros(vor_lat.n_vertices) + 1.2
    initial_n = np.zeros(vor_lat.n_vertices) + filling*4
    
    ham = alt_hamiltonian(vor_lat, t1, t2, J, U, initial_m, initial_n)
    assert( ham.conj().T == ham).all() # check if the hamiltonian is hermitian

    e, v = np.linalg.eigh(ham)
    m_values, n_values = find_m_and_n_values(v, filling)
    average_m_per_state = find_m_per_state(v)
    average_spin_per_state = find_spin_per_state(v)

    fermi_occupation = np.linspace(0, 1, len(e)) <= filling

    # check if the average m per state is the same as the m values
    assert np.isclose((average_m_per_state*fermi_occupation).sum(), m_values.sum(), atol=1e-10)

    omega = -3
    specs = [
        spectral_function(vor_lat, e, v, omega, local_operator=np.array([1, 0, 0, 0])),
        spectral_function(vor_lat, e, v, omega, local_operator=np.array([0, 1, 0, 0])),
        spectral_function(vor_lat, e, v, omega, local_operator=np.array([0, 0, 1, 0])),
        spectral_function(vor_lat, e, v, omega, local_operator=np.array([0, 0, 0, 1])),
    ]
