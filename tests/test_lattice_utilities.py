
from koala.voronization import generate_lattice
from koala.pointsets import uniform, bluenoise
from koala import graph_utils as gu
from koala import example_graphs as eg
from koala.lattice import Lattice
from koala.graph_color import color_lattice
from koala import plotting as pl
from tqdm import tqdm
from alter_morph.lattice_utilities import add_contacts
from scipy import linalg as la

from alter_morph.hamiltonians import (
    alt_hamiltonian,
    find_m_per_state,
    find_m_values,
    find_spin_per_state,
    fermi_probability,
    spectral_function,
)

from copy import copy
import numpy as np
import matplotlib.pyplot as plt


def test_add_contacts():
    n_vertices = 20
    points = uniform(n_vertices)
    vor_lat = generate_lattice(points)
    open_boundaries = gu.cut_boundaries(vor_lat)

    contact0 = add_contacts(vor_lat, x_y_contacts=[False, False])
    # assert contact0 == vor_lat
    assert contact0.n_vertices == open_boundaries.n_vertices
    assert contact0.n_edges == open_boundaries.n_edges
    

    contact1 = add_contacts(vor_lat, x_y_contacts=[True, False])
    contact2 = add_contacts(vor_lat, x_y_contacts=[False, True])
    contact3 = add_contacts(vor_lat, x_y_contacts=[True, True])
