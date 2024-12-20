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

    contact1, added_indices1, og_indices1 = add_contacts(
        vor_lat, x_y_contacts=[True, False], return_added_indices=True
    )

    contact2, added_indices2, og_indices2 = add_contacts(
        vor_lat, x_y_contacts=[False, True], return_added_indices=True
    )
    contact3, added_indices3, og_indices3 = add_contacts(
        vor_lat, x_y_contacts=[True, True], return_added_indices=True
    )

    contacts = [contact1, contact2, contact3]
    added_indices = [added_indices1, added_indices2, added_indices3]
    og_indices = [og_indices1, og_indices2, og_indices3]

    for j in range(3):
        ind = added_indices[j]
        og_ind = og_indices[j]
        l = contacts[j]
        for set_n in range(len(ind)):
            for left_right in range(2):
                assert len(ind[set_n][left_right]) == len(og_ind[set_n][left_right])
    