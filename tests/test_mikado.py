from alter_morph.mikado_lattice import koala_mikado, relaxation, random_line_graph
import numpy as np


def test_koala_mikado():
    lat1 = koala_mikado(100,0)
    lat2 = koala_mikado(100,10)
    lat3 = koala_mikado(100,100)

def test_random_line_graph():
    coords, bonds = random_line_graph(1,50,relax_steps=10)
    coords, bonds = random_line_graph(1,50,relax_steps=50)

def test_relaxation():
    coords, bonds = random_line_graph(1,50,relax_steps=10)
    new_coords, new_bonds = relaxation(coords, bonds,0.7, 30)
    if not np.all(new_coords.shape == coords.shape):
        raise ValueError("relaxation failed")
