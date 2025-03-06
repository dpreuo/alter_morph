from cluster_tum.sim import find_phase, find_all_Jscan
import pickle
import numpy as np


def test_find_phase():
    lattice = pickle.load(open('cluster_tum/voronoi_20.pickle','rb'))
    find_phase(t1=1,t2=0,J=0,filling=0,theta_offset=0,lattice=lattice,iteration_steps=1,learning_rate=0,tol_mdiff=1e-6,name='MF_test')


def test_find_all_Jscan():
    lattice = pickle.load(open('cluster_tum/voronoi_20.pickle','rb'))
    find_all_Jscan(t1=1,t2=0,Js=np.linspace(0,1,11),filling=0,theta_offset=0,lattice=lattice,iteration_steps=1,learning_rate=0,tol_mdiff=1e-6,name='MF_test')