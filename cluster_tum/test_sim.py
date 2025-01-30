from cluster_tum.sim import find_phase
import pickle


def test_find_phase():
    lattice = pickle.load(open('cluster_tum/voronoi_20.pickle','rb'))
    find_phase(t1=1,t2=0,J=0,n=0,theta_offset=0,lattice=lattice,iteration_steps=1,learning_rate=0,tol_mdiff=1e-6,name='MF_test')
