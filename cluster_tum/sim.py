"""Example simulation file, containing a function that (usually) runs for a very long time... and outputs a pickle file

To run your own simulations, adjust the functions by your liking,
but keep (or copy & paste) the last 11 lines of this file."""

#needed to import altermorph correctly
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__))+'/src')

import pickle
import numpy as np
from alter_morph.mean_field import hartree_fock


def find_phase(**param):
    """
    finds the converged mvalues for a single point in phase space. Stores results in pickle file
    """
    lattice = param['lattice']
    
    initial_parameters = {
        "t1": param['t1'],
        "t2": param['t2'],
        "J": param['J'],
        "filling": param['filling'],
        "initial_m": np.full(lattice.n_vertices, 1),
        "theta_offset": param['theta_offset'],
    }

    m_values = hartree_fock(
        param['lattice'],
        initial_parameters,
        param['iteration_steps'],
        mixing_proportion=param['learning_rate'],
        verbose=False,
    )

    initial_parameters.pop('initial_m')

    saved_results = dict(
        hparam = dict(**initial_parameters,m=m_values[-1]),
        numerical_param = dict(m_values=m_values,iteration_steps=param['iteration_steps'], learning_rate=param['learning_rate'],tol_mdiff=param['tol_mdiff']), 
    )
    
    pickle.dump(saved_results, open(param['name'] + '.pickle', 'wb'))
    


def generateFilename(name='_',**kwargs):
    """generates an unique filname"""
    s = name
    for key in kwargs.keys():
        s += '_{0}={1}'.format(str(key), str(kwargs[key]))
    return s


if __name__ == "__main__":
	
    import cluster_jobs
    cluster_jobs.run_simulation_commandline(globals())