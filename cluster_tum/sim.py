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
from alter_morph.hamiltonians import alt_hamiltonian
from scipy import linalg as la
from alter_morph.hamiltonians import alt_hamiltonian
from scipy import linalg as la


def find_phase(**param):
    """
    finds the converged mvalues for a single point in phase space. Stores results in pickle file
    """
    lattice = param['lattice']
    
    initial_parameters = {
        "t1": param['t1'],
        "t2": param['t2'],
        "J": param['J'],
        "U": param['J'],
        "filling": param['filling'],
        "initial_m": np.full(lattice.n_vertices, 1),
        "initial_n": np.full(lattice.n_vertices, param['filling'] * 4),
        "theta_offset": param['theta_offset'],
    }

    m_values, n_values = hartree_fock(
        param['lattice'],
        initial_parameters,
        param['iteration_steps'],
        mixing_proportion=param['learning_rate'],
        verbose=False,
        tol_mdiff=param['tol_mdiff'],
        adjust_learning_rate=True
    )

    hamiltonian = alt_hamiltonian(
        lattice,
        initial_parameters['t1'],
        initial_parameters['t2'],
        initial_parameters['J'],
        initial_parameters['U'],
        m_values[-1],
        n_values[-1],
        initial_parameters['theta_offset']
    )

    energies = la.eigvalsh(hamiltonian)

    initial_parameters.pop('initial_m')
    initial_parameters.pop('initial_n')

    saved_results = dict(
        hparam = dict(**initial_parameters,m=m_values[-1],n=n_values[-1],energies=energies),
        numerical_param = dict(m_values=m_values,n_values=n_values,iteration_steps=param['iteration_steps'], learning_rate=param['learning_rate'],tol_mdiff=param['tol_mdiff']), 
    )
    
    pickle.dump(saved_results, open(param['name'] + '.pickle', 'wb'))
    

def find_all_Jscan(**param):
    """
    finds the converged mvalues for a all points specfied by Js in phase space. Stores results in pickle file
    """
    lattice = param['lattice']
    initial_parameters = {
        "t1": param['t1'],
        "t2": param['t2'],
        "filling": param['filling'],
        "initial_m": np.full(lattice.n_vertices, 1),
        "initial_n": np.full(lattice.n_vertices, param['filling'] * 4),
        "theta_offset": param['theta_offset'],
    }

    saved_results=[]
    #run stuff starting with largest J to speed up convergence
    for J in param['Js'][::-1]:
        initial_parameters['J'] = J
        initial_parameters['U'] = J

        m_values, n_values = hartree_fock(
            param['lattice'],
            initial_parameters,
            param['iteration_steps'],
            mixing_proportion=param['learning_rate'],
            verbose=False,
            tol_mdiff=param['tol_mdiff'],
            adjust_learning_rate=True
        )

        #get energies
        hamiltonian = alt_hamiltonian(
            lattice,
            param["t1"],
            param["t2"],
            J,
            J,
            m_values[-1],
            n_values[-1],
            theta_offset=param["theta_offset"],
        )
        energies = la.eigvalsh(hamiltonian)

        #save results
        initial_parameters.pop('initial_m')
        initial_parameters.pop('initial_n')
        saved_results.append(dict(
            hparam = dict(**initial_parameters,m=m_values[-1],n=n_values[-1]),
            energies = energies,
            numerical_param = dict(m_values=m_values,n_values=n_values,iteration_steps=param['iteration_steps'], learning_rate=param['learning_rate'],tol_mdiff=param['tol_mdiff']), 
            ))

        #take values from last run as initial values
        initial_parameters['initial_m'] = m_values[-1]
        initial_parameters['initial_n'] = n_values[-1]

    saved_results = saved_results[::-1]
    
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