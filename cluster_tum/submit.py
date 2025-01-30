"""Example how to create a `config` for a job array and submit it using cluster_jobs.py."""

import cluster_jobs
import copy
import itertools
import numpy as np  # only needed if you use np below
import pickle # only needed if you use np below

config = {
    'jobname': 'altermorph',
    'task': {
        'type': 'PythonFunctionCall',
        'module': 'sim', #specify file from which to import a costly function
        'function': 'find_phase' #name of the function which solves system for specific paramteres
    },
    'task_parameters': [],  # list of dict containing the **kwargs given to the `function`
    'requirements_slurm': {  # passed on to SLURM
        'mem': '1G',
        'time': '0:30:00',  # d-hh:mm:ss
        'nodes': 1,  # number of nodes
        'mail-user': "valentin.leeb@tum.de",
        'qos':'short',
    },
    'options': {  # further replacements for the job script; used to determine extra requirements
        # 'mail': 'no@example.com',
        'cores_per_task': 1,
    }
}

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
lattice = pickle.load(open('voronoi_20.pickle','rb')) #voronoi with system length 20

params = {	'name': ['MF'],
            'J':np.linspace(0,1,11),
            'n':np.linspace(0,1,11),
            't1':[1],
            't2':[0.5],
            'theta_offset':[0],
            'lattice':[lattice],#INCLUDE the voronoi lattice here as parameter
            #numerical parameters
            'learning_rate':[0.5],
            'iteration_steps': [200],
            'tol_mdiff':[1e-3],
            }
            
## for all possible list values
for j,entry in enumerate(itertools.product(*[params[i] for i in params])):
    kwargs = {param: value for param, value in zip(params, entry)}
    kwargs['name'] = kwargs['name'] + str(j)
    config['task_parameters'].append(copy.deepcopy(kwargs))


# cluster_jobs.TaskArray(**config).run_local(task_ids=[2, 3], parallel=2) # run selected tasks
# cluster_jobs.JobConfig(**config).submit()  # run all tasks locally by creating a bash job script
cluster_jobs.SlurmJob(**config).submit()  # submit to SLURM
# cluster_jobs.SGEJob(**config).submit()  # submit to SGE
