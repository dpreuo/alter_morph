"""Example simulation file, containing a function that (usually) runs for a very long time... and outputs a pickle file

To run your own simulations, adjust the functions by your liking,
but keep (or copy & paste) the last 11 lines of this file."""


import pickle




def optimizeMF_multiAnsatz(**param):
    """try the different MF ansaetze and compare them in energy, only safe the one lowest in energy"""
    
    #write a function which finds the phase diagram here
    #
    #
    #
    #
    #
    #
    #
    

    pickle.dump(res_list[idx_min], open(name + '.pickle', 'wb'))
    print(param)
    return mfs[idx_min], es[idx_min], res_list[idx_min]




def generateFilename(name='_',**kwargs):
    """generates an unique filname"""
    s = name
    for key in kwargs.keys():
        s += '_{0}={1}'.format(str(key), str(kwargs[key]))
    return s


if __name__ == "__main__":
	
    import cluster_jobs
    cluster_jobs.run_simulation_commandline(globals())