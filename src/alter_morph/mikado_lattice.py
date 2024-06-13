
import numpy as np


def get_drs(coords, coords2=None, period=None):
    if coords2 is None:
        coords2 = coords
    dim = coords.shape[-1]
    drs = []
    for i in range(dim):
        drs.append(np.add.outer(coords[:, i], -coords2[:, i]).flatten())
    drs = np.array(drs)
    if period is not None:
        drs = drs - period[:, None] * np.sign(drs) * (np.abs(drs) > period[:, None]/2)
    return drs

def relaxation_2(coords, bonds, force = None, force_params = None, z_min = 6):
    dim = np.size(coords, 1)
    if force is None:
        def force(dist):
            return np.zeros(np.shape(dist))
    displacement = np.zeros(np.shape(coords))
    for n in np.arange(np.size(coords,0)) :
        neighbors = np.vstack((np.isin(bonds[:,1],np.array([n])),
                            np.isin(bonds[:,0],np.array([n])))).T
        if np.sum(neighbors)>=z_min:
            neighbors = bonds[neighbors]
            neighbors = coords[neighbors]
            dist = get_drs(neighbors, coords[n,:].reshape(1,dim))
            resultante = (np.sum(force(dist, **force_params), 1))
#                           +np.sum(bond_force, 0))
#             print(resultante)
            displacement[n,:] += resultante
    return coords+displacement

def spring_2(dist, l0, k):
    norm = np.sum(dist**2,0)
    norm = np.sqrt(norm)
#     print(dist)
    intensity = k*(norm-l0)/norm
#     print(np.shape(intensity), np.shape(dist))
    return intensity*dist


def intersection(p1, t1, p2, t2):
    theta = t2 - t1
    phi = np.arctan2(p2/p1 - np.cos(theta), np.sin(theta))
    r = p1 / np.cos(phi)
    return r, phi + t1

def cut_graph(coords, bonds, shape):
    # Return the subgraph within shape
    #mask = shape(coords)
    mask = np.array([shape(coords[k]) for k in range(len(coords))])
    new_coords = coords[mask]
    index_map = np.cumsum(mask) - 1
    new_bonds = bonds[np.logical_and(*mask[bonds].T)]
    new_bonds = index_map[new_bonds]
    return new_coords, new_bonds

def random_line_graph(R, density, shape=None, relax_steps=0):
    """Generate fourfold coordinated graph using a random set of straight lines.
    Computation time is linear in the number of nodes. See also:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC300370/
    Parameters
    ----------
    R : float
        Radius of the circular area of the graph
    density : float
        Expected density of the nodes. The actual number of nodes is random with
        Poisson distribution.
    shape : callable (optional)
        Boolean function of 2D coordinates that specifies the shape of the graph
        area. Should be inside the radius `R` circle. By default the radius `R`
        circle is returned. Should take array inputs.
    relax_steps : integer, default 0
        Relaxation steps to perform on the graph. Nodes on the edges (with less
        than 4 neighbors) are left fixed, points inside are moved to the barycenter
        of their neighbors in each step.
    Returns
    -------
    coords : ndarray
        Array of the 2D coordinates of the nodes.
    bonds : ndarray
        Array of pairs of integers corresponding to the edges of the graph.
    """
    # Offsets of lines from origin, poisson distribution on positive line segment
    # (factor of 2 compared to MILES comes from only using positive values)
    N = np.random.poisson(2 * np.sqrt(np.pi * density) * R)
    ps = np.random.uniform(0, R, N)
    # Angles of the lines
    thetas = np.random.uniform(0, 2 * np.pi, N)
    # Array of intersection points of pairs of lines in polar coordinates
    rs, phis = intersection(ps, thetas, ps[:, np.newaxis], thetas[:, np.newaxis])
    # get rid of diagonal
    rs[np.diag_indices(N)] = np.inf
    # connect neighbors
    # Indexing of intersections
    indx = np.zeros((N, N), dtype=int)
    indx[np.triu_indices(N, k=1)] = range(N * (N - 1) // 2)
    indx[np.triu_indices(N, k=1)[::-1]] = range(N * (N - 1) // 2)
    indx[np.diag_indices(N)] = -99999
    # offsets of intersection points
    ds = np.sin(phis - thetas[:, np.newaxis]) * rs
    # sort points by offset
    order = np.argsort(ds, axis=1)
    bonds = np.empty((0, 2), dtype=int)
    for i, line in enumerate(order):
        new_bonds = np.array([indx[i, line[:-2]], indx[i, line[1:-1]]]).T
        bonds = np.vstack([bonds, new_bonds])
    # Pick out the upper triangle (without diagonal)
    ind = np.triu_indices(N, k=1)
    coords = rs[ind] * np.array([np.cos(phis[ind]), np.sin(phis[ind])])
    coords = coords.T

    # remove points outside radius R
    if shape is None:
        def shape(x): np.abs(x[0]) < R and np.abs(x[1])<R
        """
        def shape(x):
            if (np.abs(x[0])<R and np.abs(x[1])<R):
                return 1
            else:
                return 0
        """
    coords, bonds = cut_graph(coords, bonds, shape)

    # relax the graph
    
    for _ in range(relax_steps):
        coords = relaxation_2(coords, bonds, spring_2, {'l0':1/density, 'k':.3}, z_min=1)

    return coords, bonds