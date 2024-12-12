from koala.lattice import Lattice
import numpy as np


def add_contacts(
    lattice: Lattice,
    x_y_contacts=[True, False],
    cross_edges=False,
    make_uniform=False,
    return_added_indices=False,
):
    """Add contacts to the lattice in the x and y directions.

    Args:
        lattice (Lattice): Lattice to add contacts to. Must have periodic boundary conditions in the chosen directions.
        x_y_contacts (list, optional): List of booleans indicating whether to add contacts in the x and y directions. Defaults to [True, False].
        cross_edges (bool, optional): Whether to add perpendicular edges to the new vertices. Defaults to False.
        make_uniform (bool, optional): Whether to evenly space the new vertices. Defaults to False.
        return_added_indices (bool, optional): Whether to return the indices of the added vertices. Defaults to False.

    Returns:
        Lattice: Lattice with added contacts.
        tuple: Tuple of indices of the added vertices, if return_added_indices is True.
    """

    vertices = lattice.vertices.positions
    edges = lattice.edges.indices
    crossing = lattice.edges.crossing

    # check the lattice has periodic boundary conditions in the chosen directions
    for xy in range(2):
        if x_y_contacts[xy]:
            if np.all(crossing[:, xy] == 0):
                raise ValueError(
                    f"Lattice does not have periodic boundary conditions in direction {xy}"
                )

    # remove diagonal edges
    diagonal_edges = (crossing[:, 0] != 0) * (crossing[:, 1] != 0)
    edges = edges[~diagonal_edges]
    crossing = crossing[~diagonal_edges]

    # flip edges with negative crossing
    for xy in range(2):
        edges[crossing[:, xy] < 0] = edges[crossing[:, xy] < 0][:, ::-1]
        crossing[crossing[:, xy] < 0] = -crossing[crossing[:, xy] < 0]

    # rescale_vertices
    L_eff = np.sqrt(lattice.n_edges)
    rescaling = (L_eff - 4) / L_eff
    vertices = (vertices - 0.5) * rescaling + 0.5

    # remove crossing edges
    edges_out = edges[np.sum(np.abs(crossing), axis=1) == 0]
    crossing_out = crossing[np.sum(np.abs(crossing), axis=1) == 0]
    added_indices_out = []
    for xy in range(2):
        if not x_y_contacts[xy]:
            continue

        # find crossing edges and their adjacent vertices
        crossing_edges = edges[crossing[:, xy] > 0]
        starting_vertices = np.unique(crossing_edges[:, 0])
        ending_vertices = np.unique(crossing_edges[:, 1])

        starting_positions = vertices[starting_vertices]
        ending_positions = vertices[ending_vertices]

        # create new vertices
        new_starting_vertex_positions = starting_positions
        new_ending_vertex_positions = ending_positions
        new_starting_vertex_positions[:, xy] = 1 / L_eff
        new_ending_vertex_positions[:, xy] = 1 - 1 / L_eff

        # evenly space the new vertices
        new_starting_order = np.argsort(new_starting_vertex_positions[:, 1 - xy])
        new_ending_order = np.argsort(new_ending_vertex_positions[:, 1 - xy])
        if make_uniform:
            new_starting_vertex_positions[new_starting_order, 1 - xy] = np.linspace(
                np.min(new_starting_vertex_positions[:, 1 - xy]),
                np.max(new_starting_vertex_positions[:, 1 - xy]),
                len(new_starting_vertex_positions),
            )
            new_ending_vertex_positions[new_ending_order, 1 - xy] = np.linspace(
                np.min(new_ending_vertex_positions[:, 1 - xy]),
                np.max(new_ending_vertex_positions[:, 1 - xy]),
                len(new_ending_vertex_positions),
            )

        # add new vertices to the lattice
        vertices = np.concatenate(
            [vertices, new_starting_vertex_positions, new_ending_vertex_positions]
        )
        new_vertex_indices = np.arange(
            len(vertices)
            - len(new_starting_vertex_positions)
            - len(new_ending_vertex_positions),
            len(vertices),
        )

        # add new edges to the lattice
        remap_starting = dict(
            zip(starting_vertices, new_vertex_indices[: len(starting_vertices)])
        )
        remap_ending = dict(
            zip(ending_vertices, new_vertex_indices[len(starting_vertices) :])
        )
        new_starting_edges = crossing_edges.copy()
        new_starting_edges[:, 0] = [remap_starting[v] for v in new_starting_edges[:, 0]]
        new_ending_edges = crossing_edges.copy()
        new_ending_edges[:, 1] = [remap_ending[v] for v in new_ending_edges[:, 1]]
        edges_out = np.concatenate([edges_out, new_starting_edges, new_ending_edges])

        starting_indices = np.arange(
            len(vertices)
            - len(new_starting_vertex_positions)
            - len(new_ending_vertex_positions),
            len(vertices) - len(new_ending_vertex_positions),
        )[new_starting_order]
        ending_indices = np.arange(
            len(vertices) - len(new_ending_vertex_positions), len(vertices)
        )[new_ending_order]

        added_indices_out.append((starting_indices, ending_indices))

        # add perpendicular to the new vertices if cross_edges is True
        if cross_edges:
            cross_starting_edges = np.array(
                [starting_indices[:-1], starting_indices[1:]]
            )
            cross_ending_edges = np.array([ending_indices[:-1], ending_indices[1:]])

            edges_out = np.concatenate(
                [edges_out, cross_starting_edges.T, cross_ending_edges.T]
            )

    if len(added_indices_out) == 1:
        added_indices_out = added_indices_out[0]
    added_indices_out = tuple(added_indices_out)

    crossing_out = np.zeros((len(edges_out), 2), dtype=int)

    if return_added_indices:
        return Lattice(vertices, edges_out, crossing_out), added_indices_out
    return Lattice(vertices, edges_out, crossing_out)
