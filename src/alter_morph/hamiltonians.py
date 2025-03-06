from koala.lattice import Lattice
import numpy as np


def _hopping_matrix(t1, t2, theta, theta_offset=0.0, n=1):
    # val and perus ansatz for the orbital resolved hopping matrix
    theta = theta + theta_offset
    tdiff = t1 - t2
    arr = np.array(
        [
            [
                tdiff * np.cos(n * theta) * np.cos(n * theta) + t2,
                tdiff * np.cos(n * theta) * np.sin(n * theta),
            ],
            [
                -tdiff * np.cos(n * theta) * np.sin(n * theta),
                tdiff * np.sin(n * theta) * np.sin(n * theta) + t2,
            ],
        ]
    )
    
    return -arr


def alt_hamiltonian(
    lattice: Lattice,
    t1: float,
    t2: float,
    J: float,
    U: float,
    m_values: np.ndarray,
    n_values: np.ndarray,
    theta_offset=0.0,
    boundary_phase=None,
    add_energy_shift=False,
):
    """Generates an altermagnetic Hamiltonian for a given lattice

    Args:
        lattice (Lattice): The lattice to generate the Hamiltonian for
        t1 (float): Strong hopping parameter where direction matches orbital
        t2 (float): Weak hopping parameter where direction opposes orbital
        J (float): Interacting coupling parameter
        m_values (np.ndarray): Mean field values for the magnetic moments per site
        boundary_phase (np.ndarray, optional): Phase for twisted boundary conditions in x and y. Defaults to 0.

    Returns:
        np.ndarray: The Hamiltonian matrix
    """

    # this basis should be (up x, up y, down x, down y)

    # initialize the Hamiltonian
    ham_type = (
        complex
        if boundary_phase is not None and np.any(np.nonzero(boundary_phase))
        else float
    )
    ham = np.zeros((4 * lattice.n_vertices, 4 * lattice.n_vertices), dtype=ham_type)

    # generate the hopping terms
    for n_edge in range(lattice.n_edges):
        vector = lattice.edges.vectors[n_edge]
        crossing = lattice.edges.crossing[n_edge]
        e0, e1 = lattice.edges.indices[n_edge]
        theta = np.arctan2(vector[0], vector[1])
        h0_term = np.kron(np.eye(2), _hopping_matrix(t1, t2, theta, theta_offset))

        # and apply boundary phase if twisting boundaries
        if (
            boundary_phase is not None
            and np.any(np.nonzero(crossing))
            and np.any(np.nonzero(boundary_phase))
        ):
            phase = np.sum(boundary_phase * crossing)
            h0_term = h0_term * np.exp(1j * phase)

        ham[4 * e0 : 4 * e0 + 4, 4 * e1 : 4 * e1 + 4] = h0_term
    ham += ham.T.conj()

    # generate the onsite interaction based terms
    for n in range(lattice.n_vertices):
        vertex_neighbours = lattice.vertices.adjacent_vertices[n]
        neighbour_magnetisations = m_values[vertex_neighbours]
        neighbour_densities = n_values[vertex_neighbours]

        total_magnetisation = np.sum(neighbour_magnetisations)
        total_density = np.sum(neighbour_densities)

        ham[4 * n : 4 * n + 4, 4 * n : 4 * n + 4] += (
            np.diag(J * total_magnetisation *np.array([-1, 1, 1, -1])
                     + U * total_density)
        )
    
    if add_energy_shift:
        edges = lattice.edges.indices
        mag_prod = m_values[edges[:, 0]] * m_values[edges[:, 1]]
        n_prod = n_values[edges[:, 0]] * n_values[edges[:, 1]]
        energy_mean_field_shift = np.sum(mag_prod) * J - np.sum(n_prod) * U
        energy_mean_field_shift = energy_mean_field_shift / (lattice.n_vertices*4)
        # print(energy_mean_field_shift)
        diag_norm = np.eye(4 * lattice.n_vertices)
        ham += energy_mean_field_shift * diag_norm

    return ham


def find_m_and_n_values(states, filling):
    """Given a set of eigenvectors, calculates the mean field values for the altermagnetic magnetic moments

    Args:
        v (np.ndarray): The eigenvectors
        filling (float): The filling fraction

    Returns:
        np.ndarray: The spatially resolved mean field values
    """

    fermi_occupation = np.linspace(0, 1, len(states)) <= filling
    local_densities = np.sum(states * states.conj() * fermi_occupation, axis=1)

    m_legend = np.tile(np.array([1, -1, -1, 1]), len(states) // 4)
    m_values = local_densities * m_legend
    m_values = m_values.reshape(-1, 4).sum(axis=-1)

    n_values = local_densities.reshape(-1, 4).sum(axis=-1)

    # check for complex values - this should not happen
    if not np.allclose(m_values.imag, 0):
        raise ValueError(
            "Magnetization values are complex - somewhere, somehow you fucked it up"
        )
    elif not np.allclose(n_values.imag, 0):
        raise ValueError(
            "Density values are complex - somewhere, somehow you fucked it up"
        )
    

    return m_values.real, n_values.real


def find_m_per_state(v: np.ndarray):
    """
    Given a set of eigenvectors, calculates the average mean
    field values for the altermagnetic magnetic moments per eigenstate

    Args:
        v (np.ndarray): The eigenvectors

    Returns:
        np.ndarray: The average mean field values per eigenstate
    """

    m_legend = np.array([1, -1, -1, 1] * (len(v) // 4))
    occupations = v * v.conj()
    m_x_s_resolved = m_legend[:, None] * occupations
    return np.sum(m_x_s_resolved, axis=0)


def find_spin_per_state(v):
    """
    Given a set of eigenvectors, calculates the average spin expectation
    value per eigenstate

    Args:
        v (np.ndarray): The eigenvectors

    Returns:
        np.ndarray: The average spin expectation value per eigenstate
    """

    spin_legend = np.array([1, -1, 1, -1] * (len(v) // 4))
    occupations = v * v.conj()
    spin_x_s_resolved = spin_legend[:, None] * occupations
    return np.sum(spin_x_s_resolved, axis=0)


def fermi_probability(fermi_spectrum, beta):
    """
    Calculates the fermionic probability distribution.

    Args:
        fermi_spectrum (numpy.ndarray): Array of energies.
        beta (float): Inverse temperature.

    Returns:
        numpy.ndarray: Array of probabilities.
    """
    energies_pos = fermi_spectrum * (fermi_spectrum > 0)
    energies_neg = fermi_spectrum * (fermi_spectrum <= 0)

    # this is stable when energies are positive
    f_pos = 1 - 1 / (1 + np.exp(-beta * energies_pos))
    # this is stable when energies are negative
    f_neg = 1 / (1 + np.exp(beta * energies_neg))
    f = f_pos * (fermi_spectrum > 0) + f_neg * (fermi_spectrum <= 0)

    return f


# useful for testing
def qwz_hamiltonian(lattice: Lattice, u_vals=-1.465, h_field=0):
    """Generates the Hamiltonian for the Qi-Wu-Zhang model

    Args:
        lattice (Lattice): The lattice to generate the Hamiltonian for
        u_vals (float, optional): The on-site staggered energy. Defaults to -1.465.
        h_field (float, optional): The applied magnetic field. Defaults to 0.

    Returns:
        np.ndarray: The Hamiltonian
    """

    if isinstance(u_vals, (int, float, complex)):
        u_vals = np.array([u_vals] * lattice.n_vertices)

    a_site_part = np.diag(u_vals + 0j)
    b_site_part = -np.diag(u_vals + 0j)

    ab_link_part = np.zeros((lattice.n_vertices, lattice.n_vertices), dtype=complex)
    ba_link_part = np.zeros((lattice.n_vertices, lattice.n_vertices), dtype=complex)

    p1 = lattice.vertices.positions[lattice.edges.indices[:, 0]]
    p2 = lattice.vertices.positions[lattice.edges.indices[:, 1]]
    average_y = 0.5 * (p1[:, 1] + p2[:, 1])
    change_x = p2[:, 0] - p1[:, 0]
    phases = -h_field * average_y * change_x * lattice.n_plaquettes

    for edge_index in range(lattice.n_edges):
        field_phase = 1j * phases[edge_index] * 2 * np.pi

        edge = lattice.edges.indices[edge_index]

        vec = lattice.edges.vectors[edge_index]
        angle = np.angle(vec[0] + 1j * vec[1])

        ab_link_part[edge[0], edge[1]] = 0.5j * np.exp(-1j * angle + field_phase)
        ba_link_part[edge[0], edge[1]] = 0.5j * np.exp(1j * angle + field_phase)
        a_site_part[edge[0], edge[1]] = 0.5 * np.exp(field_phase)
        b_site_part[edge[0], edge[1]] = -0.5 * np.exp(field_phase)

        # add conjugates
        ba_link_part[edge[1], edge[0]] = -0.5j * np.exp(1j * angle - field_phase)
        ab_link_part[edge[1], edge[0]] = -0.5j * np.exp(-1j * angle - field_phase)
        a_site_part[edge[1], edge[0]] = 0.5 * np.exp(-field_phase)
        b_site_part[edge[1], edge[0]] = -0.5 * np.exp(-field_phase)

    return np.block([[a_site_part, ab_link_part], [ba_link_part, b_site_part]])


def kubo_conductivity(lattice, hamiltonian, energies, states, fermi_level, beta, omega):
    """Calculates the Kubo conductivity for a given lattice

    Args:
        lattice (Lattice): The lattice to calculate the conductivity for
        energies (np.ndarray): The energies of the states
        states (np.ndarray): The eigenvectors of the Hamiltonian
        occupation (np.ndarray): The occupation of the states

    Returns:
        float: The Kubo conductivity
    """

    occupation = fermi_probability(energies - fermi_level, beta)
    fn_fm = occupation[:, np.newaxis] - occupation
    energy_diffs = energies[:, None] - energies
    energy_diffs[energy_diffs == 0] = 1e-10
    energy_diffs_plus_omega = energy_diffs + omega + 1e-10j
    efactor = 1 / (energy_diffs * energy_diffs_plus_omega)
    overall_factor = efactor * fn_fm

    on_site_degeneracy = len(energies) // lattice.n_vertices

    x_diff_matrix = np.zeros((lattice.n_vertices, lattice.n_vertices))
    y_diff_matrix = np.zeros((lattice.n_vertices, lattice.n_vertices))
    x_diff_matrix[lattice.edges.indices[:, 0], lattice.edges.indices[:, 1]] = (
        lattice.edges.vectors[:, 0]
    )
    y_diff_matrix[lattice.edges.indices[:, 0], lattice.edges.indices[:, 1]] = (
        lattice.edges.vectors[:, 1]
    )
    x_diff_matrix -= x_diff_matrix.T
    y_diff_matrix -= y_diff_matrix.T
    x_diff_matrix = np.kron(
        np.ones([on_site_degeneracy, on_site_degeneracy]),
        x_diff_matrix,
    )
    y_diff_matrix = np.kron(
        np.ones([on_site_degeneracy, on_site_degeneracy]),
        y_diff_matrix,
    )
    jx = 1j * hamiltonian * x_diff_matrix
    jy = 1j * hamiltonian * y_diff_matrix
    jx_in_energy_basis = states.conj().T @ jx @ states
    jy_in_energy_basis = states.conj().T @ jy @ states

    xxmatrix = (
        overall_factor * jx_in_energy_basis
    ) @ jx_in_energy_basis  # - jx_in_energy_basis @ (overall_factor * jx_in_energy_basis)
    yymatrix = (
        overall_factor * jy_in_energy_basis
    ) @ jy_in_energy_basis  # - jy_in_energy_basis @ (overall_factor * jy_in_energy_basis)
    xymatrix = (
        overall_factor * jx_in_energy_basis
    ) @ jy_in_energy_basis  # - jx_in_energy_basis @ (overall_factor * jy_in_energy_basis)
    yxmatrix = (
        overall_factor * jy_in_energy_basis
    ) @ jx_in_energy_basis  # - jy_in_energy_basis @ (overall_factor * jx_in_energy_basis)

    sigma_xx = np.trace(xxmatrix)
    sigma_yy = np.trace(yymatrix)
    sigma_xy = np.trace(xymatrix)
    sigma_yx = np.trace(yxmatrix)

    return -np.array([[sigma_xx, sigma_xy], [sigma_yx, sigma_yy]]) * 2j


def spectral_function_reshape(
    lattice: Lattice,
    energies: np.ndarray,
    states: np.ndarray,
    omega: float,
    eta=1e-6,
    local_operator=np.array([1, 1, 1, 1]),
    n_k=None,
):
    """
    Calculates the spectral function for a given set of energies
    and states, projected onto a local set of spin and orbital states.

    This method is based on reshaping in orbital and real subspace and should in principle be very fast. It is however very slow.

    Args:
        lattice (Lattice): The lattice
        energies (np.ndarray): The eigenvalues of the Hamiltonian
        states (np.ndarray): The eigenvectors of the Hamiltonian
        omega (float): The energy to calculate the spectral function at
        eta (float, optional): The broadening parameter for the spectral function. Defaults to 1e-6.
        local_operator (np.ndarray, optional): An operator living in the orbital space. Defaults to np.array([1, 1, 1, 1]).
        n_k (int, optional): The number of k points to sample, if None will be sqrt(n_vertices). Defaults to None.

    Returns:
        np.ndarray: The spectral function
    """

    if n_k is None:
        n_k = np.sqrt(lattice.n_vertices).astype(int)
        # print(n_k)

    positions = lattice.vertices.positions
    n_vertices = lattice.n_vertices
    n_orbitals = len(local_operator)

    ########################
    # approach based on reshaping
    states = states.reshape(
        n_vertices, n_orbitals, n_vertices * n_orbitals
    )  # .shape=(n_vertices,n_orbitals,n_vertices*n_orbitals)

    k_vals = np.arange(-n_k // 2, n_k // 2) * 2 * np.pi
    ks = np.array(np.meshgrid(k_vals, k_vals))  # .shape=(2,ky,kx)

    phase = np.exp(
        1j * np.einsum("ryx,nr->yxn", ks, positions)
    )  # .shape=(ky,kx,n_vertices)
    k_states = np.einsum(
        "yxn,noh->yxoh", phase, states
    )  # .shape=(ky,kx,n_orbitals,n_vertices*n_orbitals)
    k_densities = (
        np.abs(k_states) ** 2
    )  # .shape=(ky,kx,n_orbitals,n_vertices*n_orbitals)

    Ak = (
        eta / np.pi / ((energies - omega) ** 2 + eta**2) / n_vertices / n_orbitals
    )  # .shape=(n_vertices*n_orbitals)
    Ako = np.einsum("yxoh,h->yxo", k_densities, Ak)  # .shape=(ky,kx,n_orbitals)
    spectral_function = np.einsum("yxo,o->yx", Ako, local_operator)  # .shape=(ky,kx)

    return spectral_function


def spectral_function(
    lattice: Lattice,
    energies: np.ndarray,
    states: np.ndarray,
    omega: float,
    eta=1e-6,
    local_operator=np.array([1, 1, 1, 1]),
    n_k=None,
):
    """
    Calculates the spectral function for a given set of energies
    and states, projected onto a local set of spin and orbital states

    Args:
        lattice (Lattice): The lattice
        energies (np.ndarray): The eigenvalues of the Hamiltonian
        states (np.ndarray): The eigenvectors of the Hamiltonian
        omega (float): The energy to calculate the spectral function at
        eta (float, optional): The broadening parameter for the spectral function. Defaults to 1e-6.
        local_operator (np.ndarray, optional): An operator living in the orbital space. Defaults to np.array([1, 1, 1, 1]).
        n_k (int, optional): The number of k points to sample, if None will be sqrt(n_vertices). Defaults to None.

    Returns:
        np.ndarray: The spectral function
    """

    if n_k is None:
        n_k = np.sqrt(lattice.n_vertices).astype(int)
        # print(n_k)

    positions = lattice.vertices.positions
    n_vertices = lattice.n_vertices
    n_orbitals = len(local_operator)
    #######################
    k_vals = np.arange(-n_k // 2, n_k // 2) * 2 * np.pi
    ks = np.array(np.meshgrid(k_vals, k_vals))  # .shape=(2,ky,kx)

    Ak = (
        eta / np.pi / ((energies - omega) ** 2 + eta**2) / n_vertices / n_orbitals
    )  # .shape=(n_vertices*n_orbitals)
    phase = np.exp(
        1j * np.einsum("ryx,nr->yxn", ks, positions)
    )  # .shape=(ky,kx,n_vertices)

    # 1D kron of last axis
    kronO = np.einsum(
        "ayxn,ao->yxno", phase[np.newaxis], local_operator[np.newaxis]
    ).reshape(*phase.shape[:-1], phase.shape[-1] * local_operator.shape[-1])
    kron1 = np.einsum(
        "ayxn,ao->yxno", phase[np.newaxis], np.ones(n_orbitals)[np.newaxis]
    ).reshape(*phase.shape[:-1], phase.shape[-1] * local_operator.shape[-1])

    # direct summation
    # print(np.einsum_path('b,cb,xyc,yxa,ab->yx',Ak,states.conj(),kron1.conj(),kronO,states,optimize='optimal'))
    spectral_function = np.einsum(
        "b,cb,yxc,yxa,ab->yx",
        Ak,
        states.conj(),
        kron1.conj(),
        kronO,
        states,
        optimize=["einsum_path", (0, 1), (2, 3), (0, 2), (0, 1)],
    )

    return np.real(spectral_function)  # elements should be real anyways


def spectral_function_old(
    lattice: Lattice,
    energies: np.ndarray,
    states: np.ndarray,
    omega: float,
    eta=1e-6,
    local_projector=np.array([1, 1, 0, 0]),
    n_k=None,
):
    """
    Calculates the spectral function for a given set of energies
    and states, projected onto a local set of spin and orbital states

    This method involves a direct summation over the k points and is very slow.

    Args:
        lattice (Lattice): The lattice
        energies (np.ndarray): The eigenvalues of the Hamiltonian
        states (np.ndarray): The eigenvectors of the Hamiltonian
        omega (float): The energy to calculate the spectral function at
        eta (float, optional): The broadening parameter for the spectral function. Defaults to 1e-6.
        local_projector (np.ndarray, optional): The local set of spin/orbitals to project the states onto. Defaults to np.array([1, 1, 0, 0]).
        n_k (int, optional): The number of k points to sample, if None will be sqrt(n_vertices). Defaults to None.

    Returns:
        np.ndarray: The spectral function
    """

    if n_k is None:
        n_k = np.sqrt(lattice.n_vertices).astype(int)
        # print(n_k)

    positions = lattice.vertices.positions

    k_vals = np.arange(-n_k // 2, n_k // 2) * 2 * np.pi

    e_inv = 1 / (energies - omega + 1j * eta)
    resolvent = states * e_inv @ states.conj().T

    spectral_function = np.zeros((len(k_vals), len(k_vals)), dtype=complex)

    for idx, kx in enumerate(k_vals):
        for idy, ky in enumerate(k_vals):
            k_state = np.exp(1j * (kx * positions[:, 0] + ky * positions[:, 1]))
            # k_state = k_state / np.linalg.norm(k_state)
            projected_state = np.kron(k_state, local_projector)
            spectral_function[idx, idy] = (
                projected_state.conj() @ resolvent @ projected_state
            )

    return spectral_function
