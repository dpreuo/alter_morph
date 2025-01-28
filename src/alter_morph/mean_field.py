from koala.lattice import Lattice
from .hamiltonians import alt_hamiltonian, find_m_values
import numpy as np
from scipy import linalg as la
from tqdm import tqdm


def single_hartree_fock_step(
    lattice: Lattice,
    initial_parameters: dict,
    m_values: np.ndarray,
    boundary_phase=None,
    uniform_m=False,
):
    # make and solve the Hamiltonian
    hamiltonian = alt_hamiltonian(
        lattice,
        initial_parameters["t1"],
        initial_parameters["t2"],
        initial_parameters["J"],
        m_values,
        theta_offset=initial_parameters["theta_offset"],
        boundary_phase=boundary_phase,
    )
    energies, states = la.eigh(hamiltonian)

    # calculate the new magnetization values
    m_values = find_m_values(states, initial_parameters["filling"])

    # also have the option to average the magnetization values
    if uniform_m:
        m_values = np.mean(m_values) * np.ones_like(m_values)

    return m_values


def hartree_fock(
    lattice: Lattice,
    initial_parameters: dict,
    n_steps: int,
    mixing_proportion=0.3,
    **kwargs,
):

    prange = tqdm(range(n_steps))
    skip_counter = 0

    m_values = np.zeros((n_steps + 1, lattice.n_vertices))
    m_values[0] = initial_parameters["initial_m"]

    for n in prange:
        m_values[n + 1] = (1 - mixing_proportion) * single_hartree_fock_step(
            lattice, initial_parameters, m_values[n], **kwargs
        ) + mixing_proportion * m_values[n]

        # check for convergence
        diff = np.linalg.norm(m_values[n + 1] - m_values[n])
        avg_m = np.mean(m_values[n + 1])
        prange.set_description(f"Avg:{avg_m:.2f}, diff: {diff:.6f}")

        # if the difference is small enough, we can stop
        if diff < 1e-6:
            skip_counter += 1
            if skip_counter >= 3:
                m_values = m_values[: n + 1]
                break

    return m_values


def hartree_fock_with_boundary_twisting(
    lattice: Lattice,
    initial_parameters: dict,
    n_steps: int,
    n_twists: int,
    mixing_proportion=0.3,
    **kwargs
):
    prange = tqdm(range(n_steps))
    skip_counter = 0

    m_values = np.zeros((n_steps + 1, lattice.n_vertices))
    m_values[0] = initial_parameters["initial_m"]

    k_values = np.linspace(0, 2 * np.pi, n_twists, endpoint=False)
    KX, KY = np.meshgrid(k_values, k_values)
    boundary_phases = np.stack([KX.flatten(), KY.flatten()], axis=-1)

    for n in prange:

        averaged_m = np.zeros(lattice.n_vertices)
        for k_val in boundary_phases:
            averaged_m += single_hartree_fock_step(
                lattice, initial_parameters, m_values[n], k_val, **kwargs
            )
        m_found = averaged_m / len(boundary_phases)
        m_values[n + 1] = m_found*(1 - mixing_proportion) + m_values[n]*mixing_proportion

        # check for convergence
        diff = np.linalg.norm(m_values[n + 1] - m_values[n])
        avg_m = np.mean(m_values[n + 1])
        prange.set_description(f"Avg:{avg_m:.2f}, diff: {diff:.6f}")

        # if the difference is small enough, we can stop
        if diff < 1e-6:
            skip_counter += 1
            if skip_counter >= 3:
                m_values = m_values[: n + 1]
                break

    return m_values
