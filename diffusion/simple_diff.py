import numpy as np
import numpy.typing as npt
import networkx as nx
import numba


@numba.njit
def get_diffusion_diff(W:np.ndarray, state_dynamic: np.ndarray, th):
    """
    Calculate the squared difference between the dynamic state and the state after potential adoption
    based on a diffusion process over a network.

    Parameters:
        W : Adjacency matrix of the network
        state_dynamic (np.ndarray): Array representing the dynamic state of each node over time.

    Returns:
        float: Squared difference between the dynamic state and the state after potential adoption.
    """

    # Get the adjacency matrix of the network
    # Calculate the total number of neighbors for each node
    nodes_n = np.sum(W, axis=1, dtype=np.float64)[:,np.newaxis]  # Size: N x 1 matrix since sum along axis = 1

    # Calculate the sum of neighbor states for each node over time
    neighbor_state_sum_dynamic = W @ state_dynamic  # Size: N x T, where T is the number of time steps

    # Calculate the adoption rate of neighbors for each node over time
        # Use np.true divide to cast type
    neighbor_adopted_rate = np.nan_to_num(neighbor_state_sum_dynamic/nodes_n)

    # Determine nodes that should adopt based on neighbor adoption rate
    agency_should_adopt = neighbor_adopted_rate > th  # Size: N x T

    # Calculate the state of each node after potential adoption
    agency_state_after_adopt = agency_should_adopt | state_dynamic.astype(np.bool_)  # Size: N x T

    # Calculate the squared difference between dynamic state and state after potential adoption
    diff = np.sum(
        np.square(state_dynamic[:, 1:] - agency_state_after_adopt[:, :-1])
    )  # Squared difference, Result: Scalar

    return int(diff)

@numba.njit
def get_diffusion_diff_dynamic(W:np.ndarray, state_dynamic: np.ndarray, th):
    """
    Calculate the squared difference between the dynamic state and the state after potential adoption
    based on a diffusion process over a network.

    Parameters:
        W : Adjacency matrix of the network
        state_dynamic (np.ndarray): Array representing the dynamic state of each node over time.

    Returns:
        float: Squared difference between the dynamic state and the state after potential adoption.
    """

    # Get the adjacency matrix of the network
    # Calculate the total number of neighbors for each node
    nodes_n = np.sum(W, axis=1, dtype=np.float64)[:,np.newaxis]  # Size: N x 1 matrix since sum along axis = 1

    # Calculate the sum of neighbor states for each node over time
    neighbor_state_sum_dynamic = W @ state_dynamic  # Size: N x T, where T is the number of time steps

    # Calculate the adoption rate of neighbors for each node over time
        # Use np.true divide to cast type
    neighbor_adopted_rate = np.nan_to_num(neighbor_state_sum_dynamic/nodes_n)

    # Determine nodes that should adopt based on neighbor adoption rate
    agency_should_adopt = neighbor_adopted_rate > th  # Size: N x T

    # Calculate the state of each node after potential adoption
    agency_state_after_adopt = agency_should_adopt | state_dynamic.astype(np.bool_)  # Size: N x T

    # Calculate the squared difference between dynamic state and state after potential adoption
    diff = np.sum(
        np.square(state_dynamic[:, 1:] - agency_state_after_adopt[:, :-1]),
        axis=0
    )  # Squared difference, Result: Scalar

    return diff

def random_rewire(G:nx.Graph, th: float):
    if th<0 or th>1:
        raise ValueError("Threshold should be in (0,1)!")

    random_G = nx.random_degree_sequence_graph([d for n, d in G.degree()])
    rand_adj_matrix = nx.adjacency_matrix(random_G).toarray()
    return rand_adj_matrix

@numba.njit
def prediction_markov_diffusion_diff_dynamic(W:npt.NDArray[np.float_], state_dynamic:npt.NDArray[np.bool_], th:float):
    """
    The prediction depends solely on the initial realized state given by
    state_dynamic. For each period so on, the predicted state is the diffusion
    from previous prediction state.
    """

    n_nodes = state_dynamic.shape[0]        # number of agencies
    n_periods = state_dynamic.shape[1]      # number of periods including init state
    n_neighbors = np.sum(W, axis=1, dtype=np.float64)  # Size: N x 1 matrix since sum along axis = 1

    init_state = state_dynamic[:,0]
    state_prediction  = np.zeros_like(state_dynamic)
    state_prediction[:,0] = init_state

    for i in range(1,n_periods-1):
        prev_state = state_prediction[:, i-1]           # N x 1
        neighbors_state_sum = W @ prev_state            # (N x N) x (N x 1)
        neighbors_adopt_ratio = np.nan_to_num(neighbors_state_sum / n_neighbors)
        agency_should_adopt  = neighbors_adopt_ratio > th
        state_prediction[:, i] = agency_should_adopt

    return np.sum((state_prediction - state_dynamic)**2, axis=0)