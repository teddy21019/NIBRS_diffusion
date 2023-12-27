import numpy as np

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
    nodes_n = np.sum(W, axis=1, dtype=np.float64)  # Size: N x 1 matrix since sum along axis = 1

    # Calculate the sum of neighbor states for each node over time
    neighbor_state_sum_dynamic = (W @ state_dynamic).astype(float)  # Size: N x T, where T is the number of time steps

    # Calculate the adoption rate of neighbors for each node over time
        # Use np.true divide to cast type
    neighbor_adopted_rate = np.divide(neighbor_state_sum_dynamic, nodes_n,
                                      out=np.zeros_like(neighbor_state_sum_dynamic),
                                      where=nodes_n != 0)

    # Determine nodes that should adopt based on neighbor adoption rate
    agency_should_adopt = neighbor_adopted_rate > th  # Size: N x T

    # Calculate the state of each node after potential adoption
    agency_state_after_adopt = agency_should_adopt | state_dynamic  # Size: N x T

    # Calculate the squared difference between dynamic state and state after potential adoption
    diff = np.sum(
        np.square(state_dynamic[:, 1:] - agency_state_after_adopt[:, :-1])
    )  # Squared difference, Result: Scalar

    return int(diff)
