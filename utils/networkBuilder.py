import numpy as np
import networkx as nx
from scipy.stats import nbinom
import matplotlib.pyplot as plt
import secrets
# ---------------- Utilidades para generar grafos ----------------
def generate_degrees(n, mean_k, var_k):
    """Genera una secuencia de grados con media y varianza aproximadas.
    This function's primary role is to create a sequence of degrees for a graph with $n$ nodes,
    ensuring the sequence adheres to a desired average degree ($\text{mean\_k}$) and
    degree variability ($\text{var\_k}$).
    It achieves this by sampling from a Negative Binomial distribution
    (a common choice for modeling real-world networks) and then performs necessary clean-up steps:
    it clips the degrees to be positive and less than $n$, and it adjusts one degree
    by $\pm 1$ if the total sum of degrees is odd, which is a mathematical requirement
    for a simple graph to exist.
    """


    if var_k <= mean_k:
        raise ValueError("La varianza debe ser mayor que la media para la Negativa Binomial.")
    p = mean_k / var_k
    r = mean_k * p / (1 - p)
    degs = nbinom(r, p).rvs(size=n)
    degs = np.clip(degs, 1, n-1)
    if np.sum(degs) % 2 == 1:
        idx = np.random.randint(0, n)
        degs[idx] += 1 if degs[idx] < n-1 else -1
    return degs

def ensure_connected(G):
    """Ensure the graph is connected by linking components if necessary.
    This function guarantees that the input graph $G$ is a single, unified component.
    It first checks if the graph is already connected. If it is not, it iterates through all
    separate connected components, randomly selecting one node from each pair of adjacent
    components and adding a single edge between them until the entire graph is connected.
    This ensures that the final graph structure is ready for analysis or simulation that
    requires connectivity.


    """
    if nx.is_connected(G):
        return G

    components = list(nx.connected_components(G))
    # Connect components sequentially with random nodes from each
    for i in range(len(components) - 1):
        u = np.random.choice(list(components[i]))
        v = np.random.choice(list(components[i + 1]))
        G.add_edge(u, v)

    assert nx.is_connected(G)
    return G

def generate_connected_graph(n=500, mean_k=6, var_k=20, seed=None):
    """Genera un grafo conexo con la secuencia de grados objetivo.
    This is the main execution function that generates a complete, connected graph with
    specific size and degree properties. It first sets the random seed for reproducibility.
    It then calls generate_degrees to create the target degree sequence and uses
    a network construction algorithm like $\text{Havel-Hakimi}$ or
    $\text{Configuration Model}$ (from $\text{NetworkX}$) to build the graph based on those degrees.
    Finally, it calls ensure_connected
    to guarantee the resulting graph is usable, returning the final, fully-constructed graph $G$.
    """
    np.random.seed(seed)
    degrees = generate_degrees(n, mean_k, var_k)
    try:
        G = nx.havel_hakimi_graph(degrees)
    except nx.NetworkXError:
        G = nx.configuration_model(degrees)
        G = nx.Graph(G)
        G.remove_edges_from(nx.selfloop_edges(G))
    G = ensure_connected(G)
    return G

def generate_secure_seed():
    """
    Generates a cryptographically secure, 32-bit integer.
    This range (0 to 2**32 - 1) is fully compatible with np.random.seed().
    """
    return secrets.randbits(32)