import networkx as nx
import numpy as np
import wntr

from src.simulate_leaks import load_leak_table


cutoff = 30   # minutes. a sensor "detects" a leak if it notices it this fast


def load_network(network):
    return wntr.network.WaterNetworkModel(f"networks/{network}.inp")


def junction_demand(wn):
    """average water demand at every junction, scaled so the biggest is 1."""
    demand = wntr.metrics.expected_demand(wn).mean()
    # a junction that feeds water in has negative demand, count it as 0
    demand = demand.clip(lower=0)
    return (demand / demand.max()).values


def edge_weights(wn):
    """importance of every link = its edge betweenness, scaled so the biggest is 1."""
    G = nx.Graph()
    for name, link in wn.links():
        # pumps and valves have no length, so they count as 1
        G.add_edge(link.start_node_name, link.end_node_name, length=getattr(link, "length", 1.0))
    betweenness = nx.edge_betweenness_centrality(G, weight="length")
    biggest = max(betweenness.values())
    return {edge: value / biggest for edge, value in betweenness.items()}


def add_budget(Q, k, penalty):
    """add penalty * (number of sensors - k)^2 to Q and return the constant part."""
    N = len(Q)
    for i in range(N):
        Q[i, i] += penalty * (1 - 2 * k)
        for j in range(i + 1, N):
            Q[i, j] += 2 * penalty
    return penalty * k ** 2


def build_Q_topological(network, k, penalty=5.0, p_cost=0.5):
    """
    qubo 1 (from the paper): a link is covered if a sensor sits at one of its ends.

    energy = weight of uncovered links + p_cost * placement cost + budget penalty
    """
    wn = load_network(network)
    nodes = wn.junction_name_list
    idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    Q = np.zeros((N, N))
    const = 0.0

    # uncovered link costs w * (1 - xa) * (1 - xb) = w - w*xa - w*xb + w*xa*xb
    for (a, b), w in edge_weights(wn).items():
        if a in idx and b in idx:
            i, j = sorted([idx[a], idx[b]])
            const += w
            Q[i, i] -= w
            Q[j, j] -= w
            Q[i, j] += w
        elif a in idx or b in idx:
            # other end is a tank or reservoir, which cannot hold a sensor
            i = idx[a] if a in idx else idx[b]
            const += w
            Q[i, i] -= w

    # busy junctions are cheap (cost 0), quiet ones are expensive (cost 1)
    cost = 1 - junction_demand(wn)
    for i in range(N):
        Q[i, i] += p_cost * cost[i]

    const += add_budget(Q, k, penalty)
    return Q, const, nodes


def sensitivity_matrix(network):
    """
    S[i, j] = 1 if a sensor at junction j detects leak i within the cutoff.
    w[i] = how much leak i matters (demand at the leaking junction).
    """
    table = load_leak_table(network)
    S = (table <= cutoff).values.astype(int)
    w = junction_demand(load_network(network))
    return S, w, list(table.columns)


def build_Q_leak(network, k, penalty=5.0):
    """
    qubo 2 (ours): built from the simulated leaks.

    energy = sum over leaks of w * (1 - number of sensors that detect it)^2 + budget penalty
    """
    S, w, nodes = sensitivity_matrix(network)

    # G[j, k] = total weight of the leaks that both j and k detect
    G = S.T @ (w[:, None] * S)

    # w * (1 - n)^2 = w - w * sum_j S_j x_j + 2w * sum_{j<k} S_j S_k x_j x_k
    Q = 2 * np.triu(G, 1) - np.diag(np.diag(G))
    const = w.sum()

    const += add_budget(Q, k, penalty)
    return Q, const, nodes


def energy(Q, const, x):
    """energy of a placement x (list of 0s and 1s)."""
    x = np.array(x)
    return x @ Q @ x + const


def to_bits(sensors, nodes):
    """turn a list of sensor junctions into 0s and 1s."""
    return [1 if n in sensors else 0 for n in nodes]


def to_sensors(x, nodes):
    """turn 0s and 1s back into a list of sensor junctions."""
    return [n for n, bit in zip(nodes, x) if bit]