import glob
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import wntr

# to keep all possible solutions we would get in one place
# just remeber to do sys.path.append(".") if you are in /notebook or .. if somewhere else
# otherwise just run it from the parent repo :)
SOLUTIONS = "results/solutions"


# save solution as a json format

def save_solution(network, solver, formulation, k, sensors, energy, runtime_s, params=None, samples=None,
                  sample_energies=None):


    """
    to save one solver run as a json file in results/solutions.

    usage:

    save_solution(
        network="Net1",
        solver="gurobi",
        formulation="topological",
        k=3,
        sensors=sol_g,
        energy=energy_g,
        runtime_s=dt_g,
    )
    Params:
    network      the network in question (Net1, Net2, etc)
    solver       which solver? (gurobi, qaoa..)
    formulation  which qubo was used? "topological" or "leak"
    k            number of sensors
    sensors      junctions with a placed sensor
    energy       lowest energy
    runtime_s    wall clock time in seconds
    params       (optional) any settings
    samples      (optional) all the placements the solver returned
    sample_energies  (optional) the energy of each of those placements

    running the same network, formulation, solver and k again replaces the old file.
    """


    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    solution = {
        "network": network,
        "solver": solver,
        "formulation": formulation,
        "k": k,
        "sensors": [str(s) for s in sensors],
        "energy": float(energy),
        "runtime_s": float(runtime_s),
        "params": params or {},
        "samples": samples or [],
        "sample_energies": sample_energies or [],
        "time": time,
    }
    os.makedirs(SOLUTIONS, exist_ok=True)
    path = f"{SOLUTIONS}/{network}_{formulation}_{solver}_k{k}.json"
    with open(path, "w") as f:
        json.dump(solution, f, indent=2)
    return path


def load_solution(path):
    """read a saved solution back as a dictionary."""
    with open(path) as f:
        return json.load(f)


def load_all_solutions():
    """every saved solution in one table, one row per solver run."""
    paths = sorted(glob.glob(f"{SOLUTIONS}/*.json"))
    return pd.DataFrame([load_solution(p) for p in paths])


def plot_network(network, sensors=(), ax=None, title=None, labels=True):
    """draw the network. junctions are dots, tanks and reservoirs are squares, sensors are stars."""
    wn = wntr.network.WaterNetworkModel(f"networks/{network}.inp")
    pos = {name: wn.get_node(name).coordinates for name in wn.node_name_list}
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    for _, link in wn.links():
        a, b = pos[link.start_node_name], pos[link.end_node_name]
        ax.plot([a[0], b[0]], [a[1], b[1]], color="lightgray", zorder=1)

    for name in wn.node_name_list:
        x, y = pos[name]
        if name in sensors:
            ax.scatter(x, y, s=250, marker="*", color="tab:red", edgecolor="black", zorder=3)
        elif name in wn.junction_name_list:
            ax.scatter(x, y, s=40, color="tab:blue", zorder=2)
        else:
            ax.scatter(x, y, s=60, marker="s", color="gray", zorder=2)
        if labels:
            ax.annotate(name, (x, y), xytext=(6, 6), textcoords="offset points", fontsize=9)

    ax.set_title(title or network)
    ax.axis("off")
    return ax