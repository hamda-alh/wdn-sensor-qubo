import json
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import wntr

from src.metrics import is_feasible, feasibility_rate, leak_delays, detection_rate, mean_delay
from src.simulate_leaks import load_leak_table
from src.utils import load_solution


BENCHMARKS = "results/benchmarks"

def plot_delays(network, sensors, delays, path):
    """draw the network, color each junction by how long a leak there takes to be noticed."""
    wn = wntr.network.WaterNetworkModel(f"networks/{network}.inp")
    x = pd.Series({name: wn.get_node(name).coordinates[0] for name in wn.node_name_list})
    y = pd.Series({name: wn.get_node(name).coordinates[1] for name in wn.node_name_list})

    fig, ax = plt.subplots(figsize=(7, 6))
    for _, link in wn.links():
        a, b = link.start_node_name, link.end_node_name
        ax.plot([x[a], x[b]], [y[a], y[b]], color="lightgray", linewidth=1, zorder=1)

    missed = delays[delays.isna()].index
    caught = delays[delays.notna()].index
    ax.scatter(x[missed], y[missed], s=25, marker="x", color="gray", label="leak not detected", zorder=2)
    dots = ax.scatter(x[caught], y[caught], s=60, c=delays[caught], cmap="Blues_r", vmin=0,
                      edgecolor="black", label="leak detected", zorder=3)
    ax.scatter(x[sensors], y[sensors], s=220, marker="*", color="black", label="sensor", zorder=4)

    fig.colorbar(dots, label="minutes until detected")
    ax.legend()
    ax.axis("off")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def benchmark(solution_path):
    solution = load_solution(solution_path)
    network, k, sensors = solution["network"], solution["k"], solution["sensors"]

    table = load_leak_table(network)
    delays = leak_delays(table, sensors)

    name = os.path.basename(solution_path).replace(".json", "")
    metrics = {
        "solution": name,
        "network": network,
        "solver": solution["solver"],
        "formulation": solution["formulation"],
        "k": k,
        "sensors": " ".join(sensors),
        "feasible": is_feasible(sensors, k),
        "feasibility_rate": feasibility_rate(solution["samples"], k),
        "energy": solution["energy"],
        "runtime_s": solution["runtime_s"],
        "detection_rate": detection_rate(delays),
        "mean_delay_min": mean_delay(delays),
    }

    # one folder per solution
    folder = f"{BENCHMARKS}/{name}"
    os.makedirs(folder, exist_ok=True)
    with open(f"{folder}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    delays.rename("delay_min").to_csv(f"{folder}/delays.csv")
    plot_delays(network, sensors, delays, f"{folder}/delays.png")

    # one row per solution in a shared table, so runs are easy to compare
    summary_path = f"{BENCHMARKS}/summary.csv"
    summary = pd.DataFrame([metrics])
    if os.path.exists(summary_path):
        old = pd.read_csv(summary_path)
        summary = pd.concat([old[old["solution"] != name], summary])
    summary.to_csv(summary_path, index=False)

    return metrics


if __name__ == "__main__":
    for solution_path in sys.argv[1:]:
        metrics = benchmark(solution_path)
        for key, value in metrics.items():
            print(f"{key:18} {value}")
        print()