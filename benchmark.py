import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import wntr

from src.metrics import (is_feasible, feasibility_rate, energy_gap, success_rate,
                         leak_delays, detection_rate, mean_delay)
from src.qubo import cutoff
from src.simulate_leaks import load_leak_table, leak_table_path
from src.utils import load_solution, SOLUTIONS


BENCHMARKS = "results/benchmarks"


def plot_delays(network, sensors, delays, path=None, ax=None, max_minutes=None):
    """
    draw the network, color each junction by how long a leak there takes to be noticed.
    give a path to save it as a file, or an ax to draw it inside another figure.
    max_minutes caps the color scale (everything slower gets the same color).
    """
    wn = wntr.network.WaterNetworkModel(f"networks/{network}.inp")
    x = pd.Series({name: wn.get_node(name).coordinates[0] for name in wn.node_name_list})
    y = pd.Series({name: wn.get_node(name).coordinates[1] for name in wn.node_name_list})

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    for _, link in wn.links():
        a, b = link.start_node_name, link.end_node_name
        ax.plot([x[a], x[b]], [y[a], y[b]], color="lightgray", linewidth=1, zorder=1)

    missed = delays[delays.isna()].index
    caught = delays[delays.notna()].index
    ax.scatter(x[missed], y[missed], s=25, marker="x", color="gray", label="leak not detected", zorder=2)
    dots = ax.scatter(x[caught], y[caught], s=60, c=delays[caught], cmap="Blues_r", vmin=0, vmax=max_minutes,
                      edgecolor="black", label="leak detected", zorder=3)
    ax.scatter(x[sensors], y[sensors], s=220, marker="*", color="tab:red", edgecolor="black", label="sensor", zorder=4)

    ax.figure.colorbar(dots, ax=ax, label="minutes until detected", extend="max" if max_minutes else "neither")
    ax.legend()
    ax.axis("off")
    if path:
        ax.figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(ax.figure)
    return ax


def gurobi_energy(network, formulation, k):
    """energy of the exact solution for the same problem, if we have it saved."""
    path = f"{SOLUTIONS}/{network}_{formulation}_gurobi_k{k}.json"
    if not os.path.exists(path):
        return None
    return load_solution(path)["energy"]


def benchmark(solution_path, plot=False):
    """score one saved solution. returns a dictionary of metrics."""
    solution = load_solution(solution_path)
    network, formulation, k, sensors = solution["network"], solution["formulation"], solution["k"], solution["sensors"]

    name = os.path.basename(solution_path).replace(".json", "")
    metrics = {
        "solution": name,
        "network": network,
        "solver": solution["solver"],
        "formulation": formulation,
        "k": k,
        "sensors": " ".join(sensors),
        "energy": solution["energy"],
        "runtime_s": solution["runtime_s"],
        "feasible": is_feasible(sensors, k),
        "feasibility_rate": feasibility_rate(solution["samples"], k),
    }

    # compare with the exact optimum
    best = gurobi_energy(network, formulation, k)
    if best is not None:
        metrics["energy_gap"] = energy_gap(solution["energy"], best)
        metrics["success_rate"] = success_rate(solution.get("sample_energies"), best)

    # leak detection on the normal network
    delays = leak_delays(load_leak_table(network), sensors)
    metrics["detection_rate"] = detection_rate(delays, cutoff)
    metrics["mean_delay_min"] = mean_delay(delays)

    # same thing on the noisy network (the one the qubo never saw)
    if os.path.exists(leak_table_path(network, noisy=True)):
        noisy_delays = leak_delays(load_leak_table(network, noisy=True), sensors)
        metrics["detection_rate_noisy"] = detection_rate(noisy_delays, cutoff)
        metrics["mean_delay_min_noisy"] = mean_delay(noisy_delays)

    if plot:
        os.makedirs(BENCHMARKS, exist_ok=True)
        plot_delays(network, sensors, delays, f"{BENCHMARKS}/{name}.png")

    return metrics


# python benchmark.py results/solutions/*.json                       -> one table: results/benchmarks/summary.csv
# python benchmark.py results/solutions/Net2_leak_sa_k4.json plot     -> also saves the delay map as a png
if __name__ == "__main__":
    plot = "plot" in sys.argv
    paths = [p for p in sys.argv[1:] if p != "plot"]

    rows = []
    for solution_path in paths:
        metrics = benchmark(solution_path, plot)
        rows.append(metrics)
        if len(paths) == 1:
            for key, value in metrics.items():
                print(f"{key:22} {value}")

    # one row per solution in a shared table, so runs are easy to compare
    os.makedirs(BENCHMARKS, exist_ok=True)
    summary_path = f"{BENCHMARKS}/summary.csv"
    summary = pd.DataFrame(rows)
    if os.path.exists(summary_path):
        old = pd.read_csv(summary_path)
        summary = pd.concat([old[~old["solution"].isin(summary["solution"])], summary])
    summary.sort_values("solution").to_csv(summary_path, index=False)
    print(f"saved {len(rows)} benchmarks in {summary_path}")