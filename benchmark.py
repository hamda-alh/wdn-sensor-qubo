import json
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


def gurobi_energy(network, formulation, k):
    """energy of the exact solution for the same problem, if we have it saved."""
    path = f"{SOLUTIONS}/{network}_{formulation}_gurobi_k{k}.json"
    if not os.path.exists(path):
        return None
    return load_solution(path)["energy"]


def benchmark(solution_path):
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

    # one folder per solution
    folder = f"{BENCHMARKS}/{name}"
    os.makedirs(folder, exist_ok=True)
    with open(f"{folder}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    delays.rename("delay_min").to_csv(f"{folder}/delays.csv")
    plot_delays(network, sensors, delays, f"{folder}/delays.png")

    return metrics


# python benchmark.py results/solutions/Net1_leak_sa_k3.json
# python benchmark.py results/solutions/*.json          (everything at once)
if __name__ == "__main__":
    rows = []
    for solution_path in sys.argv[1:]:
        metrics = benchmark(solution_path)
        rows.append(metrics)
        for key, value in metrics.items():
            print(f"{key:22} {value}")
        print()

    # one row per solution in a shared table, so runs are easy to compare
    summary_path = f"{BENCHMARKS}/summary.csv"
    summary = pd.DataFrame(rows)
    if os.path.exists(summary_path):
        old = pd.read_csv(summary_path)
        summary = pd.concat([old[~old["solution"].isin(summary["solution"])], summary])
    summary.sort_values("solution").to_csv(summary_path, index=False)
    print(f"saved {len(rows)} benchmarks, summary in {summary_path}")