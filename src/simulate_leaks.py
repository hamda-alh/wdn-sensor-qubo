import sys

import numpy as np
import pandas as pd
import wntr


# leak settings (time is in seconds)
duration = 24 * 3600
leak_start = 3600
timestep = 60
threshold = 0.5          # pressure drop in meters that counts as an alarm
discharge_coeff = 0.75

# size of the hole in m^2. Net3 is much bigger, so the same hole barely changes its pressure
leak_areas = {"Net1": 0.0005, "Net2": 0.0005, "Net3": 0.002}


def add_noise(wn, seed):
    """
    change the network a little, so it is not exactly the one the qubo was built from.
    each seed gives a different noisy network.
    """
    rng = np.random.default_rng(seed)

    # customers use 90-110% of their normal water
    for j in wn.junction_name_list:
        for d in wn.get_node(j).demand_timeseries_list:
            d.base_value *= rng.uniform(0.9, 1.1)

    # pipes get up to 15% rougher (older pipes)
    for p in wn.pipe_name_list:
        wn.get_link(p).roughness *= rng.uniform(0.85, 1.0)

    # reservoir levels move by up to 2 m
    for r in wn.reservoir_name_list:
        wn.get_node(r).base_head += rng.uniform(-2, 2)


def simulate(network, leak_node=None, noisy=False, seed=0):
    """run the network and return the pressure at every junction over time."""
    wn = wntr.network.WaterNetworkModel(f"networks/{network}.inp")
    wn.options.time.duration = duration
    wn.options.time.hydraulic_timestep = timestep
    wn.options.time.report_timestep = timestep
    wn.options.hydraulic.demand_model = "PDD"
    if noisy:
        add_noise(wn, seed)
    if leak_node:
        wn.get_node(leak_node).add_leak(wn, area=leak_areas[network], discharge_coeff=discharge_coeff, start_time=leak_start)
    pressure = wntr.sim.WNTRSimulator(wn).run_sim().node["pressure"]
    return pressure[wn.junction_name_list]


def build_leak_table(network, noisy=False, seed=0):
    """
    simulate a leak at every junction, one at a time.

    returns a table where each row is a leak location, each column is a junction,
    and each value is how many minutes a sensor there needs to notice the leak
    (empty if it never does).
    """
    # the no-leak run and the leak runs get the same noise, so the only difference is the leak
    normal = simulate(network, noisy=noisy, seed=seed)
    rows = {}
    for leak_node in normal.columns:
        drop = normal - simulate(network, leak_node, noisy, seed)
        alarm = drop[drop.index >= leak_start] >= threshold
        first_alarm = alarm.idxmax().where(alarm.any())
        rows[leak_node] = (first_alarm - leak_start) / 60
    table = pd.DataFrame(rows).T
    table.index.name = "leak_node"
    return table


def leak_table_path(network, noisy=False, seed=0):
    name = network
    if noisy:
        # seed 0 keeps the old name, the others get their number
        name = f"{network}_noisy" if seed == 0 else f"{network}_noisy{seed}"
    return f"results/leaks/{name}.csv"


def load_leak_table(network, noisy=False, seed=0):
    """read the saved leak table for a network."""
    return pd.read_csv(leak_table_path(network, noisy, seed), index_col="leak_node", dtype={"leak_node": str})


# python src/simulate_leaks.py Net1            -> results/leaks/Net1.csv
# python src/simulate_leaks.py Net1 noisy      -> results/leaks/Net1_noisy.csv
# python src/simulate_leaks.py Net1 noisy 3    -> results/leaks/Net1_noisy3.csv  (another noisy network)
if __name__ == "__main__":
    network = sys.argv[1]
    noisy = "noisy" in sys.argv
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    table = build_leak_table(network, noisy, seed)
    path = leak_table_path(network, noisy, seed)
    table.to_csv(path)
    print(f"saved {len(table)} leak scenarios to {path}")