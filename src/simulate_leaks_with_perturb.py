import sys

import numpy as np   # >>> ADDED (noise): needed for the random numbers in perturb()
import pandas as pd
import wntr


# leak settings
# time is in seconds
duration = 6 * 3600
leak_start = 3600
timestep = 60
threshold = 0.5          # pressure drop in meters that counts as an alarm
leak_area = 0.0005       # m^2
discharge_coeff = 0.75


# =====================================================================================
# >>> ADDED (noise) - START
# Small, realistic everyday noise for the network. It is OFF by default, so
# `python src/simulate_leaks.py Net1` still makes exactly the same results/leaks/Net1.csv.
# Turn it on with:  python src/simulate_leaks.py Net1 noisy   (saves results/leaks/Net1_noisy.csv)
# =====================================================================================
noisy = False            # on/off switch for the noise
noise_seed = 0           # which random pattern; same seed = same noise every run


def perturb(wn):
    """add small, realistic everyday noise to the network. each block is one kind of noise;
    put # in front of a block to turn that kind off."""
    rng = np.random.default_rng(noise_seed)

    # 1. random demand: every customer uses 90-110% of their normal water
    for j in wn.junction_name_list:
        for d in wn.get_node(j).demand_timeseries_list:
            d.base_value *= rng.uniform(0.9, 1.1)

    # 2. pipe aging: every pipe gets 0-15% rougher (lower roughness C = more friction)
    for p in wn.pipe_name_list:
        wn.get_link(p).roughness *= rng.uniform(0.85, 1.0)

    # 3. supply pressure: every reservoir's water level shifts by -2 to +2 m
    for r in wn.reservoir_name_list:
        wn.get_node(r).base_head += rng.uniform(-2, 2)
# >>> ADDED (noise) - END
# =====================================================================================


def simulate(network, leak_node=None):
    """run the network and return the pressure at every junction over time."""
    wn = wntr.network.WaterNetworkModel(f"networks/{network}.inp")
    wn.options.time.duration = duration
    wn.options.time.hydraulic_timestep = timestep
    wn.options.time.report_timestep = timestep
    wn.options.hydraulic.demand_model = "PDD"
    if noisy:            # >>> ADDED (noise): change the network before simulating,
        perturb(wn)      # >>> ADDED (noise): only when the noise is turned on
    if leak_node:
        wn.get_node(leak_node).add_leak(wn, area=leak_area, discharge_coeff=discharge_coeff, start_time=leak_start)
    pressure = wntr.sim.WNTRSimulator(wn).run_sim().node["pressure"]
    return pressure[wn.junction_name_list]


def build_leak_table(network):
    """
    simulate a leak at every junction, one at a time.

    returns a table where each row is a leak location, each column is a junction,
    and each value is how many minutes a sensor there needs to notice the leak
    (empty if it never does).
    """
    normal = simulate(network)
    rows = {}
    for leak_node in normal.columns:
        drop = normal - simulate(network, leak_node)
        alarm = drop[drop.index >= leak_start] >= threshold
        first_alarm = alarm.idxmax().where(alarm.any())
        rows[leak_node] = (first_alarm - leak_start) / 60
    table = pd.DataFrame(rows).T
    table.index.name = "leak_node"
    return table


# might move it to src/utils.py later
def load_leak_table(network):
    """read the saved leak table for a network."""
    return pd.read_csv(f"results/leaks/{network}.csv", index_col="leak_node", dtype={"leak_node": str})


if __name__ == "__main__":
    network = sys.argv[1]
    # >>> CHANGED (noise) - START
    # before:
    #     table = build_leak_table(network)
    #     table.to_csv(f"results/leaks/{network}.csv")
    #     print(f"saved {len(table)} leak scenarios to results/leaks/{network}.csv")
    # now: an optional second word "noisy" turns the noise on and saves to a separate
    # file, so the normal table (which benchmark.py reads) is never overwritten.
    noisy = len(sys.argv) > 2 and sys.argv[2] == "noisy"
    name = f"{network}_noisy" if noisy else network
    table = build_leak_table(network)
    table.to_csv(f"results/leaks/{name}.csv")
    print(f"saved {len(table)} leak scenarios to results/leaks/{name}.csv")
    # >>> CHANGED (noise) - END
