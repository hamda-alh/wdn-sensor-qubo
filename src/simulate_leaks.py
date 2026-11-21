import sys

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


def simulate(network, leak_node=None):
    """run the network and return the pressure at every junction over time."""
    wn = wntr.network.WaterNetworkModel(f"networks/{network}.inp")
    wn.options.time.duration = duration
    wn.options.time.hydraulic_timestep = timestep
    wn.options.time.report_timestep = timestep
    wn.options.hydraulic.demand_model = "PDD"
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
    table = build_leak_table(network)
    table.to_csv(f"results/leaks/{network}.csv")
    print(f"saved {len(table)} leak scenarios to results/leaks/{network}.csv")


    



