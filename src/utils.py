import json
from datetime import datetime
from pathlib import Path

# to keep all possible solutions we would get in one place
SOLUTIONS = Path(__file__).parent.parent / "results" / "solutions"


# save solution as a json format

def save_solution(network, solver, formulation, k, sensors, energy, runtime_s, params=None, samples=None):


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
        "time": time,
    }
    path = SOLUTIONS / f"{network}_{solver}_k{k}_{time}.json"
    with open(path, "w") as f:
        json.dump(solution, f, indent=2)
    return path


def load_solution(path):
    """read a saved solution back as a dictionary."""
    with open(path) as f:
        return json.load(f)