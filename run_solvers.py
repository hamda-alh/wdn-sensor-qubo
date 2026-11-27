import sys

from src.qubo import build_Q_topological, build_Q_leak, pick_penalty, cutoff, energy, to_sensors
from src.solvers import solve_gurobi, solve_sa
from src.utils import save_solution


# qubo settings, the same for every solver
p_cost = 0.5

# simulated annealing settings
num_reads = 100
num_sweeps = 1000
seed = 0


def build_Q(network, formulation, k, penalty):
    if formulation == "topological":
        return build_Q_topological(network, k, penalty, p_cost)
    return build_Q_leak(network, k, penalty)


def run(network, formulation, k):
    """run gurobi and simulated annealing on one problem and save both solutions."""
    # the qubo without the penalty, used by gurobi and to pick the penalty
    Q0, const0, nodes = build_Q(network, formulation, k, penalty=0)
    penalty = pick_penalty(Q0)
    Q, const, _ = build_Q(network, formulation, k, penalty)

    # saved with every solution, so we know later what the run was based on
    info = {"n_junctions": len(nodes), "p_cost": p_cost}
    if formulation == "leak":
        info = {"n_junctions": len(nodes), "cutoff_min": cutoff}

    # gurobi gets the budget as a real constraint, so it uses the qubo without the penalty
    x, e, runtime, _ = solve_gurobi(Q0, const0, k=k)
    save_solution(network, "gurobi", formulation, k, to_sensors(x, nodes), e, runtime,
                  params={"budget": "constraint", **info})
    print(f"{network} {formulation} k={k}  gurobi  energy={e:.4f}  time={runtime:.3f}s")

    x, e, runtime, samples = solve_sa(Q, const, num_reads, num_sweeps, seed)
    save_solution(network, "sa", formulation, k, to_sensors(x, nodes), e, runtime,
                  params={"penalty": penalty, "num_reads": num_reads, "num_sweeps": num_sweeps,
                          "seed": seed, **info},
                  samples=[to_sensors(s, nodes) for s in samples],
                  sample_energies=[float(energy(Q, const, s)) for s in samples])
    print(f"{network} {formulation} k={k}  sa      energy={e:.4f}  time={runtime:.3f}s")


# python run_solvers.py Net1 topological 3
# python run_solvers.py Net2 leak 1 2 3 4 5        (several k at once)
if __name__ == "__main__":
    network, formulation = sys.argv[1], sys.argv[2]
    for k in sys.argv[3:]:
        run(network, formulation, int(k))