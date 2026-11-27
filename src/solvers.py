import itertools
import time

import dimod
import gurobipy as gp
import neal
import numpy as np


# every solver takes the qubo (Q, const) and returns
#   x        best placement found, as a list of 0s and 1s
#   energy   energy of x
#   runtime  seconds
#   samples  every placement the solver returned (only sa returns more than one)


def solve_gurobi(Q, const, k=None, time_limit=60):
    """
    exact solver. this is our reference for the best possible energy.

    if k is given, the budget is a real constraint (exactly k sensors) and Q should be
    built with penalty=0. gurobi is much faster this way than with the penalty inside Q.
    """
    model = gp.Model()
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit
    x = model.addMVar(len(Q), vtype=gp.GRB.BINARY)
    model.setObjective(x @ Q @ x + const, gp.GRB.MINIMIZE)
    if k is not None:
        model.addConstr(x.sum() == k)

    start = time.perf_counter()
    model.optimize()
    runtime = time.perf_counter() - start

    if model.Status != gp.GRB.OPTIMAL:
        print("gurobi hit the time limit, the answer may not be the optimum")
    best = [int(round(v)) for v in x.X]
    return best, model.ObjVal, runtime, [best]


def solve_sa(Q, const, num_reads=100, num_sweeps=1000, seed=0):
    """
    simulated annealing.
    num_reads = how many separate runs, num_sweeps = how many steps in each run.
    """
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q, offset=const)
    sampler = neal.SimulatedAnnealingSampler()

    start = time.perf_counter()
    result = sampler.sample(bqm, num_reads=num_reads, num_sweeps=num_sweeps, seed=seed)
    runtime = time.perf_counter() - start

    # one placement per run
    samples = [[int(s[i]) for i in range(len(Q))] for s in result.samples()]
    best = [int(result.first.sample[i]) for i in range(len(Q))]
    return best, result.first.energy, runtime, samples


def solve_brute_force(Q, const):
    """try every possible placement. only for tiny networks like Net1."""
    N = len(Q)
    if N > 20:
        raise ValueError("too many junctions for brute force")

    start = time.perf_counter()
    X = np.array(list(itertools.product([0, 1], repeat=N)))
    energies = np.sum((X @ Q) * X, axis=1) + const
    runtime = time.perf_counter() - start

    best = X[energies.argmin()].tolist()
    return best, energies.min(), runtime, [best]
