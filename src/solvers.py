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
#   samples  every placement the solver returned (one per run for sa, one per shot for qaoa)


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


def solve_qaoa(Q, const, reps=1, maxiter=30, shots=512, seed=42):
    """
    qaoa on a simulator (qiskit). needs one qubit per junction, so only for small networks.
    reps = number of qaoa layers, maxiter = optimizer steps, shots = how many times the circuit is measured.
    samples has one placement per shot.
    """
    # imported here so the other solvers still work without qiskit installed
    from qiskit.primitives import StatevectorSampler
    from qiskit_algorithms import QAOA
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_algorithms.utils import algorithm_globals
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.algorithms import MinimumEigenOptimizer

    N = len(Q)
    if N > 20:
        raise ValueError("too many junctions to simulate qaoa")

    # qaoa works better when the numbers in Q are small, so we divide by the biggest one.
    # this does not change which placement is best
    scale = np.abs(Q).max()

    # write the qubo in the form qiskit wants
    problem = QuadraticProgram()
    for i in range(N):
        problem.binary_var(name=f"x_{i}")
    linear = {f"x_{i}": Q[i, i] / scale for i in range(N)}
    pairs = {(f"x_{i}", f"x_{j}"): Q[i, j] / scale for i in range(N) for j in range(i + 1, N) if Q[i, j] != 0}
    problem.minimize(constant=const / scale, linear=linear, quadratic=pairs)

    algorithm_globals.random_seed = seed       # same starting angles every time
    sampler = StatevectorSampler(default_shots=shots, seed=seed)
    qaoa = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=maxiter), reps=reps)

    start = time.perf_counter()
    result = MinimumEigenOptimizer(qaoa).solve(problem)
    runtime = time.perf_counter() - start

    # every measured bitstring and its share of the shots. qiskit writes qubit 0 last, so we reverse it
    measured = result.min_eigen_solver_result.eigenstate
    samples = []
    for bits, share in measured.items():
        x = [int(b) for b in reversed(bits)]
        samples += [x] * round(share * shots)

    # energies with the original Q, so they compare with the other solvers
    energies = [np.array(x) @ Q @ np.array(x) + const for x in samples]
    best = samples[int(np.argmin(energies))]
    return best, min(energies), runtime, samples


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


def solve_max_detection(S, k):
    """
    the most leaks that k sensors can detect. S[i, j] = 1 if a sensor at junction j detects leak i.
    this is not a qubo. it is the true best answer, used as a reference line.
    """
    n_leaks, n_junctions = S.shape
    model = gp.Model()
    model.Params.OutputFlag = 0
    x = model.addMVar(n_junctions, vtype=gp.GRB.BINARY)     # 1 if junction j has a sensor
    y = model.addMVar(n_leaks, ub=1)                        # 1 if leak i is detected
    model.addConstr(y <= S @ x)                             # a leak needs at least one sensor that sees it
    model.addConstr(x.sum() == k)
    model.setObjective(y.sum(), gp.GRB.MAXIMIZE)
    model.optimize()

    best = [int(round(v)) for v in x.X]
    return best, int(round(model.ObjVal))