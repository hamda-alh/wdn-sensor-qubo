# Quantum Optimization Approaches to Sensor Placement in Water Distribution Networks

Course project for (18-619), Carnegie Mellon University

- **Problem:** a water network can only afford a few pressure sensors. Where should they go so that leaks are noticed quickly?
- **Approach:** we write the problem as a QUBO in two ways and compare them.
  - **Topological QUBO** Previous study (Speziali et al.): cover the most important pipes using only the network topology.
  - **Leak QUBO** (ours): simulate a leak at every junction and place sensors where they notice the most leaks.
- **Solvers:** Gurobi (exact), simulated annealing (SA) and QAOA (Qiskit simulator).
- **Networks:** the EPANET examples Net1 (9 junctions), Net2 (35) and Net3 (92).

## Project Structure

```
wdn-sensor-qubo/
├── run_solvers.py          runs the solvers on one network and QUBO, saves the solutions
├── benchmark.py            scores saved solutions
├── requirements.txt
├── networks/               Net1.inp, Net2.inp, Net3.inp
├── src/
│   ├── simulate_leaks.py   simulates a leak at every junction
│   ├── qubo.py             builds both QUBOs and picks the penalty
│   ├── solvers.py          Gurobi, SA, QAOA, brute force
│   ├── metrics.py          (energy gap, leak detection, ..)
│   └── utils.py            save + load solutions + draw a network
├── notebooks/              Full Analysis
└── results/
    ├── leaks/                    leak tables: one clean and five noisy per network
    ├── solutions/                one json file per solver run
    ├── qaoa_before_rescaling/    first QAOA results, kept for comparison
    └── figures/                  figures made by the notebooks
```

## Setup

```
pip install -r requirements.txt
```

Gurobi works with the free licence that comes with `gurobipy`. QAOA needs `qiskit`, `qiskit-algorithms` and `qiskit-optimization`; the other solvers run without them.

## How to run

Run everything from the repo root. The results are already in the repo, so steps 1 and 2 are only needed to reproduce them.

**1. Simulate the leaks** (Net3 takes about 5 minutes per table)

```
python src/simulate_leaks.py Net1            # clean network
python src/simulate_leaks.py Net1 noisy      # noisy network 0
python src/simulate_leaks.py Net1 noisy 3    # noisy network 3
```

**2. Run the solvers.** The numbers are the sensor budgets `k`, one run each.

```
python run_solvers.py Net2 leak 2 4 6 8 10
python run_solvers.py Net1 topological 1 2 3 4 5 qaoa    # also run QAOA (Net1 only, about 1 minute per k)
```

**3. Score the solutions**

```
python benchmark.py results/solutions/*.json                        # one table: results/benchmarks/summary.csv
python benchmark.py results/solutions/Net2_leak_sa_k4.json plot     # also saves a delay map
```

**4. Open the notebooks** for the analysis and figures.

## Notebooks

| Notebook | Question |
|---|---|
| `penalty_sweep.ipynb` | How big must the penalty be so that solutions use exactly k sensors? |
| `scalability.ipynb` | Do SA and QAOA find the best energy, and how does that change when the network gets bigger? |
| `leak_detection.ipynb` | Which QUBO gives placements that detect more leaks? Does a low energy mean good detection? |
| `energy_and_delay_figs.ipynb` | One figure per QUBO: energy of every solver, and a map of how fast leaks are detected. |

`baseline.ipynb` is an early version kept for reference. It does not use the code in `src/`.

## Parameters

All fixed values and where to change them.

### Leak simulation (top of `src/simulate_leaks.py`)

| Parameter | Value | Meaning |
|---|---|---|
| `duration` | 24 hours | length of each simulation |
| `leak_start` | 1 hour | when the leak opens |
| `timestep` | 60 seconds | how often pressure is computed and recorded |
| `threshold` | 0.5 m | pressure drop at a junction that counts as an alarm |
| `leak_areas` | 0.0005 m² (Net1, Net2), 0.002 m² (Net3) | size of the hole. Net3 is much bigger, so the same hole barely changes its pressure |
| `discharge_coeff` | 0.75 | how easily water flows out of the hole |
| demand model | `PDD` (set in `simulate`) | demand depends on pressure, so a leak lowers the pressure realistically |

### Noisy networks (`add_noise` in `src/simulate_leaks.py`)

Used only to score placements. The QUBOs are always built from the clean network.

| Change | Range |
|---|---|
| demand at every junction | 90% to 110% |
| roughness of every pipe | 85% to 100% (rougher pipes) |
| reservoir level | up to 2 m higher or lower |
| noise seed | the number after `noisy` on the command line. We use 0 to 4 |

### QUBO (`src/qubo.py`)

| Parameter | Value | Where | Meaning |
|---|---|---|---|
| `cutoff` | 30 minutes | top of the file | a sensor detects a leak only if it notices it this fast |
| pipe weight | edge betweenness, scaled to 0-1 | `edge_weights` | how important a pipe is (topological QUBO) |
| demand | scaled to 0-1, negative demand counts as 0 | `junction_demand` | placement cost is `1 - demand` (topological QUBO); leak weight is `demand` (leak QUBO) |
| penalty | 10% above the most one sensor can improve the objective | `pick_penalty` | forces exactly k sensors. Computed for each network and QUBO |
| `p_cost` | 0.5 | top of `run_solvers.py` | how much the placement cost counts (topological QUBO only) |

### Solvers (top of `run_solvers.py`, functions in `src/solvers.py`)

| Solver | Parameter | Value | Meaning |
|---|---|---|---|
| Gurobi | budget | a real constraint, no penalty | exact reference |
| Gurobi | `time_limit` | 60 seconds | never reached on these networks |
| SA | `num_reads` | 100 | separate runs |
| SA | `num_sweeps` | 1000 | steps in each run |
| SA | `seed` | 0 | |
| QAOA | `qaoa_reps` | 1 | number of layers |
| QAOA | `qaoa_maxiter` | 30 | steps of the COBYLA optimizer |
| QAOA | `qaoa_shots` | 512 | measurements |
| QAOA | `max_qaoa_qubits` | 20 | QAOA is skipped on larger networks (one qubit per junction) |
| QAOA | rescaling | on | Q is divided by its largest number before it is given to Qiskit (in `solve_qaoa`) |
| QAOA | seed | 42 | set inside `solve_qaoa` |

### Sensor budgets

Given on the command line. We used Net1: 1 to 5, Net2: 2, 4, 6, 8, 10, Net3: 2, 5, 10, 15, 20.



## Team and reference

Hamda Alhosani, Rania Alshawabkeh, Alexa Noto

The topological QUBO follows S. Speziali et al., "Solving Sensor Placement Problems In Real Water Distribution Networks Using Adiabatic Quantum Computation", arXiv:2108.04075 (2021).

## TODO

Right now most parameters are fixed inside the scripts. The plan is to make the params available on command.

| Script | Parameters to expose | Example |
|---|---|---|
| `run_solvers.py` | which solvers to run | `--solvers gurobi sa qaoa` |
| `run_solvers.py` | SA: number of runs, steps and seed | `--num-reads 100 --num-sweeps 1000 --seed 0` |
| `run_solvers.py` | QAOA: layers, optimizer steps, measurements | `--qaoa-reps 1 --qaoa-maxiter 30 --qaoa-shots 512` |
| `run_solvers.py` | penalty: a fixed value in place of the rule | `--penalty 5` |
| `run_solvers.py` | weight of the placement cost | `--p-cost 0.5` |
| `run_solvers.py` and `benchmark.py` | detection cutoff in minutes | `--cutoff 30` |
| `src/simulate_leaks.py` | leak size and alarm threshold | `--leak-area 0.002 --threshold 0.5` |
| `src/simulate_leaks.py` | simulation length | `--hours 24` |
| `benchmark.py` | which noisy network to score on | `--noise-seed 3` |

Values that rarely change (timestep, leak start, discharge coefficient, noise ranges) would stay in the scripts.

One thing to keep in mind when this is added: the cutoff must be the same when building the leak QUBO and when scoring, and all solvers must use the same penalty and `p_cost`, otherwise their energies cannot be compared.