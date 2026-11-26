# Realistic noise in `src/simulate_leaks.py`

`src/simulate_leaks.py` can add small, realistic everyday noise to the network before simulating leaks. This shows
whether a sensor placement still works on a normal, slightly messy day.

The noise is **off by default**. Without it, the script makes exactly the same `results/leaks/{network}.csv` as before,
and `benchmark.py` is not affected.

## What the noise does

The `perturb(wn)` function changes the network before every simulation:

| # | Noise | What changes |
|---|---|---|
| 1 | Random demand | Every customer uses 90–110% of their normal water |
| 2 | Pipe aging | Every pipe gets 0–15% rougher (roughness C × 0.85–1.0), so more friction |
| 3 | Supply pressure | Every reservoir's water level shifts by −2 to +2 m |

- **The same seed gives the same noise every time.** The normal run and every leak run share the same noise, so the
  only difference between them is the leak.
- **To turn one kind of noise off**, put `#` in front of its block inside `perturb()`.
- **To try a different random pattern**, change `noise_seed = 0` at the top of the file to 1, 2, 3, and so on.

Every change to Hamda's original file is marked in the code with `>>> ADDED (noise)` or `>>> CHANGED (noise)`.

## Commands

Run from the repo root, the folder that contains `networks/` and `results/`:

```
python src/simulate_leaks.py Net1          # normal     -> results/leaks/Net1.csv        (same as before)
python src/simulate_leaks.py Net1 noisy    # with noise -> results/leaks/Net1_noisy.csv
python src/simulate_leaks.py Net2 noisy    #            -> results/leaks/Net2_noisy.csv
```

Net1 takes about 7 seconds and Net2 about 15.

## Running in Google Colab

**Cell 1: get the repo and the tools**
```python
!pip install -q wntr
!git clone -q https://github.com/hamda-alh/wdn-sensor-qubo.git
%cd wdn-sensor-qubo
```

**Cell 2: swap in the new file.** Skip this cell once the new file is pushed to the repo.
```python
from google.colab import files
files.upload()                                   # choose simulate_leaks.py
!mv simulate_leaks.py src/simulate_leaks.py
```

**Cell 3: build the noisy leak tables**
```python
!python src/simulate_leaks.py Net1 noisy
!python src/simulate_leaks.py Net2 noisy
```

**Cell 4: compare sensor placements on the normal and noisy network**
```python
import pandas as pd
from src.metrics import leak_delays, detection_rate, mean_delay

placements = {
    'Net1': {"paper's QUBO": ['11','12','22'],         'leak QUBO': ['10','11','23']},
    'Net2': {"paper's QUBO": ['7','11','13','15','23'], 'leak QUBO': ['11','16','22','23','28']},
}

rows = []
for net, pl in placements.items():
    for version in ['normal', 'noisy']:
        file = f'results/leaks/{net}.csv' if version == 'normal' else f'results/leaks/{net}_noisy.csv'
        table = pd.read_csv(file, index_col='leak_node', dtype={'leak_node': str})
        for qubo, sensors in pl.items():
            d = leak_delays(table, sensors)
            rows.append({'network': net, 'QUBO': qubo, 'version': version,
                         'detected %': round(detection_rate(d)),
                         'avg delay (min)': round(mean_delay(d), 1),
                         'caught within 30 min %': round((d <= 30).mean() * 100)})
pd.DataFrame(rows)
```

**Optional: download the noisy tables**
```python
files.download('results/leaks/Net1_noisy.csv')
files.download('results/leaks/Net2_noisy.csv')
```

## Running on your own computer

1. Put the new `simulate_leaks.py` in `src/`, replacing the old one.
2. Open a terminal in the repo root.
3. Run `python src/simulate_leaks.py Net1 noisy` and the same for `Net2`.
4. Run the Cell 4 code in Python from the same folder.

## Results (seed 0)

% of leaks caught within 30 minutes:

| Network | Paper's QUBO (normal → noisy) | Leak QUBO (normal → noisy) |
|---|---|---|
| Net1, 3 sensors | 78% → 89% | 89% → 89% |
| Net2, 5 sensors | 31% → 31% | 66% → 71% |

The leak QUBO still wins under realistic noise. Some numbers go up slightly because the noise shifts pressures, so a
few leaks cross the 0.5 m alarm line sooner. It is a slightly different network, not a bug. For a stronger claim,
repeat with about 5 seeds and report the average and spread.

## Notes

- **Each noisy run overwrites `{network}_noisy.csv`.** Rename it first if you want to keep results from several seeds.
- **`benchmark.py` always reads the normal table** (`load_leak_table` reads `results/leaks/{network}.csv`). Never save
  noisy results over the normal file, or every benchmark result will silently change.
