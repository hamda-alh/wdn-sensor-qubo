# qubo metrics

def is_feasible(sensors, k):
    """true if the placement uses exactly k different junctions."""
    return len(set(sensors)) == k


def feasibility_rate(samples, k):
    """fraction of the solver's samples that use exactly k sensors."""
    if not samples:
        return None
    return sum(is_feasible(s, k) for s in samples) / len(samples)


def energy_gap(energy, best_energy):
    """how far a solver's energy is above the exact optimum (0 = it found the optimum)."""
    return energy - best_energy


def success_rate(sample_energies, best_energy):
    """fraction of the solver's samples that reached the exact optimum."""
    if not sample_energies:
        return None
    return sum(abs(e - best_energy) < 1e-6 for e in sample_energies) / len(sample_energies)


# leak metrics

def leak_delays(table, sensors):
    """minutes until the first of the sensors notices each leak (empty if none does)."""
    return table[sensors].min(axis=1)


def detection_rate(delays, cutoff=30):
    """percent of leaks that a sensor notices within the cutoff (in minutes)."""
    return 100 * (delays <= cutoff).mean()


def mean_delay(delays):
    """average minutes to notice a leak, counting only the leaks that are noticed at some point."""
    return delays.mean()