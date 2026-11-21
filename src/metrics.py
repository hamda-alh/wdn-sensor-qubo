
# qubo metrics
def is_feasible(sensors, k):
    """true if the placement uses exactly k different junctions."""
    return len(set(sensors)) == k


def feasibility_rate(samples, k):
    """fraction of the solver's samples that use exactly k sensors."""
    if not samples:
        return None
    return sum(is_feasible(s, k) for s in samples) / len(samples)

# leak metrics
def leak_delays(table, sensors):
    """minutes until the first of the sensors notices each leak (empty if none does)."""
    return table[sensors].min(axis=1)


def detection_rate(delays):
    """percent of leaks that at least one sensor notices."""
    return 100 * delays.notna().mean()


def mean_delay(delays):
    """average minutes to notice a leak, counting only the detected ones."""
    return delays.mean()