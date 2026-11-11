"""Run leak simulations and compare sensor placements."""

import math
from dataclasses import dataclass

import pandas as pd
import wntr


def build_benchmark(
    network_path,
    leak_nodes=None,
    *,
    duration_s=21600,
    leak_start_s=3600,
    timestep_s=60,
    threshold_m=0.5,
    leak_area_m2=0.0005,
    discharge_coeff=0.75,
):
    """Return a benchmark with one scenario per leak junction.

    Times are in seconds. The default suite places a sustained leak at each
    junction in turn. A smaller list is useful for a quick run.
    """
    if timestep_s <= 0 or not 0 <= leak_start_s <= duration_s:
        raise ValueError("Use a positive timestep and a leak onset within the run.")
    if duration_s % timestep_s or leak_start_s % timestep_s:
        raise ValueError("Duration and leak onset must align with the timestep.")
    for value in (threshold_m, leak_area_m2, discharge_coeff):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Threshold, leak area, and discharge coefficient must be positive.")

    def simulate(leak_node=None):
        network = wntr.network.WaterNetworkModel(str(network_path))
        network.options.time.duration = duration_s
        network.options.time.hydraulic_timestep = timestep_s
        network.options.time.report_timestep = timestep_s
        network.options.time.report_start = 0
        network.options.hydraulic.demand_model = "PDD"
        if leak_node is not None:
            network.get_node(leak_node).add_leak(
                network, area=leak_area_m2,
                discharge_coeff=discharge_coeff, start_time=leak_start_s,
            )
        result = wntr.sim.WNTRSimulator(network).run_sim(convergence_error=True)
        pressure = result.node["pressure"].loc[:, junctions]
        expected_times = pd.Index(range(0, duration_s + 1, timestep_s))
        pressure = pressure.reindex(expected_times)
        if not pressure.abs().lt(math.inf).all().all():
            raise RuntimeError("Simulation returned missing or nonfinite pressure samples.")
        return pressure

    junctions = wntr.network.WaterNetworkModel(str(network_path)).junction_name_list
    scenarios = list(junctions if leak_nodes is None else leak_nodes)
    if not scenarios or len(set(scenarios)) != len(scenarios):
        raise ValueError("Provide at least one leak junction and no duplicates.")
    if unknown := set(scenarios) - set(junctions):
        raise ValueError(f"Unknown leak junctions: {sorted(unknown)}")

    baseline = simulate()
    alarm_times = pd.DataFrame(index=pd.Index(scenarios, name="leak_node"),
                               columns=junctions, dtype=float)
    failures = {}
    for node in scenarios:
        try:
            pressure = simulate(node)
        except RuntimeError as error:
            failures[node] = str(error)
            continue
        alarms = (baseline - pressure).loc[leak_start_s:] >= threshold_m
        detected = alarms.columns[alarms.any()]
        alarm_times.loc[node, detected] = alarms[detected].idxmax()

    return LeakBenchmark(alarm_times, leak_start_s, failures)


@dataclass
class LeakBenchmark:
    first_alarm_s: pd.DataFrame
    leak_start_s: int
    failures: dict[str, str]

    def evaluate(self, selected_nodes, *, expected_k=None):
        """Score a placement given its selected junction IDs."""
        sensors = list(selected_nodes)
        if len(set(sensors)) != len(sensors):
            raise ValueError("Selected junctions must be unique.")
        if expected_k is not None and len(sensors) != expected_k:
            raise ValueError(f"Expected {expected_k} sensors, got {len(sensors)}.")
        if unknown := set(sensors) - set(self.first_alarm_s.columns):
            raise ValueError(f"Unknown sensor junctions: {sorted(unknown)}")

        alarms = self.first_alarm_s.drop(index=list(self.failures))[sorted(sensors)]
        first_alarm = alarms.min(axis=1)
        detected = first_alarm.notna()
        details = pd.DataFrame({
            "status": "undetected",
            "first_alarm_s": first_alarm,
            "delay_min": (first_alarm - self.leak_start_s) / 60,
            "triggering_junction": pd.Series(None, index=alarms.index, dtype=object),
        })
        details.loc[detected, "status"] = "detected"
        if detected.any():
            details.loc[detected, "triggering_junction"] = alarms.loc[detected].idxmin(axis=1)
        details = details.reindex(self.first_alarm_s.index)
        details.loc[details.index.isin(self.failures), "status"] = "simulation_failed"

        detected_count = int(detected.sum())
        return {
            "coverage_pct": 100 * detected_count / len(alarms) if len(alarms) else None,
            "mean_delay_min": float(details["delay_min"].mean()) if detected_count else None,
            "detected_count": detected_count,
            "undetected_count": len(alarms) - detected_count,
            "simulation_failures": self.failures.copy(),
            "per_scenario": details,
        }
