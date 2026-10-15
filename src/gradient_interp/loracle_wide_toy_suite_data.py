"""Fixed wider checkpoint gap with a within-trajectory 20-update comparator."""

from copy import deepcopy

from .loracle_toy_suite_data import build_data as build_original_data

PHASE_STEPS = (20, 100)
SNAPSHOTS = ("1", "short", "2")
INTERVALS = (
    ("first", 0, 1),
    ("short", 1, 2),
    ("adjacent", 1, 3),
    ("cumulative", 0, 3),
)


def build_data():
    data = build_original_data()
    for t in data["trajectories"]:
        t["row_order"][t["tasks"][1]] *= 5
    # Copy the wide interval's exact option order for its matched short comparator.
    # Append cases so every inherited case index and shuffled-control seed is unchanged.
    short = []
    for case in data["cases"]:
        if case["window"] == "adjacent":
            value = deepcopy(case)
            value.update(id=case["trajectory"] + "-short", window="short")
            short.append(value)
    data["cases"].extend(short)
    data["checkpoint_steps"] = {"initial": 0, "1": 20, "short": 40, "2": 120}
    data["phase_steps"] = list(PHASE_STEPS)
    return data


def validate_data(data):
    if data != build_data():
        raise ValueError("Frozen wider-gap data or paired comparison changed")


def history_position(trajectory_index, phase, step):
    if phase not in (1, 2) or not 1 <= step <= PHASE_STEPS[phase - 1]:
        raise ValueError("Invalid training step")
    return trajectory_index * sum(PHASE_STEPS) + sum(PHASE_STEPS[: phase - 1]) + step
