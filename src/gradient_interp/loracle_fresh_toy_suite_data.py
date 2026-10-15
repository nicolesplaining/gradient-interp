"""One fresh-adapter comparison using the exact completed task-A starting states."""

from .loracle_wide_toy_suite_data import build_data as build_wide_data

SNAPSHOTS = ("short", "2")
STEPS = 100


def build_data():
    data = build_wide_data()
    data["cases"] = [
        c for c in data["cases"] if c["window"] in ("short", "adjacent", "permuted", "zero")
    ]
    data["segment_b_steps"] = STEPS
    data["adapter_method"] = "frozen_segment_a_plus_fresh_default_b"
    return data


def validate_data(data):
    if data != build_data():
        raise ValueError("Fresh-adapter frozen task inputs changed")
