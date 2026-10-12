import numpy as np

from gradient_interp.confirmation_report import control_intervals


def test_removal_share_is_scale_invariant_with_paired_resampling():
    values = np.random.default_rng(14).normal(size=(3, 4, 3, 5))
    stats = control_intervals(
        {"atom_removed": values, "random_removed": values.copy(), "rescaled_full": values * 2},
        [0, 1, 2, 0],
        [{"kind": "pure", "batch_size": 4}] * 4,
        repeats=96,
    )
    for row in stats["atom_minus_control"].values():
        assert row["difference"] == 0
        assert row["crossed_seed_batch_probe_95ci"] == [0, 0]


def test_removal_share_detects_known_target_concentration():
    control = np.ones((3, 4, 3, 5))
    atom = np.zeros_like(control)
    targets = [0, 1, 2, 0]
    for batch, target in enumerate(targets):
        atom[:, batch, target] = 1
    stats = control_intervals(
        {"atom_removed": atom, "rescaled_full": control},
        targets,
        [{"kind": "pure", "batch_size": 4}] * 4,
        repeats=96,
    )
    row = stats["atom_minus_control"]["rescaled_full"]
    np.testing.assert_allclose(row["difference"], 2 / 3)
    np.testing.assert_allclose(row["crossed_seed_batch_probe_95ci"], [2 / 3, 2 / 3])
