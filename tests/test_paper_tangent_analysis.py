import json

import numpy as np
import pytest

from gradient_interp.paper_tangent_analysis import (
    boot_statistics,
    draw_bootstrap,
    guard,
    interval_fields,
    scoped,
)


def test_response_bootstrap_preserves_pairing_and_zero_change():
    rng = np.random.default_rng(13)
    f = rng.normal(size=(1, 1, 1, 4, 32, 1))
    f = np.broadcast_to(f, (2, 1, 2, 4, 32, 1)).copy()
    f[:, :, 1] = 0
    y = np.broadcast_to(rng.normal(size=(1, 1, 4, 128, 1)), (2, 1, 4, 128, 1))
    fp, yt, cq, eq = draw_bootstrap(f, y, 300)
    np.testing.assert_array_equal(fp[:, 0], fp[:, 1])
    np.testing.assert_array_equal(yt[:, 0], yt[:, 1])
    np.testing.assert_array_equal(fp[:, :, :, 1], 0)
    np.testing.assert_array_equal(cq.sum(1), 1)
    np.testing.assert_array_equal(eq.sum(1), 1)
    assert np.max(np.abs(np.corrcoef(fp[:, 0, 0, 0, 0, 0], yt[:, 0, 0, 0, 0])[0, 1])) < 0.2


def test_transfer_uses_calibration_draw_and_evaluation_target():
    p = np.array([1.0, 9.0, 100.0, 200.0])[None, None, None, None, :, None]
    y = np.array([10.0, 20.0, 30.0, 40.0])[None, None, None, :, None]
    sp, sy = scoped(p, y, "evaluation_calibration_transfer", np.array([[1.0, 0.0]]))
    np.testing.assert_array_equal(sp.ravel(), [1, 1])
    np.testing.assert_array_equal(sy.ravel(), [30, 40])
    result = boot_statistics(sp, sy, np.array([[0.0, 1.0]]))
    np.testing.assert_allclose(result["rmse"], 39)
    np.testing.assert_allclose(result["nrmse"], 39 / 40)
    np.testing.assert_allclose(result["mse_difference_vs_zero"], 39**2 - 40**2)


def test_degenerate_relative_errors_are_not_dropped_from_intervals():
    p = np.ones((2, 1, 1, 1, 2, 1))
    y = np.zeros((2, 1, 1, 2, 1))
    y[1] = 1
    result = boot_statistics(p, y, np.full((2, 2), 0.5))
    assert np.isnan(result["nrmse"][0]).all()
    interval = interval_fields(result["nrmse"].ravel(), "x")
    assert interval == {"x_lo": None, "x_hi": None, "x_defined_replicates": 1}
    assert np.isfinite(result["rmse"]).all()


def test_guard_rejects_incomplete_run_before_outcome_loading(tmp_path):
    plan = tmp_path / "analysis-plan.json"
    plan.write_text(json.dumps({"outcomes_read_before_plan": False}))
    root = tmp_path / "run"
    root.mkdir()
    for name, value in (("controller", {"status": "running"}), ("integrity", {}), ("manifest", {})):
        (root / f"{name}.json").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Both tangent runs"):
        guard([root], plan)
