import numpy as np

from gradient_interp.paper_mixture import BANKS, weighted_estimators


def test_mixture_density_correction_recovers_known_target_not_proposal_reward():
    # Balanced mixture has P(success)=.5; target has P(success)=.25.
    target = np.log(np.array([[0.25, 0.25, 0.75, 0.75]]))
    warm = np.log(np.array([[0.75, 0.75, 0.25, 0.25]]))
    rewards = np.array([[[1.0], [1.0], [0.0], [0.0]]])
    unit = np.array([[[[1.0, 1.0, -1 / 3, -1 / 3]]]])
    result = weighted_estimators(unit, rewards, target, warm, [0.0, 0.2], range(4), "mixture")
    np.testing.assert_allclose(result["baseline"], 0.25)
    np.testing.assert_allclose(result["snis"][:, 0], 0, atol=1e-15)
    np.testing.assert_allclose(result["linear"][:, 1], 0.05)
    small = weighted_estimators(unit, rewards, target, warm, [1e-6], range(4), "mixture")
    np.testing.assert_allclose(small["linear"], small["snis"], rtol=1e-6)


def test_bank_allocation_and_common_score_shift():
    assert (
        len(BANKS["base256"][0]) == len(BANKS["warm256"][0]) == len(BANKS["mixture256"][0]) == 256
    )
    assert sum(i < 256 for i in BANKS["mixture256"][0]) == 128
    rng = np.random.default_rng(16)
    target = rng.normal(size=(2, 512))
    warm = rng.normal(size=(2, 512))
    unit = rng.normal(size=(3, 6, 2, 512))
    rewards = rng.integers(0, 2, size=(2, 512, 4))
    args = (rewards, target, warm, [0.003, 0.03], *BANKS["mixture256"])
    a = weighted_estimators(unit, *args)
    b = weighted_estimators(unit + 100, *args)
    for name in ("linear", "snis", "baseline", "ess"):
        np.testing.assert_allclose(a[name], b[name], atol=1e-12)
