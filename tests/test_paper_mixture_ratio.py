import numpy as np

from gradient_interp.paper_mixture import BANKS, weighted_estimators
from gradient_interp.paper_mixture_ratio import direct_banks, unique_trajectories


def test_direct_ratio_matches_known_exponential_tilt_for_every_bank():
    rng = np.random.default_rng(18)
    p, w = rng.normal(size=(2, 512)), rng.normal(size=(2, 512))
    y = rng.normal(size=(2, 512, 4))
    u = rng.normal(size=(1, 1, 2, 512))
    direct = direct_banks(p + 0.03 * u[0, 0], p, w, y)
    zero = direct_banks(p, p, w, y)
    for name, (idx, proposal) in BANKS.items():
        approx = weighted_estimators(u, y, p, w, [0.03], idx, proposal)
        np.testing.assert_allclose(direct[f"{name}_change"], approx["snis"][0, 0, 0], atol=1e-12)
        np.testing.assert_allclose(zero[f"{name}_change"], 0, atol=1e-12)


def test_dedup_retains_multiplicity_and_never_merges_questions():
    data = {"probes": [{"id": str(i), "prompt": str(i)} for i in range(2)]}
    rows = [{"id": str(i), "sample": s, "token_ids": [s % 2]} for i in range(2) for s in range(512)]
    unique, inverse = unique_trajectories(data, {"rows": rows})
    assert len(unique) == 4 and inverse.shape == (2, 512)
    assert len(set(inverse[0]) & set(inverse[1])) == 0
    for i in range(2):
        for s in range(512):
            assert unique[inverse[i, s]]["token_ids"] == rows[i * 512 + s]["token_ids"]
