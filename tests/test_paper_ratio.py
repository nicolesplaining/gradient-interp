import math

import numpy as np
import torch

from gradient_interp.paper_ratio import direct_reweight, pack_trajectories, score_logits


def test_ragged_trajectory_scores_exclude_prompt_and_padding_include_last_token():
    ids, mask, spans = pack_trajectories([[1, 2], [1]], [[3, 4, 0], [2]], 0, "cpu")
    logits = torch.zeros((2, 5, 5))
    logits[0, 0, :] = torch.tensor([100.0, 0, 0, 0, 0])  # unscored prompt prediction
    logits[1, 1:, :] = torch.tensor([100.0, 0, 0, 0, 0])  # unscored padding prediction
    scores = score_logits(logits, ids, spans, 0.8)
    np.testing.assert_allclose(scores, [-3 * math.log(5), -math.log(5)], rtol=1e-6)
    assert mask.sum(1).tolist() == [5, 2]


def test_direct_ratios_recover_known_finite_outcome_distribution_with_multiplicity():
    # Exact baseline frequencies: P(success)=1/4. Updated probability is 3/4.
    reward = np.array([[[1.0], [0.0], [0.0], [0.0]]])
    ratio = np.log(np.array([[3.0, 1 / 3, 1 / 3, 1 / 3]]))
    result = direct_reweight(ratio, reward)
    np.testing.assert_allclose(result["snis_level"], 0.75)
    np.testing.assert_allclose(result["snis_change"], 0.5)
    np.testing.assert_allclose(result["log_mean_weight"], 0, atol=1e-15)
    missing = direct_reweight(ratio, np.zeros_like(reward))
    np.testing.assert_array_equal(missing["snis_change"], 0)
