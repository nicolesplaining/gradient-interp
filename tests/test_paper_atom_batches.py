import numpy as np
import torch

from gradient_interp.paper_atom_batches import (
    atom_forecasts,
    condition_direction,
    covariance_effects,
    intervention_conditions,
)
from gradient_interp.paper_atom_data import FORMATS, all_items, checks, make_data


def test_counterfactual_data_and_split_separation():
    data = make_data()
    assert data == make_data()
    items = all_items(data)
    assert len(items) == len({q["prompt"] for q in items}) == 48
    assert len({tuple(sorted(q["expected"])) for q in items}) == 48
    for q in items:
        for j, answer in enumerate(q["answers"]):
            c = checks(answer, q["expected"])
            assert c["answer_values_and_order"]
            assert c["numbered"] == (FORMATS[j % 3] == "numbered")
            assert c["bullets"] == (FORMATS[j % 3] == "bullets")
    # Reversals, duplicates and invented vocabulary values must fail content validation.
    q = items[0]
    assert not checks(", ".join(reversed(q["expected"])), q["expected"])["answer_values_and_order"]
    assert not checks(q["answers"][0] + q["expected"][0], q["expected"])["answer_values_and_order"]
    assert not checks(", ".join(q["expected"]).upper(), q["expected"], "lowercase")[
        "answer_values_and_order"
    ]
    for b in range(4):
        block = data["batches"][4 * b : 4 * b + 4]
        assert all(
            [q["id"] for q in x["documents"]] == [q["id"] for q in block[0]["documents"]]
            for x in block
        )


def test_atom_sum_equals_direct_covariance_with_magnitude_and_sign():
    rng = np.random.default_rng(42)
    scores = rng.normal(size=(3, 17, 5))
    responses = rng.integers(0, 2, size=(3, 17, 2)).astype(float)
    coefficients = rng.normal(size=(4, 5)) * 7
    atom_effects = covariance_effects(scores, responses)
    prediction = atom_forecasts(coefficients, atom_effects, 0.003)
    direct_unit = np.einsum("bd,qsd->qsb", coefficients, scores)
    direct = 0.003 * covariance_effects(direct_unit, responses).transpose(1, 0, 2)
    np.testing.assert_allclose(prediction, direct, atol=1e-14)
    np.testing.assert_allclose(
        atom_forecasts(2 * coefficients, atom_effects, 0.003), 2 * prediction
    )
    np.testing.assert_allclose(atom_forecasts(-coefficients, atom_effects, 0.003), -prediction)


def test_missing_events_are_retained_as_zero_effects():
    rng = np.random.default_rng(4)
    unit = rng.normal(size=(2, 16, 6))
    rewards = np.ones((2, 16, 3))
    assert np.count_nonzero(covariance_effects(unit, rewards)) == 0


def test_selective_removal_and_norm_controls():
    rng = np.random.default_rng(5)
    directions = torch.tensor(rng.normal(size=(7, 13)), dtype=torch.float32)
    gradients = torch.tensor(rng.normal(size=(16, 13)), dtype=torch.float32)
    coefficients = rng.normal(size=(16, 7))
    coefficients[0, 0] = 0
    batches = [dict(target="numbered" if i % 2 == 0 else "bullets") for i in range(16)]
    cards = dict(
        numbered=dict(atom_index=0, orientation=1), bullets=dict(atom_index=1, orientation=-1)
    )
    conditions = intervention_conditions(gradients, coefficients, directions, batches, cards, 52)
    assert len(conditions) == 72
    assert conditions == intervention_conditions(
        gradients, coefficients, directions, batches, cards, 52
    )
    for b in range(16):
        full, named, random, scale = [
            condition_direction(c, gradients, directions) for c in conditions[4 * b : 4 * b + 4]
        ]
        torch.testing.assert_close(full, gradients[b])
        torch.testing.assert_close((full - named).norm(), (full - random).norm())
        torch.testing.assert_close(named.norm(), scale.norm())
    torch.testing.assert_close(
        condition_direction(conditions[1], gradients, directions), gradients[0]
    )
    for c in conditions[64:]:
        g = condition_direction(c, gradients, directions)
        torch.testing.assert_close(g.norm(), gradients.norm(dim=1).median() * abs(c["strength"]))
