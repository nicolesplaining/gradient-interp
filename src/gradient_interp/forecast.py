"""Predict actual SGD effects from independently calibrated dictionary atoms.

Gradient Atoms supplies the decomposition/steering motivation:
https://arxiv.org/abs/2603.14665
The full-gradient first-order baseline uses the same local gradient inner product
underlying TracIn (https://arxiv.org/abs/2002.08484), for a single checkpoint.
It is a supervised, per-probe reference, not an influence-function estimate.
"""

from dataclasses import dataclass

import torch

from .dictionary import Projection, SparseDictionary


@dataclass
class Forecast:
    coefficients: torch.Tensor
    amplitudes: torch.Tensor
    predicted: torch.Tensor
    reconstructed_update: torch.Tensor
    contributions: torch.Tensor
    selected_atom: int


class AtomForecaster:
    def __init__(self, projection, dictionary):
        self.projection = projection
        self.dictionary = dictionary
        raw_atoms = projection.decode(dictionary.atoms)
        self.atom_norms = raw_atoms.norm(dim=1).clamp_min(1e-20)
        self.directions = raw_atoms / self.atom_norms[:, None]
        self.effects = None

    @classmethod
    def fit(
        cls,
        gradients,
        rank,
        n_atoms,
        sparsity,
        iterations,
        seed=0,
        whiten=True,
        algorithm="minibatch",
        alpha=0.1,
    ):
        projection = Projection.fit(gradients, rank, whiten)
        dictionary = SparseDictionary.fit(
            projection.encode(gradients), n_atoms, sparsity, iterations, seed, algorithm, alpha
        )
        return cls(projection, dictionary)

    def calibrate(self, score_update, radius):
        if radius <= 0:
            raise ValueError("Calibration radius must be positive")
        effects, discrepancies = [], []
        for direction in self.directions:
            full = score_update(radius * direction) - score_update(-radius * direction)
            full = full / (2 * radius)
            half = score_update(radius / 2 * direction) - score_update(-radius / 2 * direction)
            half = half / radius
            effects.append(half)
            discrepancies.append((full - half).abs())
        self.effects = torch.stack(effects)
        self.calibration_discrepancy = torch.stack(discrepancies)
        self.calibration_radius = radius
        return self.effects

    def predict(self, update, target_probe=None):
        if self.effects is None:
            raise RuntimeError("Calibrate the forecaster before predicting")
        coefficients = self.dictionary.encode(self.projection.encode(update))
        amplitudes = coefficients * self.atom_norms
        contributions = amplitudes[:, None] * self.effects
        importance = (
            contributions.norm(dim=1)
            if target_probe is None
            else contributions[:, target_probe].abs()
        )
        return Forecast(
            coefficients,
            amplitudes,
            contributions.sum(0),
            amplitudes @ self.directions,
            contributions,
            int(importance.argmax()),
        )

    def state(self):
        return {
            "basis": self.projection.basis,
            "scale": self.projection.scale,
            "atoms": self.dictionary.atoms,
            "effects": self.effects,
            "calibration_discrepancy": self.calibration_discrepancy,
        }


def prediction_metrics(predictions, observed):
    predictions = torch.as_tensor(predictions, dtype=torch.float64).flatten()
    observed = torch.as_tensor(observed, dtype=torch.float64).flatten()
    if predictions.shape != observed.shape or not len(observed):
        raise ValueError("Predicted and observed effects must have equal, nonzero sizes")
    if not (torch.isfinite(predictions).all() and torch.isfinite(observed).all()):
        raise ValueError("Metrics require finite predictions and observations")
    error = predictions - observed
    meaningful = observed.abs() > 1e-7
    varying = predictions.std(unbiased=False) > 1e-12 and observed.std(unbiased=False) > 1e-12
    correlation = (
        float(torch.corrcoef(torch.stack([predictions, observed]))[0, 1]) if varying else None
    )
    return {
        "n": len(observed),
        "rmse": float(error.square().mean().sqrt()),
        "mae": float(error.abs().mean()),
        "relative_l2_error": float(error.norm() / observed.norm()) if observed.norm() > 0 else None,
        "pearson": correlation,
        "sign_accuracy": float(
            (predictions[meaningful].sign() == observed[meaningful].sign()).double().mean()
        )
        if meaningful.any()
        else None,
        "sign_scored_count": int(meaningful.sum()),
    }
