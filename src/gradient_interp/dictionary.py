"""Gradient Atoms normalization with magnitude-preserving inference.

Reference: https://arxiv.org/abs/2603.14665, sections 3.2-3.3.
Upstream implementation inspected at a83858557b411249097861f25d57eb13020eb334:
https://github.com/jrosseruk/gradient_atoms/blob/a83858557b411249097861f25d57eb13020eb334/src/gradient_atoms/dictionary.py
The default uses the same sklearn estimator and lasso-LARS encoder. Unlike a
direction-only analysis, encoding restores each row's norm for effect prediction.
The SVD projection below is an explicit alternative to the paper's EKFAC basis.
"""

from dataclasses import dataclass

import torch


@dataclass
class Projection:
    basis: torch.Tensor
    scale: torch.Tensor

    @classmethod
    def fit(cls, gradients, rank, whiten=True):
        if gradients.ndim != 2 or not torch.isfinite(gradients).all():
            raise ValueError("Gradients must be a finite matrix")
        if not 1 <= rank <= min(gradients.shape):
            raise ValueError("Projection rank exceeds the number of examples or parameters")
        # Uncentered SVD preserves the origin: a zero update must stay zero.
        _, singular_values, basis = torch.svd_lowrank(
            gradients, q=min(rank + 8, min(gradients.shape)), niter=4
        )
        tolerance = torch.finfo(gradients.dtype).eps * max(gradients.shape) * singular_values[0]
        effective_rank = min(rank, int((singular_values > tolerance).sum()))
        if effective_rank == 0:
            raise ValueError("Cannot learn a projection from zero gradients")
        basis = basis[:, :effective_rank].contiguous()
        coordinates = gradients @ basis
        scale = coordinates.square().mean(0).sqrt() if whiten else torch.ones(effective_rank)
        scale = scale.clamp_min(torch.finfo(gradients.dtype).eps)
        return cls(basis, scale)

    def encode(self, gradients):
        return (gradients @ self.basis) / self.scale

    def decode(self, coordinates):
        return (coordinates * self.scale) @ self.basis.T


def sparse_code(values, atoms, sparsity):
    """Signed orthogonal matching pursuit; coefficients retain update magnitude."""
    if not 1 <= sparsity <= min(atoms.shape):
        raise ValueError("Sparsity must be between 1 and min(atom count, dimension)")
    was_vector = values.ndim == 1
    values = values.unsqueeze(0) if was_vector else values
    codes = values.new_zeros((len(values), len(atoms)))
    for row, value in enumerate(values):
        residual = value.clone()
        selected = []
        for _ in range(sparsity):
            scores = (atoms @ residual).abs()
            if selected:
                scores[selected] = -1
            index = int(scores.argmax())
            if scores[index] <= 1e-7 * value.norm().clamp_min(1e-20):
                break
            selected.append(index)
            coefficients = torch.linalg.lstsq(atoms[selected].T, value).solution
            residual = value - coefficients @ atoms[selected]
            if residual.norm() <= 1e-6 * value.norm():
                break
        if selected:
            codes[row, selected] = coefficients
    return codes[0] if was_vector else codes


@dataclass
class SparseDictionary:
    atoms: torch.Tensor
    sparsity: int
    history: list[float]
    algorithm: str = "ksvd"
    alpha: float = 0.1

    @classmethod
    def fit(cls, values, n_atoms, sparsity, iterations, seed=0, algorithm="minibatch", alpha=0.1):
        if values.ndim != 2 or not torch.isfinite(values).all():
            raise ValueError("Dictionary input must be a finite matrix")
        if not 1 <= n_atoms <= len(values):
            raise ValueError("Atom count must not exceed the number of fitting examples")
        if not 1 <= sparsity <= min(n_atoms, values.shape[1]) or iterations < 1:
            raise ValueError("Invalid dictionary sparsity or iteration count")
        norms = values.norm(dim=1, keepdim=True)
        nonzero = norms[:, 0] > 1e-12
        normalized = values[nonzero] / norms[nonzero]
        if len(normalized) < n_atoms:
            raise ValueError("Too few nonzero examples for the requested dictionary")
        if algorithm == "minibatch":
            from sklearn.decomposition import MiniBatchDictionaryLearning

            estimator = MiniBatchDictionaryLearning(
                n_components=n_atoms,
                alpha=alpha,
                transform_alpha=alpha,
                batch_size=min(256, len(normalized)),
                max_iter=iterations,
                transform_algorithm="lasso_lars",
                random_state=seed,
                n_jobs=1,
            )
            estimator.fit(normalized.numpy())
            atoms = torch.from_numpy(estimator.components_).to(values)
            codes = torch.from_numpy(estimator.transform(normalized.numpy())).to(values)
            error = float((normalized - codes @ atoms).square().mean())
            return cls(atoms, sparsity, [error], algorithm, alpha)
        if algorithm != "ksvd":
            raise ValueError(f"Unknown dictionary algorithm: {algorithm}")
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(normalized), generator=generator)[:n_atoms]
        atoms = normalized[indices].clone()
        history = []
        # Optional fixed-L0 comparison: Aharon, Elad, Bruckstein (2006), algorithm 1.
        # https://doi.org/10.1109/TSP.2006.881199
        for _ in range(iterations):
            codes = sparse_code(normalized, atoms, sparsity)
            for j in range(n_atoms):
                active = codes[:, j].abs() > 1e-10
                residual = normalized - codes @ atoms
                if not active.any():
                    index = int(residual.square().sum(1).argmax())
                    candidate = residual[index]
                    if candidate.norm() > 1e-8:
                        atoms[j] = candidate / candidate.norm()
                    continue
                error = residual[active] + codes[active, j, None] * atoms[j]
                left, singular, right = torch.linalg.svd(error, full_matrices=False)
                atoms[j] = right[0]
                codes[active, j] = left[:, 0] * singular[0]
            history.append(float((normalized - codes @ atoms).square().mean()))
        return cls(atoms, sparsity, history)

    def encode(self, values):
        if self.algorithm == "minibatch":
            from sklearn.decomposition import sparse_encode

            vector = values.ndim == 1
            matrix = values.unsqueeze(0) if vector else values
            norms = matrix.norm(dim=1, keepdim=True).clamp_min(1e-20)
            codes = sparse_encode(
                (matrix / norms).numpy(),
                self.atoms.numpy(),
                algorithm="lasso_lars",
                alpha=self.alpha,
                n_jobs=1,
            )
            result = torch.from_numpy(codes).to(values) * norms
            return result[0] if vector else result
        return sparse_code(values, self.atoms, self.sparsity)

    def decode(self, coefficients):
        return coefficients @ self.atoms
