"""Candidate context-preserving forecasts from residual-gradient SAE features.

For one linear map y = W x, dL/dW = sum_t g_t x_t^T exactly. If an SAE
reconstructs g_t as sum_k z_tk d_k + b, retaining sum_t z_tk x_t connects
each feature to the inputs that produced it. Feature counts alone discard this.

These operators implement that identity, not an empirically validated method.
They apply to one linear module with its true output adjoints. Arbitrary residual
gradients cannot be substituted, and a whole-model LoRA update is not equivalent.
Gradients must already include the intended token/document/batch loss reduction.
Forecasts of downstream scores are first-order approximations.
"""

from dataclasses import dataclass

import torch

from .residual import resolve_module


def capture_linear_example(backend, example, module_name, *, verify_weight_gradient=True):
    """Collect true adjoints at a plain linear module, including prompt positions.

    Completion-only loss can still produce nonzero prompt-position adjoints
    through later attention layers. Dropping those positions loses weight updates.
    Returned adjoints use this example's mean completion-token loss. A caller
    forming a batch must divide each example's adjoints by the batch size.
    The explicit weight gradient is returned to audit the factorization.
    """
    _, module = resolve_module(backend.model, module_name)
    if type(module) is not torch.nn.Linear or not module.weight.requires_grad:
        raise ValueError("Capture requires an unfused, trainable torch.nn.Linear weight")
    if backend.config.gradient_checkpointing:
        raise ValueError("This initial capture implementation requires checkpointing off")
    captured = []

    def capture(_module, args, output):
        captured.append((args[0], output))

    handle = module.register_forward_hook(capture)
    backend.model.zero_grad(set_to_none=True)
    try:
        loss = backend.loss(example)
        if len(captured) != 1:
            raise ValueError("The selected linear module must execute once per example")
        inputs, output = captured[0]
        if inputs.ndim != 3 or inputs.shape[0] != 1:
            raise ValueError("Expected one token sequence per example")
        if verify_weight_gradient:
            adjoints, weight_gradient = torch.autograd.grad(loss, (output, module.weight))
        else:
            (adjoints,) = torch.autograd.grad(loss, (output,))
            weight_gradient = None
        return {
            "inputs": inputs[0].detach().float().cpu(),
            "adjoints": adjoints[0].detach().float().cpu(),
            "weight_gradient": (
                weight_gradient.detach().float().cpu() if weight_gradient is not None else None
            ),
            "loss": float(loss.detach()),
        }
    finally:
        handle.remove()
        backend.model.zero_grad(set_to_none=True)


@dataclass
class LinearFeatureUpdate:
    directions: torch.Tensor  # feature x output width, final row is the SAE offset
    contexts: torch.Tensor  # feature x input width, including the offset's context
    learning_rate: float

    def weight_delta(self):
        return -self.learning_rate * self.directions.T @ self.contexts

    def forecast(self, inputs, score_gradients):
        """Directional derivative of a scalar score on any probe token positions."""
        if inputs.ndim != 2 or score_gradients.ndim != 2:
            raise ValueError("Probe inputs and score gradients must be matrices")
        if inputs.shape != (len(score_gradients), self.contexts.shape[1]):
            raise ValueError("Probe input dimensions do not match the linear module")
        if score_gradients.shape[1] != self.directions.shape[1]:
            raise ValueError("Probe gradient dimensions do not match the linear module")
        return (
            -self.learning_rate
            * ((score_gradients @ self.directions.T) * (inputs @ self.contexts.T)).sum()
        )

    def component(self, index):
        """An exact additive component of this approximate weight update."""
        if not 0 <= index < self.directions.shape[0]:
            raise ValueError("Feature index outside the decomposition")
        return -self.learning_rate * torch.outer(self.directions[index], self.contexts[index])


def feature_update(
    inputs, codes, directions, offset, learning_rate, *, pairing="paired", order=None
):
    """Keep token pairing, replace it by independent means, or permute it as a control.

    codes and directions must decode gradients in raw gradient units. An affine
    whitening transform must be undone first, including its nonzero mean.
    Context storage scales as features times input width, not one scalar/feature.
    This is deliberately not described as equal-storage to the old OMP forecast.
    """
    if any(x.ndim != 2 for x in (inputs, codes, directions)) or offset.ndim != 1:
        raise ValueError("Expected token input/code matrices, direction matrix and offset vector")
    if inputs.shape[0] == 0 or len(inputs) != len(codes):
        raise ValueError("Need matching nonempty token positions")
    if directions.shape != (codes.shape[1], len(offset)):
        raise ValueError("Feature and output widths do not match")
    arrays = (inputs, codes, directions, offset)
    if len({x.dtype for x in arrays}) != 1 or len({x.device for x in arrays}) != 1:
        raise ValueError("Tensors must use the same dtype and device")
    if not all(bool(torch.isfinite(x).all()) for x in arrays):
        raise ValueError("Inputs contain non-finite values")
    if not 0 <= learning_rate < float("inf"):
        raise ValueError("Learning rate must be finite and nonnegative")
    augmented = torch.cat((codes, torch.ones_like(codes[:, :1])), dim=1)
    if codes.shape[1] == 0:
        augmented = inputs.new_ones((len(inputs), 1))
    raw_directions = torch.cat((directions, offset[None]), dim=0)
    if pairing == "paired":
        if order is not None:
            raise ValueError("An order is only allowed for the permuted control")
        contexts = augmented.T @ inputs
    elif pairing == "independent":
        if order is not None:
            raise ValueError("An order is only allowed for the permuted control")
        contexts = augmented.sum(0)[:, None] * inputs.mean(0)[None]
    elif pairing == "permuted":
        if order is None or order.dtype != torch.long or order.device != inputs.device:
            raise ValueError("Provide a long permutation on the input device")
        expected = torch.arange(len(inputs), device=inputs.device)
        if order.shape != expected.shape or not torch.equal(order.sort().values, expected):
            raise ValueError("Order must be a permutation of the token positions")
        contexts = augmented.T @ inputs[order]
    else:
        raise ValueError("Unknown input pairing")
    return LinearFeatureUpdate(raw_directions, contexts, float(learning_rate))


def unwhiten_sae_decoder(sae, whitening):
    """Return a raw-unit affine decoder for an existing residual.TopKSAE."""
    weight = sae.decoder.weight.detach().T
    center = sae.center.detach()
    basis = whitening.basis.to(device=weight.device, dtype=weight.dtype)
    scales = whitening.scales.to(device=weight.device, dtype=weight.dtype)
    mean = whitening.mean.to(device=weight.device, dtype=weight.dtype)
    return (weight * scales) @ basis.T, (center * scales) @ basis.T + mean
