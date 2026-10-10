import hashlib
from contextlib import contextmanager

import torch


class ParameterSpace:
    def __init__(self, model):
        self.named = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        if not self.named:
            raise ValueError("No trainable parameters selected")
        self.size = sum(parameter.numel() for _, parameter in self.named)

    def layout(self):
        result, offset = [], 0
        for name, parameter in self.named:
            result.append(
                {
                    "name": name,
                    "shape": list(parameter.shape),
                    "offset": offset,
                    "size": parameter.numel(),
                }
            )
            offset += parameter.numel()
        return result

    def vector(self):
        return torch.cat([p.detach().reshape(-1).cpu().float() for _, p in self.named])

    def gradient(self):
        return torch.cat(
            [
                torch.zeros(p.numel())
                if p.grad is None
                else p.grad.detach().reshape(-1).cpu().float()
                for _, p in self.named
            ]
        )

    def fingerprint(self):
        return hashlib.sha256(self.vector().numpy().tobytes()).hexdigest()

    @contextmanager
    def temporary_update(self, delta):
        if delta.shape != (self.size,) or not torch.isfinite(delta).all():
            raise ValueError("Update must be a finite vector matching the parameter layout")
        saved = [p.detach().clone() for _, p in self.named]
        try:
            offset = 0
            with torch.no_grad():
                for _, parameter in self.named:
                    part = delta[offset : offset + parameter.numel()].reshape(parameter.shape)
                    parameter.add_(part.to(parameter))
                    offset += parameter.numel()
            yield
        finally:
            with torch.no_grad():
                for (_, parameter), original in zip(self.named, saved, strict=True):
                    parameter.copy_(original)
