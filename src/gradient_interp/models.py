"""Completion-only causal LM losses and reproducible fixed-checkpoint evaluations.

HF integration follows the official causal-LM and PEFT interfaces:
https://huggingface.co/docs/transformers/tasks/language_modeling
https://huggingface.co/docs/peft/developer_guides/lora
LoRA: https://arxiv.org/abs/2106.09685
"""

import re
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import nn

from .parameters import ParameterSpace


class ByteTokenizer:
    bos_token_id = 256
    eos_token_id = 257
    pad_token_id = 258

    def encode(self, text, add_special_tokens=False):
        return list(text.encode("utf-8"))


class TinyLM(nn.Module):
    """Offline numerical smoke model; randomly initialized, not pretrained."""

    def __init__(self, width=24, max_length=160):
        super().__init__()
        self.embedding = nn.Embedding(259, width)
        self.position = nn.Embedding(max_length, width)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    width, 2, width * 2, dropout=0, batch_first=True, norm_first=True
                )
            ]
        )
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 259, bias=False)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden = self.embedding(input_ids) + self.position(positions)
        mask = torch.triu(
            torch.ones(len(positions), len(positions), dtype=torch.bool, device=input_ids.device),
            diagonal=1,
        )
        padding = attention_mask == 0 if attention_mask is not None else None
        for layer in self.layers:
            hidden = layer(hidden, src_mask=mask, src_key_padding_mask=padding)
        return SimpleNamespace(logits=self.head(self.norm(hidden)))


@dataclass
class ModelConfig:
    kind: str = "tiny"
    name: str = ""
    revision: str = "main"
    device: str = "cpu"
    max_length: int = 160
    width: int = 24
    trainable_pattern: str = ".*"
    lora_rank: int = 0
    lora_alpha: int = 8
    lora_targets: tuple[str, ...] = ("q_proj", "v_proj")


class LanguageModel:
    def __init__(self, config: ModelConfig):
        self.config = config
        if config.max_length < 3:
            raise ValueError("max_length must be at least 3")
        if config.kind == "tiny":
            self.tokenizer = ByteTokenizer()
            self.model = TinyLM(config.width, config.max_length)
        elif config.kind == "hf":
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if not config.name:
                raise ValueError("A Hugging Face model name is required")
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.name, revision=config.revision, trust_remote_code=False
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                config.name,
                revision=config.revision,
                torch_dtype=torch.float32,
                trust_remote_code=False,
                attn_implementation="eager",
            )
            self.model.config.use_cache = False
            if config.lora_rank:
                from peft import LoraConfig, TaskType, get_peft_model

                self.model = get_peft_model(
                    self.model,
                    LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        r=config.lora_rank,
                        lora_alpha=config.lora_alpha,
                        lora_dropout=0.0,
                        bias="none",
                        target_modules=list(config.lora_targets),
                    ),
                )
        else:
            raise ValueError(f"Unknown model kind: {config.kind}")
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(
                parameter.requires_grad and re.search(config.trainable_pattern, name) is not None
            )
        self.model.to(config.device).eval()
        self.space = ParameterSpace(self.model)
        self._encoded = {}

    def encode(self, prompt, completion):
        key = (prompt, completion)
        if key in self._encoded:
            return self._encoded[key]
        # Explicit separate tokenization avoids ambiguous BPE boundary masking.
        # This defines the scored continuation, rather than silently leaking prompt loss.
        context = self.tokenizer.encode(prompt, add_special_tokens=False)
        target = self.tokenizer.encode(completion, add_special_tokens=False)
        start = self.tokenizer.bos_token_id
        if start is None:
            start = self.tokenizer.eos_token_id
        if start is None:
            raise ValueError("Tokenizer must have a BOS or EOS token")
        prefix = [start] + context
        if not target:
            raise ValueError("Completion has no tokens")
        if len(prefix) + len(target) > self.config.max_length:
            raise ValueError("Example exceeds max_length; increase it or shorten the input")
        ids = torch.tensor([prefix + target], device=self.config.device)
        # logits[:, i] predicts ids[:, i+1]; first target is at len(prefix)-1.
        result = (ids, len(prefix) - 1, len(target))
        self._encoded[key] = result
        return result

    def log_probability(self, prompt, completion):
        ids, start, length = self.encode(prompt, completion)
        logits = self.model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
        selected = logits[:, start : start + length].float().log_softmax(-1)
        targets = ids[:, start + 1 : start + length + 1]
        return selected.gather(-1, targets.unsqueeze(-1)).squeeze(-1).mean()

    def loss(self, example):
        return -self.log_probability(example.prompt, example.completion)

    def gradient(self, examples):
        # Mean of per-example mean completion-token losses. A long document has
        # the same batch weight as a short one; the actual SGD step uses this too.
        self.model.zero_grad(set_to_none=True)
        for example in examples:
            (self.loss(example) / len(examples)).backward()
        result = self.space.gradient()
        self.model.zero_grad(set_to_none=True)
        if not torch.isfinite(result).all():
            raise ValueError("Nonfinite model gradient")
        return result

    def probe_score(self, probe):
        score = self.log_probability(probe.prompt, probe.positive)
        if probe.negative is not None:
            score = score - self.log_probability(probe.prompt, probe.negative)
        return score

    def scores(self, probes):
        with torch.no_grad():
            return torch.stack([self.probe_score(probe).cpu() for probe in probes])

    def probe_gradients(self, probes):
        rows = []
        for probe in probes:
            self.model.zero_grad(set_to_none=True)
            self.probe_score(probe).backward()
            rows.append(self.space.gradient())
        self.model.zero_grad(set_to_none=True)
        return torch.stack(rows)

    def updated_scores(self, delta, probes):
        with self.space.temporary_update(delta):
            return self.scores(probes)
