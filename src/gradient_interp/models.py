"""Completion-only causal LM losses and reproducible fixed-checkpoint evaluations.

HF integration follows the official causal-LM and PEFT interfaces:
https://huggingface.co/docs/transformers/tasks/language_modeling
https://huggingface.co/docs/peft/developer_guides/lora
LoRA: https://arxiv.org/abs/2106.09685
"""

import re
from dataclasses import dataclass
from pathlib import Path
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
    dtype: str = "float32"
    max_length: int = 160
    width: int = 24
    trainable_pattern: str = ".*"
    lora_rank: int = 0
    lora_alpha: int = 8
    lora_targets: tuple[str, ...] = ("q_proj", "v_proj")
    score_batch_size: int = 1
    attention: str = "eager"


class LanguageModel:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.resolved_revision = None
        if config.dtype not in {"float32", "bfloat16", "float16"}:
            raise ValueError("dtype must be float32, bfloat16, or float16")
        if config.max_length < 3:
            raise ValueError("max_length must be at least 3")
        if config.score_batch_size < 1 or config.attention not in {"eager", "sdpa"}:
            raise ValueError("Invalid scoring batch size or attention implementation")
        if config.kind == "tiny":
            self.tokenizer = ByteTokenizer()
            self.model = TinyLM(config.width, config.max_length)
        elif config.kind == "hf":
            from huggingface_hub import hf_hub_download
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if not config.name:
                raise ValueError("A Hugging Face model name is required")
            revision = config.revision
            if not Path(config.name).is_dir():
                cached_config = Path(hf_hub_download(config.name, "config.json", revision=revision))
                if cached_config.parent.parent.name == "snapshots":
                    revision = cached_config.parent.name
                    self.resolved_revision = revision
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.name, revision=revision, trust_remote_code=False
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                config.name,
                revision=revision,
                dtype=getattr(torch, config.dtype),
                trust_remote_code=False,
                attn_implementation=config.attention,
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

    def log_probability(self, prompt, completion, reduction="mean"):
        ids, start, length = self.encode(prompt, completion)
        logits = self.model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
        selected = logits[:, start : start + length].float().log_softmax(-1)
        targets = ids[:, start + 1 : start + length + 1]
        token_logprobs = selected.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return token_logprobs.mean() if reduction == "mean" else token_logprobs.sum()

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
        reduction = "mean" if probe.metric == "mean_logprob" else "sum"
        score = self.log_probability(probe.prompt, probe.positive, reduction)
        if probe.negative is not None:
            score = score - self.log_probability(probe.prompt, probe.negative, reduction)
        if probe.metric == "probability":
            # With a comparison: P(positive) / (P(positive) + P(negative)).
            # Without one: probability of the exact continuation token sequence.
            return score.sigmoid() if probe.negative is not None else score.exp()
        return score

    def scores(self, probes):
        with torch.no_grad():
            if self.config.score_batch_size == 1:
                return torch.stack([self.probe_score(probe).cpu() for probe in probes])
            pairs, references = [], []
            for probe in probes:
                pos = len(pairs)
                pairs.append((probe.prompt, probe.positive, probe.metric))
                neg = None
                if probe.negative is not None:
                    neg = len(pairs)
                    pairs.append((probe.prompt, probe.negative, probe.metric))
                references.append((pos, neg))
            values = self.batched_log_probabilities(pairs)
            scores = []
            for probe, (pos, neg) in zip(probes, references, strict=True):
                value = values[pos] if neg is None else values[pos] - values[neg]
                if probe.metric == "probability":
                    value = value.exp() if neg is None else value.sigmoid()
                scores.append(value)
            return torch.stack(scores).cpu()

    def batched_log_probabilities(self, pairs):
        """Length-bucketed, right-padded scoring with the same completion mask."""
        encoded = [self.encode(prompt, completion) for prompt, completion, _ in pairs]
        order = sorted(range(len(pairs)), key=lambda i: encoded[i][0].shape[1])
        values = [None] * len(pairs)
        for offset in range(0, len(order), self.config.score_batch_size):
            indices = order[offset : offset + self.config.score_batch_size]
            lengths = torch.tensor(
                [encoded[i][0].shape[1] for i in indices], device=self.config.device
            )
            ids = nn.utils.rnn.pad_sequence(
                [encoded[i][0][0] for i in indices],
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )
            mask = torch.arange(ids.shape[1], device=ids.device)[None, :] < lengths[:, None]
            logits = self.model(input_ids=ids, attention_mask=mask.long()).logits[:, :-1].float()
            targets = ids[:, 1:]
            token_scores = logits.gather(-1, targets[..., None]).squeeze(-1)
            token_scores = token_scores - logits.logsumexp(-1)
            for row, i in enumerate(indices):
                _, start, count = encoded[i]
                selected = token_scores[row, start : start + count]
                values[i] = selected.mean() if pairs[i][2] == "mean_logprob" else selected.sum()
        return values

    def generate(self, prompts, max_new_tokens=64):
        """Greedy completions for fixed, prespecified qualitative checks."""
        if self.config.kind != "hf":
            raise ValueError("Generation checks require a pretrained HF model")
        responses = []
        with torch.no_grad():
            for prompt in prompts:
                start = self.tokenizer.bos_token_id
                if start is None:
                    start = self.tokenizer.eos_token_id
                prefix = torch.tensor(
                    [[start] + self.tokenizer.encode(prompt, add_special_tokens=False)],
                    device=self.config.device,
                )
                result = self.model.generate(
                    input_ids=prefix,
                    attention_mask=torch.ones_like(prefix),
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                responses.append(
                    self.tokenizer.decode(result[0, prefix.shape[1] :], skip_special_tokens=True)
                )
        return responses

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
