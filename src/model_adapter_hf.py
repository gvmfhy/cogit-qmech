#!/usr/bin/env python3
"""
HF Adapter to emulate minimal TransformerLens interfaces for generation and hooks.

Provides:
- model.to_tokens(text)
- model.generate(tokens, ...)
- model.hooks(fwd_hooks=[(hook_name, fn)]) context manager

Hook name format expected: 'blocks.{layer}.hook_resid_post'
This adapter parses the layer index and registers a forward hook on the
corresponding HF block module, modifying its output hidden states.
"""

import re
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


HOOK_NAME_PATTERN = re.compile(r"blocks\.(\d+)\.hook_resid_post")


def _get_layers_list(hf_model):
    # LLaMA/Qwen style
    if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
        return hf_model.model.layers
    # GPT-2 style
    if hasattr(hf_model, "transformer") and hasattr(hf_model.transformer, "h"):
        return hf_model.transformer.h
    raise RuntimeError("Unsupported HF model architecture for hooking layers")


class HFHookContext:
    def __init__(self, hf_model, fwd_hooks: List[Tuple[str, callable]]):
        self.hf_model = hf_model
        self.fwd_hooks = fwd_hooks
        self._handles = []

    def __enter__(self):
        layers = _get_layers_list(self.hf_model)
        for hook_name, fn in self.fwd_hooks:
            m = HOOK_NAME_PATTERN.match(hook_name)
            if not m:
                continue
            layer_idx = int(m.group(1))
            if layer_idx < 0 or layer_idx >= len(layers):
                continue

            def make_hook(user_fn):
                def hook(module, inputs, output):
                    # output can be Tensor or tuple; we expect hidden states tensor
                    if isinstance(output, tuple):
                        hidden = output[0]
                        updated = user_fn(hidden, None)
                        if isinstance(updated, torch.Tensor):
                            return (updated,) + output[1:]
                        return output
                    else:
                        updated = user_fn(output, None)
                        return updated if isinstance(updated, torch.Tensor) else output
                return hook

            handle = layers[layer_idx].register_forward_hook(make_hook(fn))
            self._handles.append(handle)
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()


class HFModelWrapper:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            # Align pad with eos for causal models
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def to_tokens(self, text: str) -> torch.Tensor:
        enc = self.tokenizer(text, return_tensors="pt")
        return enc.input_ids.to(self.device)

    def generate(self, tokens: torch.Tensor, max_new_tokens: int = 25, temperature: float = 0.8, top_k: int = 50, stop_at_eos: bool = True, verbose: bool = False) -> torch.Tensor:
        do_sample = temperature is not None and temperature > 0.0
        eos_id = self.tokenizer.eos_token_id if stop_at_eos else None
        out = self.model.generate(
            input_ids=tokens,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_k=top_k if do_sample else None,
            eos_token_id=eos_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return out

    def hooks(self, fwd_hooks: List[Tuple[str, callable]]):
        return HFHookContext(self.model, fwd_hooks)


class HFAdapter:
    """Thin adapter that exposes a .model with TL-like interfaces for our pipeline"""

    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = HFModelWrapper(model_name, device)
        # Expose dimensions if available
        try:
            self.hidden_dim = self.model.model.config.hidden_size
            self.num_layers = getattr(self.model.model.config, "num_hidden_layers", None)
        except Exception:
            self.hidden_dim = None
            self.num_layers = None


