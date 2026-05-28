# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
LoRA adapter utilities for MapAnything.

Injects low-rank adapters into the image encoder and multi-view transformer
so the model can be fine-tuned for stylized inputs with minimal parameters.
"""

import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model

# Linear layer name suffixes common across DINOv2 / ViT / uniception transformers.
_DEFAULT_ATTENTION_SUFFIXES = {"q", "k", "v", "proj", "q_proj", "k_proj", "v_proj", "out_proj"}
_DEFAULT_MLP_SUFFIXES = {"fc1", "fc2", "w1", "w2", "w3"}
_DEFAULT_TARGET_SUFFIXES = _DEFAULT_ATTENTION_SUFFIXES | _DEFAULT_MLP_SUFFIXES


def probe_target_modules(model: nn.Module, include_mlp: bool = False) -> list[str]:
    """Return the set of linear-layer name suffixes found inside attention / MLP blocks.

    Walks all named modules and collects the *leaf* name (last segment) of every
    nn.Linear whose ancestor hierarchy contains an attention-related keyword.
    Pass the result directly to :func:`apply_lora` as ``target_modules``.

    Args:
        model: The MapAnything (or any PyTorch) model to probe.
        include_mlp: Also include linear layers found inside MLP / FFN blocks.

    Returns:
        Sorted list of unique module name suffixes suitable for LoraConfig.target_modules.
    """
    attention_keywords = {"attn", "attention", "self_attn", "cross_attn", "query", "key", "value"}
    mlp_keywords = {"mlp", "ffn", "feed_forward"}

    found: set[str] = set()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        parts = name.lower().split(".")
        in_attn = any(kw in part for part in parts for kw in attention_keywords)
        in_mlp = any(kw in part for part in parts for kw in mlp_keywords)
        leaf = name.split(".")[-1]
        if in_attn and leaf in _DEFAULT_TARGET_SUFFIXES:
            found.add(leaf)
        if include_mlp and in_mlp and leaf in _DEFAULT_MLP_SUFFIXES:
            found.add(leaf)
    return sorted(found)


def apply_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: list[str] | None = None,
    dropout: float = 0.05,
    include_mlp: bool = False,
) -> PeftModel:
    """Wrap *model* with LoRA adapters and freeze all base parameters.

    Args:
        model: Pretrained MapAnything model.
        rank: LoRA rank r (number of low-rank components).
        alpha: LoRA scaling factor (lora_alpha).  Effective scale = alpha / rank.
        target_modules: List of linear layer name suffixes to adapt.
            If None, auto-detected via :func:`probe_target_modules`.
        dropout: Dropout applied to the LoRA paths.
        include_mlp: When auto-detecting, also target MLP / FFN layers.

    Returns:
        PEFT-wrapped model.  Only LoRA parameters have requires_grad=True.
    """
    if target_modules is None:
        target_modules = probe_target_modules(model, include_mlp=include_mlp)
        if not target_modules:
            raise RuntimeError(
                "probe_target_modules found no linear layers matching attention keywords. "
                "Pass target_modules explicitly."
            )

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def save_lora_weights(model: PeftModel, path: str) -> None:
    """Save only the LoRA adapter weights to *path*."""
    model.save_pretrained(path)


def load_lora_weights(base_model: nn.Module, path: str) -> PeftModel:
    """Load LoRA adapter weights from *path* onto *base_model*.

    Args:
        base_model: The original pretrained MapAnything model (without adapters).
        path: Directory previously written by :func:`save_lora_weights`.

    Returns:
        PEFT-wrapped model with loaded adapter weights.
    """
    return PeftModel.from_pretrained(base_model, path)
