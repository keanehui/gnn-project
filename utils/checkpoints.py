"""Checkpoint helpers that keep compiled and eager models compatible."""

from collections import OrderedDict
from typing import MutableMapping

import torch.nn as nn


COMPILED_PREFIX = "_orig_mod."


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the original module when torch.compile wrapped the model."""
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def strip_compiled_prefix(
    state_dict: MutableMapping[str, object],
) -> MutableMapping[str, object]:
    """Remove torch.compile key prefixes so checkpoints stay portable."""
    if not state_dict:
        return state_dict

    if not any(key.startswith(COMPILED_PREFIX) for key in state_dict.keys()):
        return state_dict

    cleaned_items = []
    for key, value in state_dict.items():
        if key.startswith(COMPILED_PREFIX):
            key = key[len(COMPILED_PREFIX) :]
        cleaned_items.append((key, value))

    if isinstance(state_dict, OrderedDict):
        return OrderedDict(cleaned_items)
    return state_dict.__class__(cleaned_items)


def load_model_state(
    model: nn.Module,
    state_dict: MutableMapping[str, object],
    strict: bool = True,
):
    """Load checkpoints saved from either eager or compiled models."""
    base_model = unwrap_model(model)
    cleaned_state_dict = strip_compiled_prefix(state_dict)
    return base_model.load_state_dict(cleaned_state_dict, strict=strict)
