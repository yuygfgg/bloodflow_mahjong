"""Versioned Actor checkpoint helpers shared by SL and PPO."""

from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Iterator

import bloodflow_mahjong as bm
import torch
from torch import Tensor, nn

from .contracts import TRAINING_INPUT_SCHEMA, validate_engine_contract
from .model import BloodFlowTransformer, TransformerConfig


ACTOR_CHECKPOINT_FORMAT = "bloodflow-mahjong-actor"
ACTOR_CHECKPOINT_VERSION = 1
_ACTOR_PREFIXES = ("static_encoder.", "history_encoder.", "actor.")
_PAYLOAD_KEYS = {
    "format",
    "version",
    "training_input_schema",
    "engine_rules_version",
    "model_config",
    "actor",
    "metadata",
}


def is_actor_parameter(name: str) -> bool:
    return name.startswith(_ACTOR_PREFIXES)


def actor_parameters(model: BloodFlowTransformer) -> Iterator[nn.Parameter]:
    for name, parameter in model.named_parameters():
        if is_actor_parameter(name):
            yield parameter


def actor_state_dict(model: BloodFlowTransformer) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if is_actor_parameter(name)
    }


def save_actor_checkpoint(
    path: Path,
    model: BloodFlowTransformer,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    validate_engine_contract()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format": ACTOR_CHECKPOINT_FORMAT,
            "version": ACTOR_CHECKPOINT_VERSION,
            "training_input_schema": TRAINING_INPUT_SCHEMA,
            "engine_rules_version": int(bm.ENGINE_RULES_VERSION),
            "model_config": asdict(model.config),
            "actor": actor_state_dict(model),
            "metadata": dict(metadata or {}),
        },
        temporary,
    )
    temporary.replace(path)


def load_actor_checkpoint(path: Path, device: torch.device) -> BloodFlowTransformer:
    validate_engine_contract()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != _PAYLOAD_KEYS:
        raise ValueError(f"{path} is not a supported Actor checkpoint")
    if payload["format"] != ACTOR_CHECKPOINT_FORMAT:
        raise ValueError(f"{path} has an unknown Actor checkpoint format")
    if int(payload["version"]) != ACTOR_CHECKPOINT_VERSION:
        raise ValueError(f"{path} has an unsupported Actor checkpoint version")
    if payload["training_input_schema"] != TRAINING_INPUT_SCHEMA:
        raise ValueError(f"{path} uses a different training input schema")
    if int(payload["engine_rules_version"]) != int(bm.ENGINE_RULES_VERSION):
        raise ValueError(f"{path} uses a different engine rules version")

    config_state = payload["model_config"]
    config_fields = {field.name for field in fields(TransformerConfig)}
    if not isinstance(config_state, dict) or set(config_state) != config_fields:
        raise ValueError(f"{path} has an invalid model config")
    model = BloodFlowTransformer(TransformerConfig(**config_state))
    expected_actor = {
        name for name in model.state_dict() if is_actor_parameter(name)
    }
    actor = payload["actor"]
    if not isinstance(actor, dict) or set(actor) != expected_actor:
        raise ValueError(f"{path} has an invalid Actor state")
    missing, unexpected = model.load_state_dict(actor, strict=False)
    expected_missing = set(model.state_dict()) - expected_actor
    if set(missing) != expected_missing or unexpected:
        raise ValueError(f"{path} has an invalid Actor state")
    return model.to(device)
