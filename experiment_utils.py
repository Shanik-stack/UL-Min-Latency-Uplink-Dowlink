from __future__ import annotations

import json
import os
from typing import Any

import numpy as np


METHOD_TAG_ALIASES = {
    "convergence_per_epoch_baseline": "conv",
    "greedy_safe_sweep": "greedy",
    "monte_carlo_precoder_net_train_test": "mc",
    "monte_carlo_precoder_net_test": "mc_test",
}

OBJECTIVE_TAG_ALIASES = {
    "user_rate": "user",
    "unweighted_sum_rate": "sum",
    "remaining_bits_weighted_sum_rate": "bitsw",
    "weighted_sum_rate": "bitsw",
    "blended_network_rate": "blend",
}

SCOPE_TAG_ALIASES = {
    "bs_shared_net": "bs",
    "per_user_nets": "user",
}

UPDATE_MODE_TAG_ALIASES = {
    "precoder_net": "net",
    "direct_precoder": "dir",
}


def parse_seed_list(seed_text: str) -> list[int]:
    return [int(part.strip()) for part in str(seed_text).split(",") if part.strip()]


def _normalize_tag_token(value: str) -> str:
    text = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def compact_method_tag(method_name: str) -> str:
    normalized = _normalize_tag_token(method_name)
    return METHOD_TAG_ALIASES.get(normalized, normalized)


def compact_objective_tag(objective_mode: str) -> str:
    normalized = _normalize_tag_token(objective_mode)
    return OBJECTIVE_TAG_ALIASES.get(normalized, normalized)


def compact_scope_tag(scope_name: str) -> str:
    normalized = _normalize_tag_token(scope_name)
    return SCOPE_TAG_ALIASES.get(normalized, normalized)


def compact_update_mode_tag(update_mode: str) -> str:
    normalized = _normalize_tag_token(update_mode)
    return UPDATE_MODE_TAG_ALIASES.get(normalized, normalized)


def join_compact_tag_parts(*parts: str | None) -> str:
    compact_parts = [_normalize_tag_token(part) for part in parts if str(part or "").strip()]
    return "_".join(part for part in compact_parts if part)


def compact_cfg_stem(cfg_name: str) -> str:
    stem = os.path.splitext(os.path.basename(str(cfg_name)))[0]
    normalized = _normalize_tag_token(stem)
    for prefix in ("downlink_", "uplink_", "config_"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
    return normalized or "cfg"


def make_method_result_tag(method_name: str, cfg_name: str, *, seed: int | None = None) -> str:
    cfg_stem = compact_cfg_stem(cfg_name)
    safe_method = _normalize_tag_token(method_name)
    if seed is None:
        return f"{safe_method}__{cfg_stem}"
    return f"{safe_method}__{cfg_stem}__s{int(seed)}"


def make_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return [make_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def save_json(data: dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_serializable(data), f, indent=4)


def save_text(lines: list[str], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
