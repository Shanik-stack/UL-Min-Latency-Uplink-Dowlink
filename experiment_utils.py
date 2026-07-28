from __future__ import annotations

import json
import os
from datetime import datetime
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
    "unweighted_sum_rate": "unwt",
    "equal_priority_sum_rate": "unwt",
    "uniform_weighted_sum_rate": "uniwt",
    "remaining_bits_weighted_sum_rate": "rembits",
    "weighted_sum_rate": "rembits",
    "priority_weighted_sum_rate": "weighted",
    "inverse_cnr_weighted_sum_rate": "invcnr",
    "asynchronality_weighted_sum_rate": "async",
    "projected_latency_gap_weighted_sum_rate": "async",
    "projected_completion_latency_gap_weighted_sum_rate": "async",
    "inverse_channel_gain_weighted_sum_rate": "invgain",
    "blended_network_rate": "blend",
    "blended_uniform_weighted_sum_rate": "blenduni",
    "blended_inverse_cnr_weighted_sum_rate": "blendcnr",
    "blended_remaining_bits_weighted_sum_rate": "blendbits",
    "blended_inverse_channel_gain_weighted_sum_rate": "blendgain",
}

SCOPE_TAG_ALIASES = {
    "bs_shared_net": "bs",
    "per_user_nets": "user",
}

SHARED_N_TARGET_MODE_TAG_ALIASES = {
    "shared_n_targets": "jointn",
    "per_user_n_targets": "usern",
}

UPDATE_MODE_TAG_ALIASES = {
    "precoder_net": "net",
    "direct_precoder": "dir",
}


def parse_seed_list(seed_text: str) -> list[int]:
    return [int(part.strip()) for part in str(seed_text).split(",") if part.strip()]


def parse_optional_seed_values(seed_values: Any) -> list[int]:
    if seed_values is None:
        return []
    if isinstance(seed_values, str):
        text = seed_values.strip()
        if text == "":
            return []
        return parse_seed_list(text)
    if isinstance(seed_values, (list, tuple)):
        return [int(value) for value in seed_values]
    return [int(seed_values)]


def build_train_seeds_from_num_train_seeds(num_train_seeds: int, test_seed: int) -> list[int]:
    seed_limit = int(num_train_seeds)
    if seed_limit < 1:
        raise ValueError("num_train_seeds must be at least 1.")
    train_seeds = [seed for seed in range(1, seed_limit + 1) if seed != int(test_seed)]
    if len(train_seeds) == 0:
        raise ValueError(
            "Resolved training-seed set is empty. Increase num_train_seeds or change test_seed."
        )
    return train_seeds


def resolve_monte_carlo_train_and_test_seeds(
    *,
    cli_train_seeds: Any = None,
    cli_num_train_seeds: int | None = None,
    cli_test_seed: int | None = None,
    config_train_seeds: Any = None,
    config_num_train_seeds: int | None = None,
    config_test_seed: int | None = None,
    fallback_train_seeds: Any = "0,1,2",
    fallback_test_seed: int = 3,
) -> tuple[list[int], int]:
    test_seed = int(
        cli_test_seed
        if cli_test_seed is not None
        else config_test_seed
        if config_test_seed is not None
        else fallback_test_seed
    )

    explicit_cli_train_seeds = parse_optional_seed_values(cli_train_seeds)
    if len(explicit_cli_train_seeds) > 0:
        return explicit_cli_train_seeds, test_seed

    if cli_num_train_seeds is not None:
        return build_train_seeds_from_num_train_seeds(int(cli_num_train_seeds), test_seed), test_seed

    config_explicit_train_seeds = parse_optional_seed_values(config_train_seeds)
    if len(config_explicit_train_seeds) > 0:
        return config_explicit_train_seeds, test_seed

    if config_num_train_seeds is not None:
        return build_train_seeds_from_num_train_seeds(int(config_num_train_seeds), test_seed), test_seed

    fallback_explicit_train_seeds = parse_optional_seed_values(fallback_train_seeds)
    if len(fallback_explicit_train_seeds) > 0:
        return fallback_explicit_train_seeds, test_seed

    raise ValueError("Could not resolve Monte Carlo training seeds.")


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


def compact_shared_n_target_mode_tag(mode_name: str) -> str:
    normalized = _normalize_tag_token(mode_name)
    return SHARED_N_TARGET_MODE_TAG_ALIASES.get(normalized, normalized)


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


def current_local_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
