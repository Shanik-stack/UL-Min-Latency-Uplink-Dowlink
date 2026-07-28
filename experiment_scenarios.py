from __future__ import annotations

from typing import Any, Sequence

import numpy as np


PAYLOAD_COMPLETION_MODE = "payload_completion"
FIXED_BLOCK_TARGETS_MODE = "fixed_block_targets"


def _normalize_mode(mode: Any) -> str:
    text = str(mode if mode is not None else PAYLOAD_COMPLETION_MODE).strip().lower()
    payload_aliases = {
        "payload": PAYLOAD_COMPLETION_MODE,
        "payload_completion": PAYLOAD_COMPLETION_MODE,
        "remaining_bits": PAYLOAD_COMPLETION_MODE,
    }
    removed_aliases = {
        "fixed",
        "fixed_blocks",
        "fixed_block_targets",
        "block_targets",
    }
    if text in payload_aliases:
        return payload_aliases[text]
    if text in removed_aliases:
        raise ValueError(
            "The fixed_block_targets scenario has been removed. "
            "Use 'payload_completion' instead."
        )
    raise ValueError(
        "Unsupported experiment scenario mode "
        f"{mode!r}. Expected: '{PAYLOAD_COMPLETION_MODE}'."
    )


def _as_int_vector(values: Any, K: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=int)
    if arr.ndim == 0:
        arr = np.full(K, int(arr.item()), dtype=int)
    if arr.shape != (K,):
        raise ValueError(f"{name} must have shape ({K},), got {arr.shape}.")
    return arr


def _as_int_matrix(values: Any, K: int, L: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=int)
    if arr.ndim == 0:
        return np.full((K, L), int(arr.item()), dtype=int)
    if arr.ndim == 1:
        if arr.shape == (K,):
            return np.repeat(arr.reshape(K, 1), L, axis=1)
        if arr.shape == (L,):
            return np.repeat(arr.reshape(1, L), K, axis=0)
    if arr.shape != (K, L):
        raise ValueError(f"{name} must have shape ({K}, {L}), ({K},), ({L},), or scalar; got {arr.shape}.")
    return arr


def normalize_experiment_scenario_config(
    scenario_cfg: Any,
    *,
    system_params: dict[str, Any],
    max_total_blocks: int | None = None,
) -> dict[str, Any]:
    raw_cfg = scenario_cfg if isinstance(scenario_cfg, dict) else {}
    if bool(raw_cfg.get("_normalized_experiment_scenario", False)):
        return dict(raw_cfg)
    K = int(system_params["K"])
    system_payload = _as_int_vector(system_params["B"], K, "system B")
    mode = _normalize_mode(raw_cfg.get("mode", raw_cfg.get("name", PAYLOAD_COMPLETION_MODE)))
    shared_cfg = {
        "mode": mode,
        "skip_infeasible_blocks": bool(raw_cfg.get("skip_infeasible_blocks", True)),
        "skip_block_adds_full_T_latency": bool(raw_cfg.get("skip_block_adds_full_T_latency", True)),
        "track_skipped_blocks": bool(raw_cfg.get("track_skipped_blocks", True)),
    }
    payload_cfg = raw_cfg.get("payload_bits", {})
    payload_source = str(raw_cfg.get("payload_bits_source", "system_B")).strip().lower()
    payload_values = None
    if isinstance(payload_cfg, dict):
        payload_source = str(payload_cfg.get("source", payload_source)).strip().lower()
        payload_values = payload_cfg.get("values")
    elif payload_cfg not in ({}, None):
        payload_source = "explicit"
        payload_values = payload_cfg
    if payload_source == "system_b":
        payload_source = "system_B"
    if payload_source not in {"system_B", "explicit"}:
        raise ValueError(
            "payload_bits_source must be 'system_B' or 'explicit', "
            f"got {payload_source!r}."
        )
    if payload_source == "explicit":
        if payload_values is None:
            payload_values = raw_cfg.get("payload_bits_values", raw_cfg.get("payload_bits"))
        payload_bits = _as_int_vector(payload_values, K, "payload_bits")
    else:
        payload_bits = np.array(system_payload, copy=True)
    return {
        **shared_cfg,
        "_normalized_experiment_scenario": True,
        "payload_bits_source": payload_source,
        "payload_bits": payload_bits.tolist(),
    }


def build_experiment_scenario(
    system_params: dict[str, Any],
    sim_params: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    scenario_cfg = normalize_experiment_scenario_config(
        sim_params.get("experiment_scenario", {}),
        system_params=system_params,
        max_total_blocks=sim_params.get("max_total_blocks"),
    )
    payload_bits = _as_int_vector(
        scenario_cfg.get("payload_bits", system_params["B"]),
        int(system_params["K"]),
        "payload_bits",
    )
    return {
        "mode": PAYLOAD_COMPLETION_MODE,
        "seed": int(seed),
        "payload_bits_per_user": payload_bits.tolist(),
        "per_user_total_target_bits": payload_bits.tolist(),
        "total_target_bits": int(np.sum(payload_bits)),
        "skip_infeasible_blocks": bool(scenario_cfg["skip_infeasible_blocks"]),
        "skip_block_adds_full_T_latency": bool(scenario_cfg["skip_block_adds_full_T_latency"]),
        "track_skipped_blocks": bool(scenario_cfg["track_skipped_blocks"]),
        "termination_rule": "until_payload_drained",
    }


def build_experiment_scenarios_for_seeds(
    system_params: dict[str, Any],
    sim_params: dict[str, Any],
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    return [
        build_experiment_scenario(system_params, sim_params, seed=int(seed))
        for seed in list(seeds)
    ]


def build_experiment_scenario_summary(scenario: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "mode": PAYLOAD_COMPLETION_MODE,
        "seed": int(scenario.get("seed", 0)),
        "termination_rule": str(scenario.get("termination_rule", "")),
        "skip_infeasible_blocks": bool(scenario.get("skip_infeasible_blocks", True)),
        "skip_block_adds_full_T_latency": bool(scenario.get("skip_block_adds_full_T_latency", True)),
        "track_skipped_blocks": bool(scenario.get("track_skipped_blocks", True)),
        "per_user_total_target_bits": [int(v) for v in scenario.get("per_user_total_target_bits", [])],
        "total_target_bits": int(scenario.get("total_target_bits", 0)),
        "payload_bits_per_user": [int(v) for v in scenario.get("payload_bits_per_user", [])],
    }
    return summary


def build_experiment_scenario_summary_lines(summary: dict[str, Any]) -> list[str]:
    lines = [
        "Experiment scenario summary",
        f"Mode: {PAYLOAD_COMPLETION_MODE}",
        f"Seed: {int(summary.get('seed', 0))}",
        f"Termination rule: {summary.get('termination_rule', '')}",
        f"Skip infeasible blocks: {bool(summary.get('skip_infeasible_blocks', True))}",
        f"Skip adds full-T latency: {bool(summary.get('skip_block_adds_full_T_latency', True))}",
        f"Track skipped blocks: {bool(summary.get('track_skipped_blocks', True))}",
        f"Per-user total target bits: {summary.get('per_user_total_target_bits', [])}",
        f"Total target bits: {int(summary.get('total_target_bits', 0))}",
        f"Payload bits per user: {summary.get('payload_bits_per_user', [])}",
    ]
    return lines
