from __future__ import annotations

from typing import Any, Sequence

import numpy as np


PAYLOAD_COMPLETION_MODE = "payload_completion"
FIXED_BLOCK_TARGETS_MODE = "fixed_block_targets"


def _normalize_mode(mode: Any) -> str:
    text = str(mode if mode is not None else PAYLOAD_COMPLETION_MODE).strip().lower()
    aliases = {
        "payload": PAYLOAD_COMPLETION_MODE,
        "payload_completion": PAYLOAD_COMPLETION_MODE,
        "remaining_bits": PAYLOAD_COMPLETION_MODE,
        "fixed": FIXED_BLOCK_TARGETS_MODE,
        "fixed_blocks": FIXED_BLOCK_TARGETS_MODE,
        "fixed_block_targets": FIXED_BLOCK_TARGETS_MODE,
        "block_targets": FIXED_BLOCK_TARGETS_MODE,
    }
    if text not in aliases:
        raise ValueError(
            "Unsupported experiment scenario mode "
            f"{mode!r}. Expected one of: {sorted(set(aliases.values()))}."
        )
    return aliases[text]


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

    if mode == PAYLOAD_COMPLETION_MODE:
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

    fixed_cfg = raw_cfg.get("fixed_block_targets", {})
    if not isinstance(fixed_cfg, dict):
        raise ValueError("fixed_block_targets must be a mapping when mode='fixed_block_targets'.")
    default_num_blocks = 1
    if max_total_blocks is not None:
        default_num_blocks = max(1, min(int(max_total_blocks), 4))
    elif "monte_carlo_training_blocks_per_seed" in raw_cfg:
        default_num_blocks = max(1, int(raw_cfg["monte_carlo_training_blocks_per_seed"]))
    elif "precoder_net_train_blocks_per_seed" in raw_cfg:
        default_num_blocks = max(1, int(raw_cfg["precoder_net_train_blocks_per_seed"]))
    num_blocks = int(
        raw_cfg.get(
            "fixed_num_blocks",
            fixed_cfg.get("num_blocks", default_num_blocks),
        )
    )
    if num_blocks <= 0:
        raise ValueError(f"fixed_block_targets num_blocks must be positive, got {num_blocks}.")

    generation_mode = str(
        raw_cfg.get(
            "fixed_block_target_generation_mode",
            fixed_cfg.get("generation_mode", "constant"),
        )
    ).strip().lower()
    valid_generation_modes = {"constant", "explicit", "uniform_integer"}
    if generation_mode not in valid_generation_modes:
        raise ValueError(
            "fixed_block_targets generation_mode must be one of "
            f"{sorted(valid_generation_modes)}, got {generation_mode!r}."
        )

    explicit_targets = None
    constant_targets = None
    target_min_bits = None
    target_max_bits = None
    target_step_bits = None

    if generation_mode == "explicit":
        explicit_targets = fixed_cfg.get("values", raw_cfg.get("fixed_block_target_values"))
        if explicit_targets is None:
            raise ValueError("fixed_block_targets explicit mode requires 'values'.")
    elif generation_mode == "constant":
        if "per_block_bits" in fixed_cfg or "fixed_block_per_block_bits" in raw_cfg:
            raise ValueError(
                "fixed_block_targets.per_block_bits has been removed. "
                "Use test.B so each user's per-block target bits come directly from B[k]."
            )
        constant_targets = np.array(system_payload, copy=True).tolist()
    else:
        target_min_bits = int(fixed_cfg.get("min_bits", raw_cfg.get("fixed_block_target_min_bits", 1)))
        target_max_bits = int(fixed_cfg.get("max_bits", raw_cfg.get("fixed_block_target_max_bits", target_min_bits)))
        target_step_bits = int(fixed_cfg.get("step_bits", raw_cfg.get("fixed_block_target_step_bits", 1)))
        if target_min_bits < 1 or target_max_bits < target_min_bits or target_step_bits <= 0:
            raise ValueError(
                "fixed_block_targets uniform_integer mode requires min_bits >= 1, "
                "max_bits >= min_bits, and step_bits > 0."
            )

    return {
        **shared_cfg,
        "_normalized_experiment_scenario": True,
        "num_blocks": int(num_blocks),
        "generation_mode": generation_mode,
        "explicit_targets": explicit_targets,
        "constant_targets": constant_targets,
        "target_min_bits": target_min_bits,
        "target_max_bits": target_max_bits,
        "target_step_bits": target_step_bits,
    }


def _rng_for_scenario(seed: int, mode: str) -> np.random.Generator:
    mode_tag = 0 if mode == PAYLOAD_COMPLETION_MODE else 1
    ss = np.random.SeedSequence([int(seed), 1729, mode_tag])
    return np.random.default_rng(ss)


def _build_fixed_block_targets_matrix(
    scenario_cfg: dict[str, Any],
    *,
    system_params: dict[str, Any],
    seed: int,
) -> np.ndarray:
    K = int(system_params["K"])
    L = int(scenario_cfg["num_blocks"])
    generation_mode = str(scenario_cfg["generation_mode"])

    if generation_mode == "explicit":
        matrix = _as_int_matrix(scenario_cfg["explicit_targets"], K, L, "fixed_block_targets.values")
    elif generation_mode == "constant":
        matrix = _as_int_matrix(
            scenario_cfg["constant_targets"],
            K,
            L,
            "fixed_block_targets.system_B_per_user",
        )
    else:
        min_bits = int(scenario_cfg["target_min_bits"])
        max_bits = int(scenario_cfg["target_max_bits"])
        step_bits = int(scenario_cfg["target_step_bits"])
        candidates = np.arange(min_bits, max_bits + step_bits, step_bits, dtype=int)
        candidates = candidates[candidates <= max_bits]
        if candidates.size == 0:
            raise ValueError("fixed_block_targets uniform_integer mode produced an empty candidate grid.")
        rng = _rng_for_scenario(int(seed), FIXED_BLOCK_TARGETS_MODE)
        per_user_targets = rng.choice(candidates, size=(K,), replace=True).astype(int)
        matrix = np.repeat(per_user_targets.reshape(K, 1), L, axis=1)

    matrix = np.asarray(matrix, dtype=int)
    if matrix.shape != (K, L):
        raise ValueError(f"fixed_block_targets matrix must have shape ({K}, {L}), got {matrix.shape}.")
    if np.any(matrix <= 0):
        raise ValueError("fixed_block_targets requires strictly positive target bits in every user/block entry.")
    if not np.all(matrix == matrix[:, [0]]):
        raise ValueError(
            "fixed_block_targets requires each user's target bits to stay constant across blocks."
        )
    return matrix


def build_experiment_scenario(
    system_params: dict[str, Any],
    sim_params: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    K = int(system_params["K"])
    scenario_cfg = normalize_experiment_scenario_config(
        sim_params.get("experiment_scenario", {}),
        system_params=system_params,
        max_total_blocks=sim_params.get("max_total_blocks"),
    )
    mode = str(scenario_cfg["mode"])
    payload_bits = _as_int_vector(scenario_cfg.get("payload_bits", system_params["B"]), K, "payload_bits")

    if mode == PAYLOAD_COMPLETION_MODE:
        return {
            "mode": mode,
            "seed": int(seed),
            "payload_bits_per_user": payload_bits.tolist(),
            "per_user_total_target_bits": payload_bits.tolist(),
            "total_target_bits": int(np.sum(payload_bits)),
            "skip_infeasible_blocks": bool(scenario_cfg["skip_infeasible_blocks"]),
            "skip_block_adds_full_T_latency": bool(scenario_cfg["skip_block_adds_full_T_latency"]),
            "track_skipped_blocks": bool(scenario_cfg["track_skipped_blocks"]),
            "termination_rule": "until_payload_drained",
        }

    block_targets = _build_fixed_block_targets_matrix(
        scenario_cfg,
        system_params=system_params,
        seed=int(seed),
    )
    active_mask = (block_targets > 0).astype(int)
    return {
        "mode": mode,
        "seed": int(seed),
        "num_blocks": int(block_targets.shape[1]),
        "block_bit_targets": block_targets.tolist(),
        "block_active_mask": active_mask.tolist(),
        "per_user_total_target_bits": block_targets.sum(axis=1, dtype=int).tolist(),
        "total_target_bits": int(np.sum(block_targets)),
        "generation_mode": str(scenario_cfg["generation_mode"]),
        "skip_infeasible_blocks": bool(scenario_cfg["skip_infeasible_blocks"]),
        "skip_block_adds_full_T_latency": bool(scenario_cfg["skip_block_adds_full_T_latency"]),
        "track_skipped_blocks": bool(scenario_cfg["track_skipped_blocks"]),
        "termination_rule": "fixed_block_horizon",
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
    mode = str(scenario.get("mode", PAYLOAD_COMPLETION_MODE))
    summary = {
        "mode": mode,
        "seed": int(scenario.get("seed", 0)),
        "termination_rule": str(scenario.get("termination_rule", "")),
        "skip_infeasible_blocks": bool(scenario.get("skip_infeasible_blocks", True)),
        "skip_block_adds_full_T_latency": bool(scenario.get("skip_block_adds_full_T_latency", True)),
        "track_skipped_blocks": bool(scenario.get("track_skipped_blocks", True)),
        "per_user_total_target_bits": [int(v) for v in scenario.get("per_user_total_target_bits", [])],
        "total_target_bits": int(scenario.get("total_target_bits", 0)),
    }
    if mode == PAYLOAD_COMPLETION_MODE:
        summary["payload_bits_per_user"] = [int(v) for v in scenario.get("payload_bits_per_user", [])]
        return summary

    block_targets = np.asarray(scenario.get("block_bit_targets", []), dtype=int)
    summary.update(
        {
            "num_blocks": int(scenario.get("num_blocks", block_targets.shape[1] if block_targets.ndim == 2 else 0)),
            "generation_mode": str(scenario.get("generation_mode", "unknown")),
            "per_block_total_target_bits": (
                block_targets.sum(axis=0, dtype=int).tolist() if block_targets.ndim == 2 else []
            ),
            "active_users_per_block": (
                (block_targets > 0).sum(axis=0, dtype=int).tolist() if block_targets.ndim == 2 else []
            ),
            "block_bit_targets": block_targets.tolist() if block_targets.ndim == 2 else [],
        }
    )
    return summary


def build_experiment_scenario_summary_lines(summary: dict[str, Any]) -> list[str]:
    mode = str(summary.get("mode", PAYLOAD_COMPLETION_MODE))
    lines = [
        "Experiment scenario summary",
        f"Mode: {mode}",
        f"Seed: {int(summary.get('seed', 0))}",
        f"Termination rule: {summary.get('termination_rule', '')}",
        f"Skip infeasible blocks: {bool(summary.get('skip_infeasible_blocks', True))}",
        f"Skip adds full-T latency: {bool(summary.get('skip_block_adds_full_T_latency', True))}",
        f"Track skipped blocks: {bool(summary.get('track_skipped_blocks', True))}",
        f"Per-user total target bits: {summary.get('per_user_total_target_bits', [])}",
        f"Total target bits: {int(summary.get('total_target_bits', 0))}",
    ]
    if mode == PAYLOAD_COMPLETION_MODE:
        lines.append(f"Payload bits per user: {summary.get('payload_bits_per_user', [])}")
        return lines

    lines.extend(
        [
            f"Num blocks: {int(summary.get('num_blocks', 0))}",
            f"Generation mode: {summary.get('generation_mode', 'unknown')}",
            f"Per-block total target bits: {summary.get('per_block_total_target_bits', [])}",
            f"Active users per block: {summary.get('active_users_per_block', [])}",
            "Block target matrix (rows=user, cols=block):",
            f"{summary.get('block_bit_targets', [])}",
        ]
    )
    return lines
