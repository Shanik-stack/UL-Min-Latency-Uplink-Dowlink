from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from experiment_utils import (
    build_train_seeds_from_num_train_seeds,
    compact_cfg_stem,
    compact_method_tag,
    compact_objective_tag,
    compact_scope_tag,
    compact_shared_n_target_mode_tag,
    compact_update_mode_tag,
    current_local_timestamp,
    join_compact_tag_parts,
    make_method_result_tag,
    make_serializable,
    parse_optional_seed_values,
    save_json,
    save_text,
)
from project_paths import (
    build_downlink_result_dirs,
    build_uplink_convergence_result_dirs,
    build_uplink_result_dirs,
)


PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENT_CONFIGS_DIR = PROJECT_ROOT / "Experiment Configs"
DECISION_SWEEP_RUNS_ROOT = PROJECT_ROOT / "Decision Sweep Runs"
DECISION_COMPARISON_RESULTS_ROOT = PROJECT_ROOT / "Results" / "Decision Sweep Comparisons"
DEFAULT_MANIFEST_NAME = "decision_sweep.yaml"

LINK_ALIASES = {
    "ul": "uplink",
    "uplink": "uplink",
    "dl": "downlink",
    "downlink": "downlink",
}

METHOD_ALIASES = {
    "convergence": "convergence",
    "conv": "convergence",
    "convergence_per_epoch": "convergence",
    "monte_carlo": "monte_carlo",
    "monte-carlo": "monte_carlo",
    "mc": "monte_carlo",
}

METHOD_SPECS: dict[tuple[str, str], dict[str, Any]] = {
    ("uplink", "convergence"): {
        "display_name": "Uplink | Convergence per epoch",
        "script_path": PROJECT_ROOT / "Uplink" / "Methods" / "Convergence per epoch" / "main.py",
        "supports_quiet": False,
    },
    ("uplink", "monte_carlo"): {
        "display_name": "Uplink | Monte Carlo",
        "script_path": PROJECT_ROOT / "Uplink" / "Methods" / "Monte Carlo" / "main.py",
        "supports_quiet": False,
    },
    ("downlink", "convergence"): {
        "display_name": "Downlink | Convergence per epoch",
        "script_path": PROJECT_ROOT / "Downlink" / "Methods" / "Convergence per epoch" / "main.py",
        "supports_quiet": True,
    },
    ("downlink", "monte_carlo"): {
        "display_name": "Downlink | Monte Carlo",
        "script_path": PROJECT_ROOT / "Downlink" / "Methods" / "Monte Carlo" / "main.py",
        "supports_quiet": True,
    },
}

DECISION_PATH_TAG_PREFIX = {
    "simulation.convergence_precoder_update_mode": "upd",
    "simulation.constraint_loss_form": "loss",
    "simulation.uplink_rate_model": "rate",
    "simulation.uplink_objective_mode": "obj",
    "simulation.convergence_block_objective_mode": "obj",
    "simulation.n_search_direction": "ndir",
    "simulation.downlink_precoder_net_scope": "scope",
    "simulation.n_kl_reduction_update_scope": "repair",
}

DECISION_VALUE_ALIASES: dict[str, dict[str, str]] = {
    "simulation.convergence_precoder_update_mode": {
        "precoder_net": "precoder_net",
        "direct_precoder": "direct_precoder",
    },
    "simulation.constraint_loss_form": {
        "plain_lagrangian": "plain_lagrangian",
        "augmented_lagrangian": "augmented_lagrangian",
    },
    "simulation.uplink_rate_model": {
        "snr": "snr",
        "sinr": "sinr",
    },
    "simulation.uplink_objective_mode": {
        "user_rate": "unweighted_sum_rate",
        "equal_priority_sum_rate": "unweighted_sum_rate",
        "unweighted_sum_rate": "unweighted_sum_rate",
        "priority_weighted_sum_rate": "inverse_cnr_weighted_sum_rate",
        "inverse_cnr_weighted_sum_rate": "inverse_cnr_weighted_sum_rate",
    },
    "simulation.convergence_block_objective_mode": {
        "user_rate": "unweighted_sum_rate",
        "equal_priority_sum_rate": "unweighted_sum_rate",
        "priority_weighted_sum_rate": "priority_weighted_sum_rate",
        "unweighted_sum_rate": "unweighted_sum_rate",
        "uniform_weighted_sum_rate": "uniform_weighted_sum_rate",
        "weighted_sum_rate": "remaining_bits_weighted_sum_rate",
        "remaining_bits_weighted_sum_rate": "remaining_bits_weighted_sum_rate",
        "inverse_cnr_weighted_sum_rate": "inverse_cnr_weighted_sum_rate",
        "inverse_channel_gain_weighted_sum_rate": "inverse_channel_gain_weighted_sum_rate",
        "backlog_weighted_sum_rate": "remaining_bits_weighted_sum_rate",
        "blended_network_rate": "blended_network_rate",
        "blended_uniform_weighted_sum_rate": "blended_uniform_weighted_sum_rate",
        "blended_inverse_cnr_weighted_sum_rate": "blended_inverse_cnr_weighted_sum_rate",
        "blended_remaining_bits_weighted_sum_rate": "blended_remaining_bits_weighted_sum_rate",
        "blended_inverse_channel_gain_weighted_sum_rate": "blended_inverse_channel_gain_weighted_sum_rate",
    },
    "simulation.n_search_direction": {
        "asc": "ascending",
        "ascending": "ascending",
        "low_to_high": "ascending",
        "desc": "descending",
        "descending": "descending",
        "high_to_low": "descending",
    },
    "simulation.downlink_precoder_net_scope": {
        "per_user_nets": "per_user_nets",
        "bs_shared_net": "bs_shared_net",
    },
    "simulation.n_kl_reduction_update_scope": {
        "all_active_users": "all_active_users",
        "infeasible_users_only": "infeasible_users_only",
        "candidate_and_infeasible_users": "candidate_and_infeasible_users",
    },
}

DECISION_VALUE_TAGS: dict[str, dict[str, str]] = {
    "simulation.convergence_precoder_update_mode": {
        "precoder_net": "net",
        "direct_precoder": "dir",
    },
    "simulation.constraint_loss_form": {
        "plain_lagrangian": "plain",
        "augmented_lagrangian": "aug",
    },
    "simulation.uplink_rate_model": {
        "snr": "snr",
        "sinr": "sinr",
    },
    "simulation.uplink_objective_mode": {
        "unweighted_sum_rate": "unwt",
        "inverse_cnr_weighted_sum_rate": "invcnr",
    },
    "simulation.convergence_block_objective_mode": {
        "priority_weighted_sum_rate": "weighted",
        "unweighted_sum_rate": "unwt",
        "uniform_weighted_sum_rate": "uniwt",
        "remaining_bits_weighted_sum_rate": "rembits",
        "inverse_cnr_weighted_sum_rate": "invcnr",
        "inverse_channel_gain_weighted_sum_rate": "invgain",
        "blended_network_rate": "blend",
        "blended_uniform_weighted_sum_rate": "blenduni",
        "blended_inverse_cnr_weighted_sum_rate": "blendcnr",
        "blended_remaining_bits_weighted_sum_rate": "blendbits",
        "blended_inverse_channel_gain_weighted_sum_rate": "blendgain",
    },
    "simulation.n_search_direction": {
        "ascending": "asc",
        "descending": "desc",
    },
    "simulation.downlink_precoder_net_scope": {
        "per_user_nets": "user",
        "bs_shared_net": "bs",
    },
    "simulation.n_kl_reduction_update_scope": {
        "all_active_users": "allact",
        "infeasible_users_only": "infeas",
        "candidate_and_infeasible_users": "candinf",
    },
}

DECISION_DISPLAY_NAMES = {
    "simulation.convergence_precoder_update_mode": "Convergence precoder update mode",
    "simulation.constraint_loss_form": "Constraint loss form",
    "simulation.uplink_rate_model": "Uplink rate model",
    "simulation.uplink_objective_mode": "Uplink objective",
    "simulation.convergence_block_objective_mode": "Downlink convergence objective",
    "simulation.n_search_direction": "n_kl search direction",
    "simulation.downlink_precoder_net_scope": "Downlink precoder scope",
    "simulation.n_kl_reduction_update_scope": "Reduced n_kl re-optimization scope",
}

CSV_FIELD_ORDER = [
    "status",
    "returncode",
    "link",
    "method",
    "scenario_mode",
    "base_config_name",
    "generated_config_name",
    "decision_tag",
    "is_base_configuration",
    "seed",
    "train_seeds",
    "highlighted_configuration",
    "run_started_at_local",
    "run_completed_at_local",
    "duration_seconds",
    "initial_total_latency",
    "final_total_latency",
    "delta_final_total_latency_vs_base",
    "total_latency_reduction_percent",
    "delta_total_latency_reduction_percent_vs_base",
    "initial_asynchronality_sum",
    "final_asynchronality_sum",
    "delta_final_asynchronality_sum_vs_base",
    "asynchronality_reduction_percent",
    "delta_asynchronality_reduction_percent_vs_base",
    "initial_avg_snr_db",
    "final_avg_snr_db",
    "delta_final_avg_snr_db_vs_base",
    "initial_avg_sinr_db",
    "final_avg_sinr_db",
    "delta_final_avg_sinr_db_vs_base",
    "total_served_bits",
    "delta_total_served_bits_vs_base",
    "total_skipped_blocks",
    "delta_total_skipped_blocks_vs_base",
    "total_target_bits",
    "total_unserved_bits",
    "training_solve_status",
    "epochs_completed",
    "last_epoch_feasible_rollout_queries",
    "last_epoch_total_rollout_queries",
    "last_epoch_feasible_rollout_fraction",
    "result_json_path",
    "log_path",
    "effective_decisions_json",
]


def _normalize_token(value: str) -> str:
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def _safe_path_token(value: str) -> str:
    cleaned = []
    for char in str(value):
        if char.isalnum():
            cleaned.append(char.lower())
        else:
            cleaned.append("_")
    text = "".join(cleaned)
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_") or "value"


def _decision_label(path: str) -> str:
    if path in DECISION_DISPLAY_NAMES:
        return str(DECISION_DISPLAY_NAMES[path])
    return str(path)


def _sorted_effective_decisions(effective_decisions: dict[str, Any]) -> list[tuple[str, Any]]:
    return sorted(effective_decisions.items(), key=lambda item: item[0])


def _normalize_report_text(value: Any) -> str:
    text = str(value)
    replacements = {
        "equal_priority_sum_rate": "unweighted_sum_rate",
        "priority_weighted_sum_rate": "inverse_cnr_weighted_sum_rate",
        "blended_network_rate": "blended_inverse_cnr_weighted_sum_rate",
        "obj-eqsum": "obj-unwt",
        "obj-prio": "obj-invcnr",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _highlighted_configuration_text(effective_decisions: dict[str, Any]) -> str:
    if len(effective_decisions) == 0:
        return "base_configuration_only"
    return "; ".join(
        f"{path.split('.')[-1]}={_normalize_report_text(value)}"
        for path, value in _sorted_effective_decisions(effective_decisions)
    )


def _report_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    latency = _float_or_none(row.get("final_total_latency"))
    reduction = _float_or_none(row.get("total_latency_reduction_percent"))
    return (
        0 if row.get("status") == "completed" else 1,
        latency if latency is not None else float("inf"),
        -(reduction if reduction is not None else float("-inf")),
        str(row.get("decision_tag", "")),
    )


def _resolve_parallelism(
    *,
    requested_value: Any,
    planned_runs: int,
) -> tuple[int, str]:
    if planned_runs < 1:
        return 1, "planned_runs=0_fallback"

    raw_value = requested_value
    if raw_value is None:
        raw_value = "auto"

    if isinstance(raw_value, int):
        if raw_value < 1:
            raise ValueError("max_parallel must be at least 1.")
        return min(int(raw_value), planned_runs), f"fixed({int(raw_value)})"

    text = str(raw_value).strip().lower()
    if text == "":
        text = "auto"

    if text in {"auto", "max", "default"}:
        logical_cpus = os.cpu_count() or 1
        return max(1, min(planned_runs, logical_cpus)), f"auto(logical_cpus={logical_cpus})"

    if text in {"all", "full", "everything"}:
        return max(1, planned_runs), "all_planned_runs"

    parsed = int(text)
    if parsed < 1:
        raise ValueError("max_parallel must be at least 1.")
    return min(parsed, planned_runs), f"fixed({parsed})"


def _resolve_manifest_path(manifest_name: str) -> Path:
    raw = str(manifest_name).strip()
    candidate = Path(raw)
    if candidate.suffix.lower() != ".yaml":
        candidate = Path(f"{raw}.yaml")

    candidates: list[Path] = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.extend(
            [
                PROJECT_ROOT / candidate,
                EXPERIMENT_CONFIGS_DIR / candidate,
                Path.cwd() / candidate,
            ]
        )

    for path in candidates:
        if path.exists():
            return path.resolve()

    searched = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(f"Could not find decision-sweep manifest '{manifest_name}'. Searched:\n{searched}")


def _normalize_link(link_name: str) -> str:
    key = _normalize_token(link_name)
    if key not in LINK_ALIASES:
        raise ValueError(f"Unknown link '{link_name}'. Expected uplink or downlink.")
    return str(LINK_ALIASES[key])


def _normalize_method(method_name: str) -> str:
    key = _normalize_token(method_name)
    if key not in METHOD_ALIASES:
        raise ValueError(f"Unknown method '{method_name}'. Expected convergence or monte_carlo.")
    return str(METHOD_ALIASES[key])


def _normalize_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Could not interpret boolean value: {value!r}")


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return []
        return [part.strip() for part in text.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def _get_nested(mapping: dict[str, Any], dotted_path: str) -> Any:
    current: Any = mapping
    for key in dotted_path.split("."):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _set_nested(mapping: dict[str, Any], dotted_path: str, value: Any) -> None:
    keys = dotted_path.split(".")
    current = mapping
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def _discover_base_configs(link_name: str) -> list[str]:
    prefix = "uplink_" if link_name == "uplink" else "downlink_"
    names = sorted(
        path.name
        for path in EXPERIMENT_CONFIGS_DIR.glob(f"{prefix}*.yaml")
        if path.is_file()
    )
    if len(names) == 0:
        raise FileNotFoundError(f"No base configs found for link '{link_name}' under {EXPERIMENT_CONFIGS_DIR}.")
    return names


def _resolve_base_configs(raw_base_configs: Any, *, link_name: str) -> list[str]:
    cfg_names = _as_string_list(raw_base_configs)
    if len(cfg_names) == 0 or any(name.lower() in {"all", "*"} for name in cfg_names):
        return _discover_base_configs(link_name)
    return cfg_names


def _scenario_mode_from_cfg(cfg: dict[str, Any]) -> str:
    mode = _get_nested(cfg, "simulation.experiment_scenario.mode")
    if mode is None:
        return "payload_completion"
    return str(mode).strip().lower()


def _normalize_decision_value(path: str, value: Any) -> Any:
    if value is None:
        return None
    alias_map = DECISION_VALUE_ALIASES.get(path)
    if alias_map is None:
        return value
    key = str(value).strip().lower()
    if key not in alias_map:
        known = ", ".join(sorted(alias_map))
        raise ValueError(f"Unsupported value {value!r} for decision '{path}'. Expected one of: {known}")
    return alias_map[key]


def _value_tag_for_decision(path: str, value: Any) -> str:
    normalized = _normalize_decision_value(path, value)
    alias_map = DECISION_VALUE_TAGS.get(path, {})
    if str(normalized) in alias_map:
        return str(alias_map[str(normalized)])
    return _normalize_token(str(normalized))


def _decision_path_prefix(path: str) -> str:
    if path in DECISION_PATH_TAG_PREFIX:
        return str(DECISION_PATH_TAG_PREFIX[path])
    leaf = path.split(".")[-1]
    return _normalize_token(leaf)[:6] or "cfg"


def _decision_is_applicable(
    *,
    link_name: str,
    method_name: str,
    scenario_mode: str,
    chosen_values: dict[str, Any],
    path: str,
) -> bool:
    if path == "simulation.convergence_precoder_update_mode":
        return method_name == "convergence"

    if path == "simulation.uplink_rate_model":
        return link_name == "uplink"

    if path == "simulation.uplink_objective_mode":
        return link_name == "uplink"

    if path == "simulation.convergence_block_objective_mode":
        return link_name == "downlink"

    if path == "simulation.n_kl_reduction_update_scope":
        return link_name == "downlink" and method_name == "convergence"

    if path == "simulation.downlink_precoder_net_scope":
        if link_name != "downlink":
            return False
        if method_name == "monte_carlo":
            return True
        return str(chosen_values.get("simulation.convergence_precoder_update_mode", "precoder_net")) == "precoder_net"

    return True


def _expand_decision_combinations(
    *,
    link_name: str,
    method_name: str,
    scenario_mode: str,
    decision_grid: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    ordered_paths = list(decision_grid.keys())
    combinations: list[dict[str, Any]] = []

    def recurse(index: int, current: dict[str, Any]) -> None:
        if index >= len(ordered_paths):
            combinations.append(dict(current))
            return

        path = ordered_paths[index]
        if not _decision_is_applicable(
            link_name=link_name,
            method_name=method_name,
            scenario_mode=scenario_mode,
            chosen_values=current,
            path=path,
        ):
            recurse(index + 1, current)
            return

        for raw_value in decision_grid[path]:
            current[path] = _normalize_decision_value(path, raw_value)
            recurse(index + 1, current)
        current.pop(path, None)

    recurse(0, {})
    return combinations


def _build_decision_tag(decision_values: dict[str, Any]) -> str:
    parts: list[str] = []
    for path, value in decision_values.items():
        parts.append(f"{_decision_path_prefix(path)}-{_value_tag_for_decision(path, value)}")
    return "__".join(parts) if parts else "base"


def _effective_decision_value(base_cfg: dict[str, Any], decision_values: dict[str, Any], path: str) -> Any:
    if path in decision_values:
        return decision_values[path]
    return _normalize_decision_value(path, _get_nested(base_cfg, path))


def _is_base_configuration(
    *,
    base_cfg: dict[str, Any],
    link_name: str,
    method_name: str,
    scenario_mode: str,
    decision_grid: dict[str, list[Any]],
    decision_values: dict[str, Any],
) -> bool:
    for path in decision_grid.keys():
        if not _decision_is_applicable(
            link_name=link_name,
            method_name=method_name,
            scenario_mode=scenario_mode,
            chosen_values=decision_values,
            path=path,
        ):
            continue
        chosen = _effective_decision_value(base_cfg, decision_values, path)
        base_value = _normalize_decision_value(path, _get_nested(base_cfg, path))
        if chosen != base_value:
            return False
    return True


def _build_generated_cfg_name(
    *,
    link_name: str,
    method_name: str,
    base_cfg_name: str,
    sweep_id: str,
    decision_tag: str,
) -> str:
    link_short = "ul" if link_name == "uplink" else "dl"
    method_short = "conv" if method_name == "convergence" else "mc"
    base_short = compact_cfg_stem(base_cfg_name)
    return f"{link_short}_{base_short}__{sweep_id}__{method_short}__{decision_tag}.yaml"


def _resolve_monte_carlo_seed_args(defaults: dict[str, Any]) -> tuple[list[int], int | None, int]:
    explicit_train_seeds = parse_optional_seed_values(defaults.get("train_seeds"))
    raw_num_train_seeds = defaults.get("num_train_seeds")
    num_train_seeds = int(raw_num_train_seeds) if raw_num_train_seeds is not None else None
    test_seed = int(defaults.get("test_seed", 3))
    if len(explicit_train_seeds) > 0:
        return explicit_train_seeds, None, test_seed
    return [], num_train_seeds, test_seed


def _build_command(
    *,
    python_executable: str,
    link_name: str,
    method_name: str,
    generated_cfg_path: Path,
    defaults: dict[str, Any],
) -> list[str]:
    spec = METHOD_SPECS[(link_name, method_name)]
    cmd = [python_executable, str(spec["script_path"]), "--cfg_name", str(generated_cfg_path)]
    quiet = _normalize_bool(defaults.get("quiet", False), default=False)
    if method_name == "convergence":
        cmd.extend(["--seed", str(int(defaults.get("seed", 3)))])
    else:
        train_seeds, num_train_seeds, test_seed = _resolve_monte_carlo_seed_args(defaults)
        if len(train_seeds) > 0:
            cmd.extend(["--train_seeds", ",".join(str(int(seed)) for seed in train_seeds)])
        elif num_train_seeds is not None:
            cmd.extend(["--num_train_seeds", str(int(num_train_seeds))])
        cmd.extend(["--test_seed", str(int(test_seed))])
        if _normalize_bool(defaults.get("skip_test", False), default=False):
            cmd.append("--skip_test")
    if quiet and bool(spec.get("supports_quiet", False)):
        cmd.append("--quiet")
    return cmd


def _resolve_downlink_weight_strategy(value: Any) -> str:
    text = "" if value is None else str(value).strip().lower()
    if text in {"", "none"}:
        return "inverse_cnr"
    if text in {"uniform", "uniform_active_user_weight"}:
        return "uniform_active_user_weight"
    if text in {"backlog", "queue", "remaining_bits"}:
        return "remaining_bits"
    if text in {"inverse_channel_gain"}:
        return "inverse_channel_gain"
    return "inverse_cnr"


def _resolve_uplink_objective_mode(value: Any) -> str:
    return str(
        _normalize_decision_value("simulation.uplink_objective_mode", value) or "unweighted_sum_rate"
    )


def _resolve_downlink_objective_mode(value: Any, *, weight_strategy: Any = None) -> str:
    normalized = str(
        _normalize_decision_value("simulation.convergence_block_objective_mode", value) or "unweighted_sum_rate"
    )
    resolved_weight_strategy = _resolve_downlink_weight_strategy(weight_strategy)
    if normalized == "priority_weighted_sum_rate":
        if resolved_weight_strategy == "uniform_active_user_weight":
            return "uniform_weighted_sum_rate"
        if resolved_weight_strategy == "remaining_bits":
            return "remaining_bits_weighted_sum_rate"
        if resolved_weight_strategy == "inverse_channel_gain":
            return "inverse_channel_gain_weighted_sum_rate"
        return "inverse_cnr_weighted_sum_rate"
    if normalized == "blended_network_rate":
        if resolved_weight_strategy == "uniform_active_user_weight":
            return "blended_uniform_weighted_sum_rate"
        if resolved_weight_strategy == "remaining_bits":
            return "blended_remaining_bits_weighted_sum_rate"
        if resolved_weight_strategy == "inverse_channel_gain":
            return "blended_inverse_channel_gain_weighted_sum_rate"
        return "blended_inverse_cnr_weighted_sum_rate"
    return normalized


def _expected_result_json_path(
    *,
    link_name: str,
    method_name: str,
    generated_cfg_path: Path,
    generated_cfg: dict[str, Any],
    defaults: dict[str, Any],
) -> Path:
    cfg_name = str(generated_cfg_path)
    scenario_mode = _scenario_mode_from_cfg(generated_cfg)
    sim_cfg = generated_cfg.get("simulation", {})

    if link_name == "uplink" and method_name == "convergence":
        update_mode = str(sim_cfg.get("convergence_precoder_update_mode", "precoder_net")).strip().lower()
        objective_mode = _resolve_uplink_objective_mode(
            sim_cfg.get("uplink_objective_mode", "unweighted_sum_rate")
        )
        result_tag = make_method_result_tag(
            join_compact_tag_parts(
                compact_method_tag("convergence_per_epoch_baseline"),
                compact_objective_tag(objective_mode),
                compact_update_mode_tag(update_mode),
            ),
            cfg_name,
            seed=int(defaults.get("seed", 3)),
        )
        return Path(build_uplink_convergence_result_dirs("Convergence per epoch", result_tag)["data"]) / "result.json"

    if link_name == "uplink" and method_name == "monte_carlo":
        _, _, test_seed = _resolve_monte_carlo_seed_args(defaults)
        objective_mode = _resolve_uplink_objective_mode(
            sim_cfg.get("uplink_objective_mode", "unweighted_sum_rate")
        )
        result_tag = make_method_result_tag(
            join_compact_tag_parts(
                compact_method_tag("monte_carlo_precoder_net_train_test"),
                compact_objective_tag(objective_mode),
            ),
            cfg_name,
            seed=int(test_seed),
        )
        return Path(build_uplink_result_dirs("Monte Carlo", result_tag)["test_data"]) / "result.json"

    if link_name == "downlink" and method_name == "convergence":
        solver_mode = str(sim_cfg.get("convergence_precoder_update_mode", "precoder_net")).strip().lower()
        objective_mode = _resolve_downlink_objective_mode(
            sim_cfg.get("convergence_block_objective_mode"),
            weight_strategy=sim_cfg.get("convergence_priority_weight_strategy"),
        )
        model_scope = str(sim_cfg.get("downlink_precoder_net_scope", "per_user_nets")).strip().lower()
        method_parts = [compact_method_tag("convergence_per_epoch_baseline"), compact_objective_tag(objective_mode)]
        solver_tag = compact_update_mode_tag(solver_mode)
        if solver_tag != "dir":
            method_parts.append(compact_scope_tag(model_scope))
        method_parts.append(solver_tag)
        result_tag = make_method_result_tag(
            join_compact_tag_parts(*method_parts),
            cfg_name,
            seed=int(defaults.get("seed", 3)),
        )
        return Path(build_downlink_result_dirs("Convergence per epoch", result_tag)["test_data"]) / "result.json"

    if link_name == "downlink" and method_name == "monte_carlo":
        _, _, test_seed = _resolve_monte_carlo_seed_args(defaults)
        scope_name = str(sim_cfg.get("downlink_precoder_net_scope", "per_user_nets")).strip().lower()
        objective_mode = _resolve_downlink_objective_mode(
            sim_cfg.get("convergence_block_objective_mode"),
            weight_strategy=sim_cfg.get("convergence_priority_weight_strategy"),
        )
        result_tag = make_method_result_tag(
            join_compact_tag_parts(
                compact_method_tag("monte_carlo_precoder_net_train_test"),
                compact_objective_tag(objective_mode),
                compact_scope_tag(scope_name),
            ),
            cfg_name,
            seed=int(test_seed),
        )
        return Path(build_downlink_result_dirs("Monte Carlo", result_tag)["test_data"]) / "result.json"

    raise ValueError(f"Unsupported (link, method) combination: {(link_name, method_name)}")


def _sum_per_user_summary_field(summary_metrics: dict[str, Any], field: str) -> int:
    total = 0
    for row in summary_metrics.get("per_user_summary", []):
        if isinstance(row, dict):
            total += int(row.get(field, 0))
    return int(total)


def _format_train_seeds_for_row(result: dict[str, Any]) -> str:
    train_seeds = result.get("train_seeds", [])
    if isinstance(train_seeds, list):
        return ",".join(str(int(seed)) for seed in train_seeds)
    return ""


def _extract_comparison_row(
    *,
    run_spec: dict[str, Any],
    result_json_path: Path | None,
    log_path: Path,
    returncode: int,
    duration_seconds: float,
) -> dict[str, Any]:
    effective_decisions = dict(run_spec.get("effective_decisions", {}))
    row = {
        "status": "failed" if returncode != 0 else "missing_result_json",
        "returncode": int(returncode),
        "link": run_spec["link"],
        "method": run_spec["method"],
        "scenario_mode": run_spec["scenario_mode"],
        "base_config_name": run_spec["base_config_name"],
        "generated_config_name": run_spec["generated_cfg_path"].name,
        "decision_tag": run_spec["decision_tag"],
        "is_base_configuration": bool(run_spec["is_base_configuration"]),
        "seed": int(run_spec["seed"]),
        "train_seeds": run_spec["train_seed_text"],
        "highlighted_configuration": _highlighted_configuration_text(effective_decisions),
        "run_started_at_local": "",
        "run_completed_at_local": "",
        "duration_seconds": float(duration_seconds),
        "initial_total_latency": None,
        "final_total_latency": None,
        "delta_final_total_latency_vs_base": None,
        "total_latency_reduction_percent": None,
        "delta_total_latency_reduction_percent_vs_base": None,
        "initial_asynchronality_sum": None,
        "final_asynchronality_sum": None,
        "delta_final_asynchronality_sum_vs_base": None,
        "asynchronality_reduction_percent": None,
        "delta_asynchronality_reduction_percent_vs_base": None,
        "initial_avg_snr_db": None,
        "final_avg_snr_db": None,
        "delta_final_avg_snr_db_vs_base": None,
        "initial_avg_sinr_db": None,
        "final_avg_sinr_db": None,
        "delta_final_avg_sinr_db_vs_base": None,
        "total_served_bits": None,
        "delta_total_served_bits_vs_base": None,
        "total_skipped_blocks": None,
        "delta_total_skipped_blocks_vs_base": None,
        "total_target_bits": None,
        "total_unserved_bits": None,
        "training_solve_status": "",
        "epochs_completed": None,
        "last_epoch_feasible_rollout_queries": None,
        "last_epoch_total_rollout_queries": None,
        "last_epoch_feasible_rollout_fraction": None,
        "result_json_path": str(result_json_path) if result_json_path is not None else "",
        "log_path": str(log_path),
        "effective_decisions_json": json.dumps(make_serializable(effective_decisions), sort_keys=True),
        "_effective_decisions": effective_decisions,
    }

    for path, value in _sorted_effective_decisions(effective_decisions):
        row[f"decision__{path.replace('.', '__')}"] = str(value)

    if result_json_path is None or not result_json_path.exists():
        return row

    with result_json_path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)

    metrics = result.get("summary_metrics", {})
    post_training_summary = result.get("post_training_summary", {})
    row.update(
        {
            "status": "completed",
            "seed": int(result.get("seed", run_spec["seed"])),
            "train_seeds": _format_train_seeds_for_row(result) or run_spec["train_seed_text"],
            "run_started_at_local": str(result.get("run_started_at_local", "")),
            "run_completed_at_local": str(result.get("run_completed_at_local", "")),
            "scenario_mode": str(
                result.get("scenario_mode", result.get("experiment_scenario_mode", run_spec["scenario_mode"]))
            ),
            "initial_total_latency": metrics.get("initial_total_latency"),
            "final_total_latency": metrics.get("final_total_latency"),
            "total_latency_reduction_percent": metrics.get("total_latency_reduction_percent"),
            "initial_asynchronality_sum": metrics.get("initial_asynchronality_sum"),
            "final_asynchronality_sum": metrics.get("final_asynchronality_sum"),
            "asynchronality_reduction_percent": metrics.get("asynchronality_reduction_percent"),
            "initial_avg_snr_db": metrics.get("initial_avg_snr_db"),
            "final_avg_snr_db": metrics.get("final_avg_snr_db"),
            "initial_avg_sinr_db": metrics.get("initial_avg_sinr_db"),
            "final_avg_sinr_db": metrics.get("final_avg_sinr_db"),
            "total_served_bits": _sum_per_user_summary_field(metrics, "served_bits"),
            "total_skipped_blocks": _sum_per_user_summary_field(metrics, "skipped_blocks"),
            "total_target_bits": _sum_per_user_summary_field(metrics, "target_bits"),
            "total_unserved_bits": _sum_per_user_summary_field(metrics, "unserved_bits"),
            "training_solve_status": str(post_training_summary.get("training_solve_status", "")),
            "epochs_completed": post_training_summary.get("epochs_completed"),
            "last_epoch_feasible_rollout_queries": post_training_summary.get("last_epoch_feasible_rollout_queries"),
            "last_epoch_total_rollout_queries": post_training_summary.get("last_epoch_total_rollout_queries"),
            "last_epoch_feasible_rollout_fraction": post_training_summary.get(
                "final_feasible_rollout_query_fraction"
            ),
        }
    )
    return row


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _best_row_by_metric(
    rows: list[dict[str, Any]],
    metric_key: str,
    *,
    prefer: str,
) -> dict[str, Any] | None:
    usable_rows = [row for row in rows if _float_or_none(row.get(metric_key)) is not None]
    if len(usable_rows) == 0:
        return None
    if prefer == "min":
        return min(usable_rows, key=lambda row: float(row[metric_key]))
    return max(usable_rows, key=lambda row: float(row[metric_key]))


def _apply_base_deltas(rows: list[dict[str, Any]]) -> None:
    metric_pairs = [
        ("final_total_latency", "delta_final_total_latency_vs_base"),
        ("total_latency_reduction_percent", "delta_total_latency_reduction_percent_vs_base"),
        ("final_asynchronality_sum", "delta_final_asynchronality_sum_vs_base"),
        ("asynchronality_reduction_percent", "delta_asynchronality_reduction_percent_vs_base"),
        ("final_avg_snr_db", "delta_final_avg_snr_db_vs_base"),
        ("final_avg_sinr_db", "delta_final_avg_sinr_db_vs_base"),
        ("total_served_bits", "delta_total_served_bits_vs_base"),
        ("total_skipped_blocks", "delta_total_skipped_blocks_vs_base"),
    ]

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["link"]), str(row["method"]), str(row["base_config_name"]))].append(row)

    for group_rows in grouped.values():
        base_rows = [row for row in group_rows if bool(row["is_base_configuration"]) and row["status"] == "completed"]
        if len(base_rows) != 1:
            continue
        base_row = base_rows[0]
        for row in group_rows:
            for value_key, delta_key in metric_pairs:
                base_value = _float_or_none(base_row.get(value_key))
                current_value = _float_or_none(row.get(value_key))
                if base_value is None or current_value is None:
                    row[delta_key] = None
                else:
                    row[delta_key] = float(current_value - base_value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(CSV_FIELD_ORDER)
    extra_fields = sorted(
        {
            key
            for row in rows
            for key in row.keys()
            if key not in fieldnames and not str(key).startswith("_")
        }
    )
    fieldnames.extend(extra_fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _metric_text(value: Any, *, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, (int,)):
        return str(int(value))
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _build_markdown_report(
    *,
    run_root: Path,
    manifest_path: Path,
    rows: list[dict[str, Any]],
    generated_at_local: str,
) -> list[str]:
    lines = [
        "# Decision Sweep Comparison",
        "",
        f"- Generated at: `{generated_at_local}`",
        f"- Manifest: `{manifest_path}`",
        f"- Run root: `{run_root}`",
        f"- Total planned runs: `{len(rows)}`",
        f"- Completed runs: `{sum(1 for row in rows if row['status'] == 'completed')}`",
        f"- Failed or missing runs: `{sum(1 for row in rows if row['status'] != 'completed')}`",
    ]

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["link"]), str(row["method"]), str(row["base_config_name"]))].append(row)

    for group_key in sorted(grouped):
        link_name, method_name, base_config_name = group_key
        group_rows = sorted(grouped[group_key], key=_report_sort_key)
        base_row = next(
            (row for row in group_rows if bool(row["is_base_configuration"]) and row["status"] == "completed"),
            None,
        )
        decision_axes = sorted(
            {
                path
                for row in group_rows
                for path in dict(row.get("_effective_decisions", {})).keys()
            }
        )
        lines.extend(
            [
                "",
                f"## {link_name} | {method_name} | {base_config_name}",
                "",
                f"- Scenario: `{group_rows[0]['scenario_mode']}`",
                f"- Seed: `{group_rows[0]['seed']}`",
                f"- Train seeds: `{group_rows[0]['train_seeds'] or 'n/a'}`",
                f"- Decision axes: `{', '.join(_decision_label(path) for path in decision_axes) if decision_axes else 'none'}`",
            ]
        )
        if base_row is not None:
            lines.append(f"- Base configuration: `{_normalize_report_text(base_row['highlighted_configuration'])}`")
        lines.extend(
            [
                "",
                "| Variant | Base | Status | Highlighted configuration | Final total latency | Delta vs base | Latency reduction % | Delta vs base | Final asynchronality | Delta vs base | Asynchronality reduction % | Delta vs base | Final avg SINR (dB) | Served bits | Log |",
                "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in group_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _normalize_report_text(row["decision_tag"]),
                        "yes" if bool(row["is_base_configuration"]) else "",
                        str(row["status"]),
                        _normalize_report_text(row["highlighted_configuration"]),
                        _metric_text(row.get("final_total_latency"), digits=6),
                        _metric_text(row.get("delta_final_total_latency_vs_base"), digits=6),
                        _metric_text(row.get("total_latency_reduction_percent"), digits=4),
                        _metric_text(row.get("delta_total_latency_reduction_percent_vs_base"), digits=4),
                        _metric_text(row.get("final_asynchronality_sum"), digits=6),
                        _metric_text(row.get("delta_final_asynchronality_sum_vs_base"), digits=6),
                        _metric_text(row.get("asynchronality_reduction_percent"), digits=4),
                        _metric_text(row.get("delta_asynchronality_reduction_percent_vs_base"), digits=4),
                        _metric_text(row.get("final_avg_sinr_db"), digits=4),
                        _metric_text(row.get("total_served_bits"), digits=0),
                        f"`{Path(str(row['log_path'])).name}`",
                    ]
                )
                + " |"
            )

        completed_rows = [row for row in group_rows if row["status"] == "completed"]
        if len(completed_rows) > 0:
            best_latency = _best_row_by_metric(completed_rows, "final_total_latency", prefer="min")
            best_reduction = _best_row_by_metric(completed_rows, "total_latency_reduction_percent", prefer="max")
            best_async = _best_row_by_metric(completed_rows, "final_asynchronality_sum", prefer="min")
            best_async_reduction = _best_row_by_metric(
                completed_rows,
                "asynchronality_reduction_percent",
                prefer="max",
            )
            lines.extend(
                [
                    "",
                ]
            )
            if best_latency is not None:
                lines.extend(
                    [
                        f"- Best final total latency: `{_normalize_report_text(best_latency['decision_tag'])}` -> `{_metric_text(best_latency.get('final_total_latency'), digits=6)}`",
                        f"- Best-latency configuration: `{_normalize_report_text(best_latency['highlighted_configuration'])}`",
                    ]
                )
            if best_reduction is not None:
                lines.extend(
                    [
                        f"- Best latency reduction: `{_normalize_report_text(best_reduction['decision_tag'])}` -> `{_metric_text(best_reduction.get('total_latency_reduction_percent'), digits=4)}%`",
                        f"- Best-reduction configuration: `{_normalize_report_text(best_reduction['highlighted_configuration'])}`",
                    ]
                )
            if best_async is not None:
                lines.extend(
                    [
                        f"- Best final asynchronality: `{_normalize_report_text(best_async['decision_tag'])}` -> `{_metric_text(best_async.get('final_asynchronality_sum'), digits=6)}`",
                        f"- Best-asynchronality configuration: `{_normalize_report_text(best_async['highlighted_configuration'])}`",
                    ]
                )
            if best_async_reduction is not None:
                lines.extend(
                    [
                        f"- Best asynchronality reduction: `{_normalize_report_text(best_async_reduction['decision_tag'])}` -> `{_metric_text(best_async_reduction.get('asynchronality_reduction_percent'), digits=4)}%`",
                        f"- Best asynchronality-reduction configuration: `{_normalize_report_text(best_async_reduction['highlighted_configuration'])}`",
                    ]
                )

    return lines


def _strip_internal_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not str(key).startswith("_")}


def _completed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("status") == "completed"]


def _write_json_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    save_json([_strip_internal_fields(row) for row in rows], str(path))


def _best_rows_by_group(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["link"]), str(row["method"]), str(row["base_config_name"]))].append(row)

    best_rows: list[dict[str, Any]] = []
    for group_key in sorted(grouped):
        completed = _completed_rows(grouped[group_key])
        if len(completed) == 0:
            continue
        best_latency = _best_row_by_metric(completed, "final_total_latency", prefer="min")
        best_reduction = _best_row_by_metric(completed, "total_latency_reduction_percent", prefer="max")
        if best_latency is None or best_reduction is None:
            continue
        best_rows.append(
            {
                "selection_type": "best_final_total_latency",
                "link": group_key[0],
                "method": group_key[1],
                "base_config_name": group_key[2],
                "decision_tag": best_latency["decision_tag"],
                "highlighted_configuration": best_latency["highlighted_configuration"],
                "final_total_latency": best_latency.get("final_total_latency"),
                "total_latency_reduction_percent": best_latency.get("total_latency_reduction_percent"),
                "final_asynchronality_sum": best_latency.get("final_asynchronality_sum"),
                "asynchronality_reduction_percent": best_latency.get("asynchronality_reduction_percent"),
                "final_avg_sinr_db": best_latency.get("final_avg_sinr_db"),
                "total_served_bits": best_latency.get("total_served_bits"),
                "result_json_path": best_latency.get("result_json_path"),
                "log_path": best_latency.get("log_path"),
            }
        )
        best_rows.append(
            {
                "selection_type": "best_latency_reduction_percent",
                "link": group_key[0],
                "method": group_key[1],
                "base_config_name": group_key[2],
                "decision_tag": best_reduction["decision_tag"],
                "highlighted_configuration": best_reduction["highlighted_configuration"],
                "final_total_latency": best_reduction.get("final_total_latency"),
                "total_latency_reduction_percent": best_reduction.get("total_latency_reduction_percent"),
                "final_asynchronality_sum": best_reduction.get("final_asynchronality_sum"),
                "asynchronality_reduction_percent": best_reduction.get("asynchronality_reduction_percent"),
                "final_avg_sinr_db": best_reduction.get("final_avg_sinr_db"),
                "total_served_bits": best_reduction.get("total_served_bits"),
                "result_json_path": best_reduction.get("result_json_path"),
                "log_path": best_reduction.get("log_path"),
            }
        )
    return best_rows


def _decision_value_best_rows(rows: list[dict[str, Any]], decision_path: str) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    value_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = dict(row.get("_effective_decisions", {})).get(decision_path)
        if value is None:
            continue
        value_rows[str(value)].append(row)

    summary_rows: list[dict[str, Any]] = []
    for value, decision_rows in sorted(value_rows.items()):
        completed = _completed_rows(decision_rows)
        if len(completed) == 0:
            continue
        best_latency = _best_row_by_metric(completed, "final_total_latency", prefer="min")
        best_reduction = _best_row_by_metric(completed, "total_latency_reduction_percent", prefer="max")
        if best_latency is None or best_reduction is None:
            continue
        summary_rows.append(
            {
                "decision_path": decision_path,
                "decision_label": _decision_label(decision_path),
                "decision_value": value,
                "selection_type": "best_final_total_latency",
                "decision_tag": best_latency["decision_tag"],
                "highlighted_configuration": best_latency["highlighted_configuration"],
                "link": best_latency["link"],
                "method": best_latency["method"],
                "base_config_name": best_latency["base_config_name"],
                "final_total_latency": best_latency.get("final_total_latency"),
                "total_latency_reduction_percent": best_latency.get("total_latency_reduction_percent"),
                "final_asynchronality_sum": best_latency.get("final_asynchronality_sum"),
                "asynchronality_reduction_percent": best_latency.get("asynchronality_reduction_percent"),
                "final_avg_sinr_db": best_latency.get("final_avg_sinr_db"),
                "total_served_bits": best_latency.get("total_served_bits"),
                "result_json_path": best_latency.get("result_json_path"),
                "log_path": best_latency.get("log_path"),
            }
        )
        summary_rows.append(
            {
                "decision_path": decision_path,
                "decision_label": _decision_label(decision_path),
                "decision_value": value,
                "selection_type": "best_latency_reduction_percent",
                "decision_tag": best_reduction["decision_tag"],
                "highlighted_configuration": best_reduction["highlighted_configuration"],
                "link": best_reduction["link"],
                "method": best_reduction["method"],
                "base_config_name": best_reduction["base_config_name"],
                "final_total_latency": best_reduction.get("final_total_latency"),
                "total_latency_reduction_percent": best_reduction.get("total_latency_reduction_percent"),
                "final_asynchronality_sum": best_reduction.get("final_asynchronality_sum"),
                "asynchronality_reduction_percent": best_reduction.get("asynchronality_reduction_percent"),
                "final_avg_sinr_db": best_reduction.get("final_avg_sinr_db"),
                "total_served_bits": best_reduction.get("total_served_bits"),
                "result_json_path": best_reduction.get("result_json_path"),
                "log_path": best_reduction.get("log_path"),
            }
        )
    return summary_rows, value_rows


def _build_decision_value_report(
    *,
    decision_path: str,
    decision_value: str,
    rows: list[dict[str, Any]],
) -> list[str]:
    completed = _completed_rows(rows)
    sorted_rows = sorted(rows, key=_report_sort_key)
    lines = [
        f"# Decision Value Comparison: {_decision_label(decision_path)} = {_normalize_report_text(decision_value)}",
        "",
        f"- Decision path: `{decision_path}`",
        f"- Completed runs: `{len(completed)}` / `{len(rows)}`",
        "",
        "| Link | Method | Base config | Variant | Final total latency | Latency reduction % | Final asynchronality | Asynchronality reduction % | Final avg SINR (dB) | Served bits | Highlighted configuration |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in sorted_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["link"]),
                    str(row["method"]),
                    str(row["base_config_name"]),
                    _normalize_report_text(row["decision_tag"]),
                    _metric_text(row.get("final_total_latency"), digits=6),
                    _metric_text(row.get("total_latency_reduction_percent"), digits=4),
                    _metric_text(row.get("final_asynchronality_sum"), digits=6),
                    _metric_text(row.get("asynchronality_reduction_percent"), digits=4),
                    _metric_text(row.get("final_avg_sinr_db"), digits=4),
                    _metric_text(row.get("total_served_bits"), digits=0),
                    _normalize_report_text(row["highlighted_configuration"]),
                ]
            )
            + " |"
        )
    if len(completed) > 0:
        best_latency = _best_row_by_metric(completed, "final_total_latency", prefer="min")
        best_reduction = _best_row_by_metric(completed, "total_latency_reduction_percent", prefer="max")
        best_async = _best_row_by_metric(completed, "final_asynchronality_sum", prefer="min")
        best_async_reduction = _best_row_by_metric(
            completed,
            "asynchronality_reduction_percent",
            prefer="max",
        )
        lines.extend(
            [
                "",
            ]
        )
        if best_latency is not None:
            lines.extend(
                [
                    f"- Best final total latency: `{_normalize_report_text(best_latency['decision_tag'])}` -> `{_metric_text(best_latency.get('final_total_latency'), digits=6)}`",
                    f"- Best latency configuration: `{_normalize_report_text(best_latency['highlighted_configuration'])}`",
                ]
            )
        if best_reduction is not None:
            lines.extend(
                [
                    f"- Best latency reduction: `{_normalize_report_text(best_reduction['decision_tag'])}` -> `{_metric_text(best_reduction.get('total_latency_reduction_percent'), digits=4)}%`",
                    f"- Best reduction configuration: `{_normalize_report_text(best_reduction['highlighted_configuration'])}`",
                ]
            )
        if best_async is not None:
            lines.extend(
                [
                    f"- Best final asynchronality: `{_normalize_report_text(best_async['decision_tag'])}` -> `{_metric_text(best_async.get('final_asynchronality_sum'), digits=6)}`",
                    f"- Best asynchronality configuration: `{_normalize_report_text(best_async['highlighted_configuration'])}`",
                ]
            )
        if best_async_reduction is not None:
            lines.extend(
                [
                    f"- Best asynchronality reduction: `{_normalize_report_text(best_async_reduction['decision_tag'])}` -> `{_metric_text(best_async_reduction.get('asynchronality_reduction_percent'), digits=4)}%`",
                    f"- Best asynchronality-reduction configuration: `{_normalize_report_text(best_async_reduction['highlighted_configuration'])}`",
                ]
            )
    return lines


def _build_decision_report(
    *,
    decision_path: str,
    rows: list[dict[str, Any]],
) -> list[str]:
    summary_rows, value_rows = _decision_value_best_rows(rows, decision_path)
    lines = [
        f"# Decision Comparison: {_decision_label(decision_path)}",
        "",
        f"- Decision path: `{decision_path}`",
        f"- Total runs touching this decision: `{len(rows)}`",
        f"- Completed runs: `{len(_completed_rows(rows))}`",
        "",
        "| Decision value | Best criterion | Link | Method | Base config | Variant | Final total latency | Latency reduction % | Final asynchronality | Asynchronality reduction % | Highlighted configuration |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _normalize_report_text(row["decision_value"]),
                    str(row["selection_type"]),
                    str(row["link"]),
                    str(row["method"]),
                    str(row["base_config_name"]),
                    _normalize_report_text(row["decision_tag"]),
                    _metric_text(row.get("final_total_latency"), digits=6),
                    _metric_text(row.get("total_latency_reduction_percent"), digits=4),
                    _metric_text(row.get("final_asynchronality_sum"), digits=6),
                    _metric_text(row.get("asynchronality_reduction_percent"), digits=4),
                    _normalize_report_text(row["highlighted_configuration"]),
                ]
            )
            + " |"
        )
    if len(summary_rows) == 0:
        lines.extend(["", "No completed runs were available for this decision."])
    else:
        lines.extend(["", "## Value folders", ""])
        for value in sorted(value_rows):
            lines.append(f"- `{_normalize_report_text(value)}` -> `values/{_safe_path_token(value)}`")
    return lines


def _write_detailed_comparison_results(
    *,
    comparison_root: Path,
    manifest_path: Path,
    run_root: Path,
    rows: list[dict[str, Any]],
    generated_at_local: str,
) -> None:
    comparison_root.mkdir(parents=True, exist_ok=True)
    overall_root = comparison_root / "overall"
    by_decision_root = comparison_root / "by_decision"
    overall_root.mkdir(parents=True, exist_ok=True)
    by_decision_root.mkdir(parents=True, exist_ok=True)

    _write_csv(overall_root / "comparison_summary.csv", rows)
    _write_json_rows(overall_root / "comparison_summary.json", rows)
    save_text(
        _build_markdown_report(
            run_root=run_root,
            manifest_path=manifest_path,
            rows=rows,
            generated_at_local=generated_at_local,
        ),
        str(overall_root / "comparison_report.md"),
    )

    best_rows = _best_rows_by_group(rows)
    _write_csv(overall_root / "best_configurations_by_group.csv", best_rows)
    save_json(make_serializable(best_rows), str(overall_root / "best_configurations_by_group.json"))

    top_completed_rows = sorted(_completed_rows(rows), key=_report_sort_key)
    _write_csv(overall_root / "best_completed_runs.csv", top_completed_rows)
    _write_json_rows(overall_root / "best_completed_runs.json", top_completed_rows)

    decision_paths = sorted(
        {
            path
            for row in rows
            for path in dict(row.get("_effective_decisions", {})).keys()
        }
    )
    for decision_path in decision_paths:
        decision_rows = [row for row in rows if decision_path in dict(row.get("_effective_decisions", {}))]
        decision_root = by_decision_root / _safe_path_token(decision_path)
        decision_root.mkdir(parents=True, exist_ok=True)

        _write_csv(decision_root / "all_rows.csv", decision_rows)
        _write_json_rows(decision_root / "all_rows.json", decision_rows)

        summary_rows, value_rows = _decision_value_best_rows(decision_rows, decision_path)
        _write_csv(decision_root / "best_by_value.csv", summary_rows)
        save_json(make_serializable(summary_rows), str(decision_root / "best_by_value.json"))
        save_text(
            _build_decision_report(decision_path=decision_path, rows=decision_rows),
            str(decision_root / "decision_report.md"),
        )

        values_root = decision_root / "values"
        values_root.mkdir(parents=True, exist_ok=True)
        for decision_value, rows_for_value in sorted(value_rows.items()):
            value_root = values_root / _safe_path_token(decision_value)
            value_root.mkdir(parents=True, exist_ok=True)
            sorted_value_rows = sorted(rows_for_value, key=_report_sort_key)
            _write_csv(value_root / "comparison_summary.csv", sorted_value_rows)
            _write_json_rows(value_root / "comparison_summary.json", sorted_value_rows)
            save_text(
                _build_decision_value_report(
                    decision_path=decision_path,
                    decision_value=decision_value,
                    rows=sorted_value_rows,
                ),
                str(value_root / "comparison_report.md"),
            )


def _discover_generated_runs(
    *,
    manifest_payload: dict[str, Any],
    manifest_path: Path,
    links_filter: set[str],
    methods_filter: set[str],
    configs_filter: set[str],
    run_root: Path,
    python_executable: str,
) -> list[dict[str, Any]]:
    defaults = dict(manifest_payload.get("defaults", {}))
    base_configs_cfg = dict(manifest_payload.get("base_configs", {}))
    decision_grids_cfg = dict(manifest_payload.get("decision_grids", {}))
    sweep_id = str(run_root.name)

    generated_root = run_root / "generated_configs"
    logs_root = run_root / "logs"
    generated_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    run_specs: list[dict[str, Any]] = []

    for raw_link_name, raw_method_grids in decision_grids_cfg.items():
        link_name = _normalize_link(raw_link_name)
        if len(links_filter) > 0 and link_name not in links_filter:
            continue
        if not isinstance(raw_method_grids, dict):
            raise ValueError(f"decision_grids.{raw_link_name} must be a mapping of methods.")

        base_config_names = _resolve_base_configs(base_configs_cfg.get(link_name, "all"), link_name=link_name)
        for base_config_name in base_config_names:
            if len(configs_filter) > 0:
                stem = Path(base_config_name).stem
                if base_config_name not in configs_filter and stem not in configs_filter:
                    continue
            base_cfg_path = _resolve_manifest_path(base_config_name)
            base_cfg = _read_yaml(base_cfg_path)
            scenario_mode = _scenario_mode_from_cfg(base_cfg)

            for raw_method_name, raw_decision_grid in raw_method_grids.items():
                method_name = _normalize_method(raw_method_name)
                if len(methods_filter) > 0 and method_name not in methods_filter:
                    continue
                if not isinstance(raw_decision_grid, dict):
                    raise ValueError(
                        f"decision_grids.{raw_link_name}.{raw_method_name} must be a mapping of dotted paths to values."
                    )

                normalized_grid: dict[str, list[Any]] = {}
                for path, values in raw_decision_grid.items():
                    value_list = list(values) if isinstance(values, (list, tuple)) else [values]
                    normalized_grid[str(path)] = [_normalize_decision_value(str(path), value) for value in value_list]

                decision_combinations = _expand_decision_combinations(
                    link_name=link_name,
                    method_name=method_name,
                    scenario_mode=scenario_mode,
                    decision_grid=normalized_grid,
                )

                for combo_index, decision_values in enumerate(decision_combinations):
                    generated_cfg = copy.deepcopy(base_cfg)
                    for path, value in decision_values.items():
                        _set_nested(generated_cfg, path, value)

                    effective_decisions: dict[str, Any] = {}
                    for path in normalized_grid.keys():
                        if not _decision_is_applicable(
                            link_name=link_name,
                            method_name=method_name,
                            scenario_mode=scenario_mode,
                            chosen_values=decision_values,
                            path=path,
                        ):
                            continue
                        effective_decisions[path] = _effective_decision_value(base_cfg, decision_values, path)

                    decision_tag = _build_decision_tag(decision_values)
                    generated_cfg_name = _build_generated_cfg_name(
                        link_name=link_name,
                        method_name=method_name,
                        base_cfg_name=base_config_name,
                        sweep_id=sweep_id,
                        decision_tag=decision_tag,
                    )
                    generated_cfg_path = generated_root / link_name / method_name / generated_cfg_name
                    _write_yaml(generated_cfg_path, generated_cfg)

                    train_seeds, num_train_seeds, test_seed = _resolve_monte_carlo_seed_args(defaults)
                    train_seed_text = ",".join(str(int(seed)) for seed in train_seeds)
                    if len(train_seed_text) == 0 and num_train_seeds is not None:
                        resolved_train_seeds = build_train_seeds_from_num_train_seeds(int(num_train_seeds), int(test_seed))
                        train_seed_text = ",".join(str(int(seed)) for seed in resolved_train_seeds)

                    command = _build_command(
                        python_executable=python_executable,
                        link_name=link_name,
                        method_name=method_name,
                        generated_cfg_path=generated_cfg_path,
                        defaults=defaults,
                    )

                    result_json_path = _expected_result_json_path(
                        link_name=link_name,
                        method_name=method_name,
                        generated_cfg_path=generated_cfg_path,
                        generated_cfg=generated_cfg,
                        defaults=defaults,
                    )
                    is_base = _is_base_configuration(
                        base_cfg=base_cfg,
                        link_name=link_name,
                        method_name=method_name,
                        scenario_mode=scenario_mode,
                        decision_grid=normalized_grid,
                        decision_values=decision_values,
                    )

                    run_specs.append(
                        {
                            "id": f"{link_name}_{method_name}_{Path(base_config_name).stem}_{combo_index:03d}",
                            "display_name": METHOD_SPECS[(link_name, method_name)]["display_name"],
                            "link": link_name,
                            "method": method_name,
                            "base_config_name": base_config_name,
                            "base_cfg_path": base_cfg_path,
                            "generated_cfg_path": generated_cfg_path,
                            "scenario_mode": scenario_mode,
                            "decision_values": dict(decision_values),
                            "effective_decisions": effective_decisions,
                            "decision_tag": decision_tag,
                            "is_base_configuration": bool(is_base),
                            "seed": int(defaults.get("seed", 3)) if method_name == "convergence" else int(test_seed),
                            "train_seed_text": train_seed_text,
                            "command": command,
                            "result_json_path": result_json_path,
                            "log_path": logs_root / f"{generated_cfg_path.stem}.log",
                        }
                    )

    serializable_runs = []
    for run_spec in run_specs:
        serializable_run = dict(run_spec)
        for path_key in ("base_cfg_path", "generated_cfg_path", "result_json_path", "log_path"):
            if path_key in serializable_run:
                serializable_run[path_key] = str(serializable_run[path_key])
        serializable_runs.append(serializable_run)

    plan_payload = {
        "generated_at_local": current_local_timestamp(),
        "manifest_path": str(manifest_path),
        "run_root": str(run_root),
        "defaults": defaults,
        "runs": serializable_runs,
    }
    save_json(plan_payload, str(run_root / "sweep_plan.json"))
    return run_specs


def _run_specs(
    *,
    run_specs: list[dict[str, Any]],
    max_parallel: int,
    continue_on_error: bool,
) -> list[dict[str, Any]]:
    execution_rows: list[dict[str, Any]] = []
    pending = list(run_specs)
    active: list[dict[str, Any]] = []
    stop_launching = False

    while pending or active:
        while not stop_launching and pending and len(active) < max_parallel:
            run_spec = pending.pop(0)
            log_path = Path(run_spec["log_path"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("w", encoding="utf-8")
            print("")
            print("=" * 100)
            print(f"Launching {run_spec['display_name']} | cfg={run_spec['generated_cfg_path'].name}")
            print(" ".join(str(part) for part in run_spec["command"]))
            print("=" * 100)
            process = subprocess.Popen(
                run_spec["command"],
                cwd=str(PROJECT_ROOT),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            active.append(
                {
                    "run_spec": run_spec,
                    "process": process,
                    "log_handle": log_handle,
                    "started_monotonic": time.perf_counter(),
                }
            )

        if len(active) == 0:
            break

        time.sleep(0.2)
        for item in list(active):
            process = item["process"]
            returncode = process.poll()
            if returncode is None:
                continue

            item["log_handle"].close()
            duration_seconds = time.perf_counter() - float(item["started_monotonic"])
            run_spec = item["run_spec"]
            result_json_path = Path(run_spec["result_json_path"])
            row = _extract_comparison_row(
                run_spec=run_spec,
                result_json_path=result_json_path if result_json_path.exists() else None,
                log_path=Path(run_spec["log_path"]),
                returncode=int(returncode),
                duration_seconds=float(duration_seconds),
            )
            execution_rows.append(row)
            active.remove(item)

            if returncode != 0 and not continue_on_error:
                stop_launching = True

    execution_rows.sort(
        key=lambda row: (
            str(row["link"]),
            str(row["method"]),
            str(row["base_config_name"]),
            str(row["decision_tag"]),
        )
    )
    return execution_rows


def _write_dry_run_report(run_root: Path, comparison_root: Path, run_specs: list[dict[str, Any]]) -> None:
    lines = [
        "Decision sweep dry run",
        f"Generated at: {current_local_timestamp()}",
        f"Control root: {run_root}",
        f"Detailed comparison root: {comparison_root}",
        f"Planned runs: {len(run_specs)}",
        "",
    ]
    for index, run_spec in enumerate(run_specs, start=1):
        lines.extend(
            [
                f"{index:03d}. {run_spec['display_name']}",
                f"  base_config: {run_spec['base_config_name']}",
                f"  generated_config: {run_spec['generated_cfg_path']}",
                f"  decision_tag: {run_spec['decision_tag']}",
                f"  is_base_configuration: {run_spec['is_base_configuration']}",
                f"  command: {' '.join(str(part) for part in run_spec['command'])}",
                f"  expected_result_json: {run_spec['result_json_path']}",
                f"  log_path: {run_spec['log_path']}",
                "",
            ]
        )
    save_text(lines, str(run_root / "dry_run_plan.txt"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate decision-combination sweeps for uplink/downlink configs, run them, and build comparison tables."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=DEFAULT_MANIFEST_NAME,
        help="Decision-sweep manifest YAML name or path. Defaults to Experiment Configs/decision_sweep.yaml",
    )
    parser.add_argument("--links", type=str, default=None, help="Optional filter, e.g. uplink or uplink,downlink")
    parser.add_argument("--methods", type=str, default=None, help="Optional filter, e.g. convergence or monte_carlo")
    parser.add_argument("--configs", type=str, default=None, help="Optional filter by base cfg filename or stem, comma-separated")
    parser.add_argument("--python_exe", type=str, default=None, help="Override Python executable used for child runs.")
    parser.add_argument(
        "--max_parallel",
        type=str,
        default=None,
        help="Maximum concurrent child runs. Use an integer, 'auto', or 'all'. Defaults to manifest/default 'auto'.",
    )
    parser.add_argument("--output_root", type=str, default=None, help="Optional control-output root for generated configs, logs, and run plans.")
    parser.add_argument(
        "--comparison_root",
        type=str,
        default=None,
        help="Optional detailed-comparison results root. Defaults to Results/Decision Sweep Comparisons/<run_id>",
    )
    parser.add_argument("--dry_run", action="store_true", help="Generate configs and the run plan without executing experiments.")
    parser.add_argument("--continue_on_error", action="store_true", help="Keep launching remaining runs after a failure.")
    args = parser.parse_args()

    manifest_path = _resolve_manifest_path(args.manifest)
    manifest_payload = _read_yaml(manifest_path)
    defaults = dict(manifest_payload.get("defaults", {}))
    links_filter = {_normalize_link(value) for value in _as_string_list(args.links)}
    methods_filter = {_normalize_method(value) for value in _as_string_list(args.methods)}
    configs_filter = {value.strip() for value in _as_string_list(args.configs)}

    continue_on_error = bool(args.continue_on_error or _normalize_bool(defaults.get("continue_on_error", True), default=True))
    python_executable = str(args.python_exe or defaults.get("python_executable", sys.executable))

    run_id = f"sweep_{time.strftime('%Y%m%d_%H%M%S')}"
    run_root = Path(args.output_root) if args.output_root else (DECISION_SWEEP_RUNS_ROOT / run_id)
    comparison_root = (
        Path(args.comparison_root)
        if args.comparison_root
        else (DECISION_COMPARISON_RESULTS_ROOT / run_id)
    )
    run_root.mkdir(parents=True, exist_ok=True)
    comparison_root.mkdir(parents=True, exist_ok=True)

    run_specs = _discover_generated_runs(
        manifest_payload=manifest_payload,
        manifest_path=manifest_path,
        links_filter=links_filter,
        methods_filter=methods_filter,
        configs_filter=configs_filter,
        run_root=run_root,
        python_executable=python_executable,
    )

    if len(run_specs) == 0:
        print("No decision-sweep runs matched the manifest and filters.")
        return 1

    requested_parallelism = args.max_parallel if args.max_parallel is not None else defaults.get("max_parallel", "auto")
    max_parallel, parallel_mode_text = _resolve_parallelism(
        requested_value=requested_parallelism,
        planned_runs=len(run_specs),
    )

    print(f"Loaded decision-sweep manifest: {manifest_path}")
    print(f"Generated sweep root: {run_root}")
    print(f"Detailed comparison root: {comparison_root}")
    print(f"Planned runs: {len(run_specs)}")
    print(f"Max parallel: {max_parallel}")
    print(f"Parallel mode: {parallel_mode_text}")

    if args.dry_run:
        _write_dry_run_report(run_root, comparison_root, run_specs)
        save_text(
            [
                "Decision sweep dry run",
                f"Generated at: {current_local_timestamp()}",
                f"Control root: {run_root}",
                f"Detailed comparison root: {comparison_root}",
                f"Resolved max parallel: {max_parallel}",
                f"Parallel mode: {parallel_mode_text}",
                "No executed experiment results are available because this was a dry run.",
            ],
            str(comparison_root / "dry_run_note.txt"),
        )
        print("Dry run complete. Generated configs and plan files were written without executing experiments.")
        return 0

    comparison_rows = _run_specs(
        run_specs=run_specs,
        max_parallel=max_parallel,
        continue_on_error=continue_on_error,
    )
    _apply_base_deltas(comparison_rows)

    generated_at_local = current_local_timestamp()
    _write_detailed_comparison_results(
        comparison_root=comparison_root,
        manifest_path=manifest_path,
        run_root=run_root,
        rows=comparison_rows,
        generated_at_local=generated_at_local,
    )
    save_text(
        [
            "Decision sweep result locations",
            f"Generated at: {generated_at_local}",
            f"Control root: {run_root}",
            f"Detailed comparison root: {comparison_root}",
            f"Resolved max parallel: {max_parallel}",
            f"Parallel mode: {parallel_mode_text}",
            f"Overall comparison CSV: {comparison_root / 'overall' / 'comparison_summary.csv'}",
            f"Overall comparison JSON: {comparison_root / 'overall' / 'comparison_summary.json'}",
            f"Overall comparison report: {comparison_root / 'overall' / 'comparison_report.md'}",
            f"Per-decision comparisons root: {comparison_root / 'by_decision'}",
        ],
        str(run_root / "comparison_results_location.txt"),
    )

    completed = sum(1 for row in comparison_rows if row["status"] == "completed")
    failed = len(comparison_rows) - completed
    print("")
    print("Decision sweep summary")
    print(f"Run root: {run_root}")
    print(f"Detailed comparison root: {comparison_root}")
    print(f"Completed runs: {completed}")
    print(f"Failed or missing runs: {failed}")
    print(f"Resolved max parallel: {max_parallel}")
    print(f"Parallel mode: {parallel_mode_text}")
    print(f"Comparison CSV: {comparison_root / 'overall' / 'comparison_summary.csv'}")
    print(f"Comparison JSON: {comparison_root / 'overall' / 'comparison_summary.json'}")
    print(f"Comparison report: {comparison_root / 'overall' / 'comparison_report.md'}")
    print(f"Decision comparisons: {comparison_root / 'by_decision'}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
