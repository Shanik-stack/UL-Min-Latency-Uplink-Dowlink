from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as np
import yaml

from experiment_scenarios import normalize_experiment_scenario_config


def _first_present(mapping: dict, *names: str, default=None):
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def _as_array(values: Any, K: int, name: str, dtype) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype)
    if arr.ndim == 0:
        arr = np.full(K, arr.item(), dtype=dtype)
    if arr.shape != (K,):
        raise ValueError(f"{name} must have shape ({K},), got {arr.shape}")
    return arr


def _resolve_downlink_block_power_budget(power_values: np.ndarray) -> float:
    if power_values.size <= 0:
        raise ValueError("Downlink P must contain at least one value.")
    budget = float(power_values[0])
    if not np.allclose(power_values, budget, rtol=1e-6, atol=1e-9):
        raise ValueError(
            "Downlink uses one BS block power budget for the full precoder F_b, "
            "so test.P must be a scalar or repeated identical values."
        )
    return budget


def _resolve_config_path(cfg_name: str) -> str:
    if not cfg_name.endswith(".yaml"):
        cfg_name = f"{cfg_name}.yaml"

    if os.path.isabs(cfg_name) and os.path.exists(cfg_name):
        return cfg_name

    loader_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(loader_dir)
    candidates = [
        os.path.join(loader_dir, cfg_name),
        os.path.join(project_root, "Experiment Configs", cfg_name),
        os.path.join(project_root, cfg_name),
        os.path.abspath(cfg_name),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(f"Could not find config file: {cfg_name}")


def load_config(cfg_name: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    cfg_path = _resolve_config_path(cfg_name)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    test_cfg = cfg["test"]
    K = int(test_cfg["K"])

    # New downlink schema uses Nb (BS tx antennas) and Nr (UE rx antennas).
    # For convenience, legacy uplink-style configs map Nr -> Nb and Nt -> Nr.
    if "Nb" in test_cfg:
        Nb = _as_array(test_cfg["Nb"], K, "Nb", int)
        Nr = _as_array(test_cfg["Nr"], K, "Nr", int)
    else:
        Nb = _as_array(test_cfg["Nr"], K, "Nr (legacy->Nb)", int)
        Nr = _as_array(test_cfg["Nt"], K, "Nt (legacy->Nr)", int)

    system_params = {
        "K": K,
        "Nb": Nb,
        "Nr": Nr,
        "dk": np.minimum(Nb, Nr),
        "B": _as_array(test_cfg["B"], K, "B", int),
        "P": _as_array(test_cfg["P"], K, "P", float),
        "fs": _as_array(test_cfg["fs"], K, "fs", float),
        "snr_db": _as_array(test_cfg["snr_db"], K, "snr_db", float),
        "epsilon": _as_array(test_cfg["epsilon"], K, "epsilon", float),
        "T": _as_array(test_cfg["T"], K, "T", int),
        "initial_bits_per_symbol": _as_array(
            test_cfg["initial_bits_per_symbol"], K, "initial_bits_per_symbol", float
        ),
    }
    system_params["block_power_budget"] = _resolve_downlink_block_power_budget(system_params["P"])
    system_params["initial_latency"] = (
        system_params["B"] / system_params["initial_bits_per_symbol"]
    ) / system_params["fs"]

    sim_cfg_raw = cfg.get("simulation", {})
    n_range = sim_cfg_raw.get("n_kl_range", {})
    default_monte_carlo_training_blocks = min(int(sim_cfg_raw.get("max_total_blocks", 256)), 2)
    scenario_cfg = normalize_experiment_scenario_config(
        sim_cfg_raw.get("experiment_scenario", {}),
        system_params=system_params,
        max_total_blocks=int(sim_cfg_raw.get("max_total_blocks", 256)),
    )
    max_epochs = int(
        _first_present(
            sim_cfg_raw,
            "max_epochs",
            "main_solve_max_epochs",
            "main_solve_max_sweeps",
            "max_precoder_epochs",
            "max_precoder_sweeps",
            "main_solve_guard_epochs",
            "main_solve_guard_sweeps",
            "reduced_n_kl_repair_max_epochs",
            "reduced_n_kl_repair_max_sweeps",
            "repair_solve_guard_epochs",
            "repair_solve_guard_sweeps",
            default=500,
        )
    )
    sim_params = {
        "max_epochs": max(1, max_epochs),
        "max_precoder_epochs": max(1, max_epochs),
        "print_every_epoch": int(_first_present(sim_cfg_raw, "print_every_epoch", "print_every_sweep", default=1)),
        "user_update_steps": int(sim_cfg_raw.get("user_update_steps", 1)),
        "user_update_lr": float(
            _first_present(
                sim_cfg_raw,
                "user_update_lr",
                "step_lr",
                default=5e-3,
            )
        ),
        "initial_lambda_rate_constraint": float(sim_cfg_raw.get("initial_lambda_rate_constraint", 0.1)),
        "initial_lambda_power_constraint": float(sim_cfg_raw.get("initial_lambda_power_constraint", 0.01)),
        "lr_rate_constraint": float(sim_cfg_raw.get("lr_rate_constraint", 1e-2)),
        "lr_power_constraint": float(sim_cfg_raw.get("lr_power_constraint", 1e-3)),
        "constraint_loss_form": str(sim_cfg_raw.get("constraint_loss_form", "plain_lagrangian")).strip().lower(),
        "augmented_lagrangian_rho_rate": float(sim_cfg_raw.get("augmented_lagrangian_rho_rate", 0.0)),
        "augmented_lagrangian_rho_power": float(sim_cfg_raw.get("augmented_lagrangian_rho_power", 0.0)),
        "kkt_primal_tol": float(sim_cfg_raw.get("kkt_primal_tol", 1e-5)),
        "kkt_complementarity_tol": float(sim_cfg_raw.get("kkt_complementarity_tol", 1e-5)),
        "kkt_stationarity_tol": float(sim_cfg_raw.get("kkt_stationarity_tol", 1e-4)),
        "max_total_blocks": int(sim_cfg_raw.get("max_total_blocks", 256)),
        "n_kl_min": int(n_range.get("min", 5)),
        "n_kl_step": int(n_range.get("step", 1)),
        "n_kl_reduction_update_scope": str(
            _first_present(
                sim_cfg_raw,
                "n_kl_reduction_update_scope",
                "reduced_n_kl_reoptimization_scope",
                default="all_active_users",
            )
        ).strip().lower(),
        "monte_carlo_training_blocks_per_seed": int(
            _first_present(
                sim_cfg_raw,
                "monte_carlo_training_blocks_per_seed",
                "precoder_net_train_blocks_per_seed",
                "precoder_train_blocks_per_seed",
                "policy_train_blocks_per_seed",
                default=default_monte_carlo_training_blocks,
            )
        ),
        "monte_carlo_training_n_kl_coarse_step": int(
            _first_present(
                sim_cfg_raw,
                "monte_carlo_training_n_kl_coarse_step",
                "precoder_net_train_n_kl_coarse_step",
                "precoder_train_n_kl_coarse_step",
                "policy_train_n_kl_coarse_step",
                default=5,
            )
        ),
        "monte_carlo_training_fallback_target_bits": int(
            _first_present(
                sim_cfg_raw,
                "monte_carlo_training_fallback_target_bits",
                "precoder_net_train_min_bits_required",
                "precoder_train_min_bits_required",
                "policy_train_min_bits_required",
                default=1,
            )
        ),
        "convergence_block_objective_mode": str(
            _first_present(
                sim_cfg_raw,
                "convergence_block_objective_mode",
                "safe_sweep_objective_mode",
                "downlink_safe_sweep_objective_mode",
                "objective_mode",
                default="unweighted_sum_rate",
            )
        ).strip().lower(),
        "remaining_bits_weight_power": float(
            _first_present(
                sim_cfg_raw,
                "remaining_bits_weight_power",
                "queue_weight_power",
                default=1.0,
            )
        ),
        "minimum_user_weight": float(
            _first_present(
                sim_cfg_raw,
                "minimum_user_weight",
                "queue_weight_min",
                default=0.25,
            )
        ),
        "network_rate_weight": float(
            _first_present(
                sim_cfg_raw,
                "network_rate_weight",
                "network_weight_beta",
                default=0.15,
            )
        ),
        "latency_penalty_weight": float(
            _first_present(
                sim_cfg_raw,
                "latency_penalty_weight",
                "utility_latency_penalty",
                default=0.5,
            )
        ),
        "downlink_precoder_net_scope": str(
            _first_present(
                sim_cfg_raw,
                "downlink_precoder_net_scope",
                "precoder_net_scope",
                default="per_user_nets",
            )
        ).strip().lower(),
        "experiment_scenario": scenario_cfg,
        "experiment_scenario_mode": str(scenario_cfg["mode"]),
    }

    run_meta = {
        "cfg_path": cfg_path,
        "cfg_stem": os.path.splitext(os.path.basename(cfg_path))[0],
    }
    return system_params, sim_params, run_meta
