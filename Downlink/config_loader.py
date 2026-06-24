from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as np
import yaml


def _as_array(values: Any, K: int, name: str, dtype) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype)
    if arr.ndim == 0:
        arr = np.full(K, arr.item(), dtype=dtype)
    if arr.shape != (K,):
        raise ValueError(f"{name} must have shape ({K},), got {arr.shape}")
    return arr


def _resolve_config_path(cfg_name: str) -> str:
    if not cfg_name.endswith(".yaml"):
        cfg_name = f"{cfg_name}.yaml"

    if os.path.isabs(cfg_name) and os.path.exists(cfg_name):
        return cfg_name

    local_candidate = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg_name)
    if os.path.exists(local_candidate):
        return local_candidate

    cwd_candidate = os.path.abspath(cfg_name)
    if os.path.exists(cwd_candidate):
        return cwd_candidate

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
    system_params["initial_latency"] = (
        system_params["B"] / system_params["initial_bits_per_symbol"]
    ) / system_params["fs"]

    sim_cfg_raw = cfg.get("simulation", {})
    n_range = sim_cfg_raw.get("n_kl_range", {})
    default_precoder_net_train_blocks = min(int(sim_cfg_raw.get("max_total_blocks", 256)), 2)
    sim_params = {
        "max_precoder_sweeps": int(sim_cfg_raw.get("max_precoder_sweeps", 25)),
        "print_every_sweep": int(sim_cfg_raw.get("print_every_sweep", 1)),
        "step_lr": float(sim_cfg_raw.get("step_lr", 5e-3)),
        "user_update_steps": int(sim_cfg_raw.get("user_update_steps", 1)),
        "user_update_lr": float(sim_cfg_raw.get("user_update_lr", sim_cfg_raw.get("step_lr", 5e-3))),
        "precoder_tol": float(sim_cfg_raw.get("precoder_tol", 1e-4)),
        "max_total_blocks": int(sim_cfg_raw.get("max_total_blocks", 256)),
        "n_kl_min": int(n_range.get("min", 5)),
        "n_kl_step": int(n_range.get("step", 1)),
        "precoder_net_train_blocks_per_seed": int(
            sim_cfg_raw.get(
                "precoder_net_train_blocks_per_seed",
                sim_cfg_raw.get(
                    "precoder_train_blocks_per_seed",
                    sim_cfg_raw.get("policy_train_blocks_per_seed", default_precoder_net_train_blocks),
                ),
            )
        ),
        "precoder_net_train_n_kl_coarse_step": int(
            sim_cfg_raw.get(
                "precoder_net_train_n_kl_coarse_step",
                sim_cfg_raw.get(
                    "precoder_train_n_kl_coarse_step",
                    sim_cfg_raw.get("policy_train_n_kl_coarse_step", 5),
                ),
            )
        ),
        "precoder_net_train_min_bits_required": int(
            sim_cfg_raw.get(
                "precoder_net_train_min_bits_required",
                sim_cfg_raw.get(
                    "precoder_train_min_bits_required",
                    sim_cfg_raw.get("policy_train_min_bits_required", 1),
                ),
            )
        ),
        "precoder_net_train_max_reduction_rounds_per_epoch": int(
            sim_cfg_raw.get("precoder_net_train_max_reduction_rounds_per_epoch", 4)
        ),
        "precoder_net_train_curriculum_warmup_epochs": int(
            sim_cfg_raw.get("precoder_net_train_curriculum_warmup_epochs", 0)
        ),
        "precoder_net_train_curriculum_interval_epochs": int(
            sim_cfg_raw.get("precoder_net_train_curriculum_interval_epochs", 1)
        ),
        "precoder_net_train_enumerate_all_masks_up_to_k": int(
            sim_cfg_raw.get("precoder_net_train_enumerate_all_masks_up_to_k", 3)
        ),
        "safe_sweep_objective_mode": str(
            sim_cfg_raw.get(
                "safe_sweep_objective_mode",
                sim_cfg_raw.get(
                    "downlink_safe_sweep_objective_mode",
                    sim_cfg_raw.get("objective_mode", "user_rate"),
                ),
            )
        ).strip().lower(),
        "queue_weight_power": float(sim_cfg_raw.get("queue_weight_power", 1.0)),
        "queue_weight_min": float(sim_cfg_raw.get("queue_weight_min", 0.25)),
        "network_weight_beta": float(sim_cfg_raw.get("network_weight_beta", 0.15)),
        "utility_latency_penalty": float(sim_cfg_raw.get("utility_latency_penalty", 0.5)),
    }

    run_meta = {
        "cfg_path": cfg_path,
        "cfg_stem": os.path.splitext(os.path.basename(cfg_path))[0],
    }
    return system_params, sim_params, run_meta
