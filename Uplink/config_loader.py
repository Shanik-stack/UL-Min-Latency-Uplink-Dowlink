from __future__ import annotations

import os

import yaml

from experiment_scenarios import normalize_experiment_scenario_config
from uplink_rate_model import normalize_uplink_rate_model
from utils import initialize_system_params


def _first_present(mapping: dict, *names: str, default=None):
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


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


def get_config(cfg_name: str) -> tuple[dict, dict]:
    cfg_path = _resolve_config_path(cfg_name)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    test_cfg = cfg["test"]
    test_k = test_cfg["K"]
    test_Nr = test_cfg["Nr"]
    test_Nt = test_cfg["Nt"]
    initial_bits_per_symbol = test_cfg.get("initial_bits_per_symbol")

    use_raw_t_initializer = (
        "raw_T" in os.path.basename(cfg_path)
        or "f_carrier" not in test_cfg
        or "v" not in test_cfg
    )

    if use_raw_t_initializer:
        system_test_params = initialize_system_params(
            B=test_cfg["B"],
            P=test_cfg["P"],
            fs=test_cfg["fs"],
            snr_db=test_cfg["snr_db"],
            desired_CNR=None,
            Nt=test_Nt,
            Nr=test_Nr,
            K=test_k,
            epsilon=test_cfg["epsilon"],
            initial_bits_per_symbol=initial_bits_per_symbol,
            T=test_cfg["T"],
        )
    else:
        system_test_params = initialize_system_params(
            B=test_cfg["B"],
            P=test_cfg["P"],
            fs=test_cfg["fs"],
            snr_db=test_cfg["snr_db"],
            desired_CNR=None,
            Nt=test_Nt,
            Nr=test_Nr,
            K=test_k,
            f_carrier=test_cfg["f_carrier"],
            v=test_cfg["v"],
            epsilon=test_cfg["epsilon"],
            initial_bits_per_symbol=initial_bits_per_symbol,
        )

    sim_cfg = cfg["simulation"]
    uplink_rate_model = normalize_uplink_rate_model(sim_cfg.get("uplink_rate_model", "sinr"))
    lr_cfg = sim_cfg.get("lr", {})
    scenario_cfg = normalize_experiment_scenario_config(
        sim_cfg.get("experiment_scenario", {}),
        system_params=system_test_params,
        max_total_blocks=int(sim_cfg.get("max_total_blocks", 256)),
    )
    n_kl_max = [system_test_params["T"][user] for user in range(system_test_params["K"])]
    max_epochs = int(
        _first_present(
            sim_cfg,
            "max_epochs",
            "main_solve_max_epochs",
            "main_solve_max_sweeps",
            "convergence_max_precoder_epochs",
            "convergence_max_precoder_sweeps",
            "max_precoder_epochs",
            "max_precoder_sweeps",
            "solve_epochs_per_n_kl",
            "solve_sweeps_per_n_kl",
            "epochs_per_n_kl",
            "main_solve_guard_epochs",
            "main_solve_guard_sweeps",
            "reduced_n_kl_repair_max_epochs",
            "reduced_n_kl_repair_max_sweeps",
            "repair_solve_guard_epochs",
            "repair_solve_guard_sweeps",
            "reduced_n_kl_max_precoder_epochs",
            "reduced_n_kl_max_precoder_sweeps",
            default=500,
        )
    )
    simulation_test_params = {
        "initial_lambda_rate_constraint": sim_cfg["initial_lambda_rate_constraint"],
        "initial_lambda_power_constraint": sim_cfg["initial_lambda_power_constraint"],
        "max_epochs": max(1, max_epochs),
        "lr_net": sim_cfg.get(
            "lr_net",
            lr_cfg.get("net", sim_cfg.get("user_update_lr", sim_cfg.get("step_lr", 1e-2))),
        ),
        "lr_rate_constraint": sim_cfg.get("lr_rate_constraint", lr_cfg.get("rate_constraint", 1e-2)),
        "lr_power_constraint": sim_cfg.get("lr_power_constraint", lr_cfg.get("power_constraint", 1e-3)),
        "constraint_loss_form": str(sim_cfg.get("constraint_loss_form", "plain_lagrangian")).strip().lower(),
        "augmented_lagrangian_rho_rate": float(sim_cfg.get("augmented_lagrangian_rho_rate", 0.0)),
        "augmented_lagrangian_rho_power": float(sim_cfg.get("augmented_lagrangian_rho_power", 0.0)),
        "n_kl_min": sim_cfg["n_kl_range"]["min"],
        "n_kl_max": n_kl_max,
        "n_kl_step": sim_cfg["n_kl_range"]["step"],
        "max_total_blocks": int(sim_cfg.get("max_total_blocks", 256)),
        "max_precoder_epochs": max(1, max_epochs),
        "print_every_epoch": int(_first_present(sim_cfg, "print_every_epoch", "print_every_sweep", default=1)),
        "monte_carlo_training_fallback_target_bits": int(
            _first_present(
                sim_cfg,
                "monte_carlo_training_fallback_target_bits",
                "precoder_net_train_min_bits_required",
                "precoder_train_min_bits_required",
                "policy_train_min_bits_required",
                default=1,
            )
        ),
        "monte_carlo_training_blocks_per_seed": int(
            _first_present(
                sim_cfg,
                "monte_carlo_training_blocks_per_seed",
                "precoder_net_train_blocks_per_seed",
                "precoder_train_blocks_per_seed",
                "policy_train_blocks_per_seed",
                default=1,
            )
        ),
        "monte_carlo_training_n_kl_coarse_step": int(
            _first_present(
                sim_cfg,
                "monte_carlo_training_n_kl_coarse_step",
                "precoder_net_train_n_kl_coarse_step",
                "precoder_train_n_kl_coarse_step",
                "policy_train_n_kl_coarse_step",
                default=5,
            )
        ),
        "kkt_primal_tol": float(
            sim_cfg.get("kkt_primal_tol", sim_cfg.get("convergence_feasibility_tol", 1e-5))
        ),
        "kkt_complementarity_tol": float(
            sim_cfg.get("kkt_complementarity_tol", sim_cfg.get("convergence_feasibility_tol", 1e-5))
        ),
        "kkt_stationarity_tol": float(
            sim_cfg.get("kkt_stationarity_tol", sim_cfg.get("convergence_precoder_tol", 1e-4))
        ),
        "reduced_n_kl_log_interval": int(
            _first_present(
                sim_cfg,
                "reduced_n_kl_log_interval",
                "print_every_reduced_n_kl",
                default=1,
            )
        ),
        "uplink_rate_model": uplink_rate_model,
        "experiment_scenario": scenario_cfg,
        "experiment_scenario_mode": str(scenario_cfg["mode"]),
    }
    system_test_params["uplink_rate_model"] = uplink_rate_model
    return system_test_params, simulation_test_params


def load_config(cfg_name: str) -> tuple[dict, dict, dict]:
    system_test_params, simulation_test_params = get_config(cfg_name)
    cfg_path = _resolve_config_path(cfg_name)
    run_meta = {
        "cfg_path": cfg_path,
        "cfg_stem": os.path.splitext(os.path.basename(cfg_path))[0],
    }
    return system_test_params, simulation_test_params, run_meta


if __name__ == "__main__":
    pass

    
    
    
    
