from __future__ import annotations

import os

import yaml

from blocklength_search import normalize_n_search_direction, normalize_n_search_strategy
from experiment_scenarios import normalize_experiment_scenario_config
from uplink_rate_model import normalize_uplink_rate_model
from utils import initialize_system_params

UNWEIGHTED_SUM_RATE_OBJECTIVE = "unweighted_sum_rate"
INVERSE_CNR_WEIGHTED_SUM_RATE_OBJECTIVE = "inverse_cnr_weighted_sum_rate"
ASYNCHRONALITY_WEIGHTED_SUM_RATE_OBJECTIVE = "asynchronality_weighted_sum_rate"
UPLINK_OBJECTIVE_MODE_ALIASES = {
    "user_rate": UNWEIGHTED_SUM_RATE_OBJECTIVE,
    "sum_rate": UNWEIGHTED_SUM_RATE_OBJECTIVE,
    "equal_priority_sum_rate": UNWEIGHTED_SUM_RATE_OBJECTIVE,
    "unweighted_sum_rate": UNWEIGHTED_SUM_RATE_OBJECTIVE,
    UNWEIGHTED_SUM_RATE_OBJECTIVE: UNWEIGHTED_SUM_RATE_OBJECTIVE,
    "priority_weighted_sum_rate": INVERSE_CNR_WEIGHTED_SUM_RATE_OBJECTIVE,
    "inverse_cnr_weighted_sum_rate": INVERSE_CNR_WEIGHTED_SUM_RATE_OBJECTIVE,
    INVERSE_CNR_WEIGHTED_SUM_RATE_OBJECTIVE: INVERSE_CNR_WEIGHTED_SUM_RATE_OBJECTIVE,
    "projected_latency_gap_weighted_sum_rate": ASYNCHRONALITY_WEIGHTED_SUM_RATE_OBJECTIVE,
    "projected_completion_latency_gap_weighted_sum_rate": ASYNCHRONALITY_WEIGHTED_SUM_RATE_OBJECTIVE,
    "asynchronality_weighted_sum_rate": ASYNCHRONALITY_WEIGHTED_SUM_RATE_OBJECTIVE,
    ASYNCHRONALITY_WEIGHTED_SUM_RATE_OBJECTIVE: ASYNCHRONALITY_WEIGHTED_SUM_RATE_OBJECTIVE,
}


def resolve_uplink_objective_mode(value) -> str:
    raw_mode = str(value or UNWEIGHTED_SUM_RATE_OBJECTIVE).strip().lower()
    if raw_mode not in UPLINK_OBJECTIVE_MODE_ALIASES:
        known = ", ".join(sorted(set(UPLINK_OBJECTIVE_MODE_ALIASES.values())))
        raise ValueError(f"Unknown uplink objective mode '{raw_mode}'. Expected one of: {known}")
    return str(UPLINK_OBJECTIVE_MODE_ALIASES[raw_mode])


def _first_present(mapping: dict, *names: str, default=None):
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def _optional_int(value, default=None):
    if value is None:
        return default
    return int(value)


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
    uplink_objective_mode = resolve_uplink_objective_mode(
        sim_cfg.get("uplink_objective_mode", UNWEIGHTED_SUM_RATE_OBJECTIVE)
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
        "convergence_precoder_update_mode": str(
            sim_cfg.get("convergence_precoder_update_mode", "precoder_net")
        ).strip().lower(),
        "constraint_loss_form": str(sim_cfg.get("constraint_loss_form", "plain_lagrangian")).strip().lower(),
        "augmented_lagrangian_rho_rate": float(sim_cfg.get("augmented_lagrangian_rho_rate", 0.0)),
        "augmented_lagrangian_rho_power": float(sim_cfg.get("augmented_lagrangian_rho_power", 0.0)),
        "n_kl_min": sim_cfg["n_kl_range"]["min"],
        "n_kl_max": n_kl_max,
        "n_kl_step": sim_cfg["n_kl_range"]["step"],
        "n_search_direction": normalize_n_search_direction(
            sim_cfg.get("n_search_direction", "descending")
        ),
        "n_search_strategy": normalize_n_search_strategy(
            sim_cfg.get("n_search_strategy", "fixed_step")
        ),
        "n_search_coarse_step": int(sim_cfg.get("n_search_coarse_step", sim_cfg["n_kl_range"]["step"])),
        "n_search_exponential_factor": int(sim_cfg.get("n_search_exponential_factor", 2)),
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
        "monte_carlo_training_max_epochs": int(
            _first_present(
                sim_cfg,
                "monte_carlo_training_max_epochs",
                "precoder_net_epochs",
                "precoder_epochs",
                "policy_epochs",
                default=max_epochs,
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
        "monte_carlo_training_full_block_weight": float(
            _first_present(
                sim_cfg,
                "monte_carlo_training_full_block_weight",
                default=1.0,
            )
        ),
        "monte_carlo_training_tail_feasible_weight": float(
            _first_present(
                sim_cfg,
                "monte_carlo_training_tail_feasible_weight",
                default=1.0,
            )
        ),
        "monte_carlo_training_tail_frontier_weight": float(
            _first_present(
                sim_cfg,
                "monte_carlo_training_tail_frontier_weight",
                default=1.5,
            )
        ),
        "monte_carlo_rollout_query_weighting_mode": str(
            _first_present(
                sim_cfg,
                "monte_carlo_rollout_query_weighting_mode",
                default="phase_balanced",
            )
        ).strip().lower(),
        "monte_carlo_train_seeds": _first_present(
            sim_cfg,
            "monte_carlo_train_seeds",
            default=None,
        ),
        "monte_carlo_num_train_seeds": _optional_int(
            _first_present(
                sim_cfg,
                "monte_carlo_num_train_seeds",
                default=None,
            ),
            default=None,
        ),
        "monte_carlo_test_seed": _optional_int(
            _first_present(
                sim_cfg,
                "monte_carlo_test_seed",
                default=None,
            ),
            default=None,
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
        "uplink_objective_mode": uplink_objective_mode,
        "experiment_scenario": scenario_cfg,
        "experiment_scenario_mode": str(scenario_cfg["mode"]),
    }
    system_test_params["uplink_rate_model"] = uplink_rate_model
    system_test_params["uplink_objective_mode"] = uplink_objective_mode
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

    
    
    
    
