from __future__ import annotations

import os

import yaml

from experiment_scenarios import normalize_experiment_scenario_config
from uplink_rate_model import normalize_uplink_rate_model
from utils import initialize_system_params


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
    simulation_test_params = {
        "initial_lambda_rate_constraint": sim_cfg["initial_lambda_rate_constraint"],
        "initial_lambda_power_constraint": sim_cfg["initial_lambda_power_constraint"],
        "epochs_per_n_kl": sim_cfg.get("epochs_per_n_kl", sim_cfg.get("max_precoder_sweeps", 10000)),
        "lr_net": sim_cfg.get(
            "lr_net",
            lr_cfg.get("net", sim_cfg.get("user_update_lr", sim_cfg.get("step_lr", 1e-2))),
        ),
        "lr_rate_constraint": sim_cfg.get("lr_rate_constraint", lr_cfg.get("rate_constraint", 1e-2)),
        "lr_power_constraint": sim_cfg.get("lr_power_constraint", lr_cfg.get("power_constraint", 1e-3)),
        "n_kl_min": sim_cfg["n_kl_range"]["min"],
        "n_kl_max": n_kl_max,
        "n_kl_step": sim_cfg["n_kl_range"]["step"],
        "max_total_blocks": int(sim_cfg.get("max_total_blocks", 256)),
        "max_precoder_sweeps": int(sim_cfg.get("max_precoder_sweeps", sim_cfg.get("epochs_per_n_kl", 10000))),
        "print_every_sweep": int(sim_cfg.get("print_every_sweep", 1)),
        "precoder_net_train_min_bits_required": int(
            sim_cfg.get(
                "precoder_net_train_min_bits_required",
                sim_cfg.get(
                    "precoder_train_min_bits_required",
                    sim_cfg.get("policy_train_min_bits_required", 1),
                ),
            )
        ),
        "precoder_net_train_blocks_per_seed": int(
            sim_cfg.get(
                "precoder_net_train_blocks_per_seed",
                sim_cfg.get(
                    "precoder_train_blocks_per_seed",
                    sim_cfg.get("policy_train_blocks_per_seed", 1),
                ),
            )
        ),
        "precoder_net_train_n_kl_coarse_step": int(
            sim_cfg.get(
                "precoder_net_train_n_kl_coarse_step",
                sim_cfg.get(
                    "precoder_train_n_kl_coarse_step",
                    sim_cfg.get("policy_train_n_kl_coarse_step", 5),
                ),
            )
        ),
        "step_lr": float(sim_cfg.get("step_lr", sim_cfg.get("lr_net", lr_cfg.get("net", 1e-2)))),
        "user_update_steps": int(sim_cfg.get("user_update_steps", 1)),
        "user_update_lr": float(
            sim_cfg.get(
                "user_update_lr",
                sim_cfg.get("lr_net", lr_cfg.get("net", sim_cfg.get("step_lr", 1e-2))),
            )
        ),
        "convergence_max_precoder_sweeps": int(
            sim_cfg.get(
                "convergence_max_precoder_sweeps",
                min(int(sim_cfg.get("max_precoder_sweeps", sim_cfg.get("epochs_per_n_kl", 500))), 500),
            )
        ),
        "convergence_min_precoder_sweeps_before_stop": int(
            sim_cfg.get("convergence_min_precoder_sweeps_before_stop", 1)
        ),
        "reduced_n_kl_max_precoder_sweeps": int(
            sim_cfg.get(
                "reduced_n_kl_max_precoder_sweeps",
                sim_cfg.get("epochs_per_n_kl", sim_cfg.get("max_precoder_sweeps", 10000)),
            )
        ),
        "reduced_n_kl_min_precoder_sweeps_before_stop": int(
            sim_cfg.get(
                "reduced_n_kl_min_precoder_sweeps_before_stop",
                sim_cfg.get("convergence_min_precoder_sweeps_before_stop", 1),
            )
        ),
        "print_every_reduced_n_kl": int(sim_cfg.get("print_every_reduced_n_kl", 1)),
        "convergence_precoder_tol": float(sim_cfg.get("convergence_precoder_tol", 1e-4)),
        "convergence_feasibility_tol": float(sim_cfg.get("convergence_feasibility_tol", 1e-5)),
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

    
    
    
    
