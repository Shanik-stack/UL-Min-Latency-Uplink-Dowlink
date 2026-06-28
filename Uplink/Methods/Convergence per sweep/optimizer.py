from Optimizer_per_block import (
    dynamic_fixed_target_precoder_training,
    dynamic_subblocklength_precoder_training,
)
from experiment_scenarios import FIXED_BLOCK_TARGETS_MODE


def dynamic_subblocklength_precoder_training_baseline(
    uplinksystem,
    sim_cfg: dict,
    channel_norm: bool = True,
    interference_F_snapshot=None,
    commit_live_precoders: bool = True,
):
    """
    Uplink convergence baseline aligned with the downlink convergence structure:
      - one shared user precoder net per user
      - online block-by-block convergence in a single run
      - no separate train/test phase inside this method

    We also cap the effective per-(user, block, n_kl) inner sweeps by default so
    the convergence baseline does not inherit the very large Monte Carlo training
    sweep counts from the shared configs.
    """

    local_sim_cfg = dict(sim_cfg)
    requested_epochs = int(local_sim_cfg.get("epochs_per_n_kl", local_sim_cfg.get("max_precoder_sweeps", 500)))
    requested_max_sweeps = int(local_sim_cfg.get("max_precoder_sweeps", requested_epochs))
    convergence_cap = int(
        local_sim_cfg.get(
            "convergence_max_precoder_sweeps",
            min(requested_max_sweeps, 500),
        )
    )
    effective_sweeps = max(1, min(requested_epochs, requested_max_sweeps, convergence_cap))
    local_sim_cfg["epochs_per_n_kl"] = int(effective_sweeps)
    local_sim_cfg["max_precoder_sweeps"] = int(effective_sweeps)
    local_sim_cfg["convergence_min_precoder_sweeps_before_stop"] = int(
        local_sim_cfg.get("convergence_min_precoder_sweeps_before_stop", 1)
    )

    if str(local_sim_cfg.get("experiment_scenario_mode", "")) == FIXED_BLOCK_TARGETS_MODE:
        convergence_data = dynamic_fixed_target_precoder_training(
            uplinksystem=uplinksystem,
            sim_cfg=local_sim_cfg,
            channel_norm=channel_norm,
            interference_F_snapshot=interference_F_snapshot,
            commit_live_precoders=commit_live_precoders,
        )
    else:
        convergence_data = dynamic_subblocklength_precoder_training(
            uplinksystem=uplinksystem,
            sim_cfg=local_sim_cfg,
            channel_norm=channel_norm,
            interference_F_snapshot=interference_F_snapshot,
            commit_live_precoders=commit_live_precoders,
        )
    convergence_data["precoder_parameterization"] = "shared_user_channel_sigma_epsilon_to_precoder_mlp_online_convergence"
    convergence_data["method_name"] = "converge_in_each_sweep_baseline"
    convergence_data["effective_convergence_sweeps_per_n_kl"] = int(effective_sweeps)
    return convergence_data
