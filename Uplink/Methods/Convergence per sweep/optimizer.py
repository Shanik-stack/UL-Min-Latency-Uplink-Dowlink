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
    """

    local_sim_cfg = dict(sim_cfg)
    effective_epochs = max(1, int(local_sim_cfg["max_epochs"]))
    local_sim_cfg["max_epochs"] = int(effective_epochs)
    local_sim_cfg["max_precoder_epochs"] = int(effective_epochs)

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
    convergence_data["precoder_parameterization"] = "shared_user_channel_n_sigma_epsilon_to_precoder_mlp_online_convergence"
    convergence_data["method_name"] = "convergence_per_epoch_baseline"
    convergence_data["configured_max_epochs"] = int(effective_epochs)
    return convergence_data
