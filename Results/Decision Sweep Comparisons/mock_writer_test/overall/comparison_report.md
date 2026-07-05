# Decision Sweep Comparison

- Generated at: `2026-07-04 00:02:00`
- Manifest: `C:\All Codes\Taiwan_Internship\UL_UPLINK_DOWNLINK_MONTE_CARLO\Experiment Configs\decision_sweep.yaml`
- Run root: `C:\All Codes\Taiwan_Internship\UL_UPLINK_DOWNLINK_MONTE_CARLO\Decision Sweep Runs\mock_writer_test`
- Total planned runs: `2`
- Completed runs: `2`
- Failed or missing runs: `0`

## downlink | monte_carlo | downlink_fixed_block_targets.yaml

- Scenario: `fixed_block_targets`
- Seed: `3`
- Train seeds: `0,1,2`
- Decision axes: `Constraint loss form, Downlink precoder scope`
- Base configuration: `downlink_precoder_net_scope=per_user_nets | constraint_loss_form=plain_lagrangian`

| Variant | Base | Status | Highlighted configuration | Final total latency | Delta vs base | Latency reduction % | Delta vs base | Final avg SINR (dB) | Delta vs base | Served bits | Log |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| scope-bs__loss-plain |  | completed | downlink_precoder_net_scope=bs_shared_net | constraint_loss_form=plain_lagrangian | 0.700000 | -0.100000 | 30.0000 | 5.0000 | 4.0000 | 0.5000 | 100 | `log_a.txt` |
| scope-user__loss-plain | yes | completed | downlink_precoder_net_scope=per_user_nets | constraint_loss_form=plain_lagrangian | 0.800000 | 0.000000 | 20.0000 | 0.0000 | 3.0000 | 0.0000 | 90 | `log_b.txt` |

- Best final total latency: `scope-bs__loss-plain` -> `0.700000`
- Best-latency configuration: `downlink_precoder_net_scope=bs_shared_net | constraint_loss_form=plain_lagrangian`
- Best latency reduction: `scope-bs__loss-plain` -> `30.0000%`
- Best-reduction configuration: `downlink_precoder_net_scope=bs_shared_net | constraint_loss_form=plain_lagrangian`