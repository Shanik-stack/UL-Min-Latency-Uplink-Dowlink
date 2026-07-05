# Decision Value Comparison: Downlink precoder scope = bs_shared_net

- Decision path: `simulation.downlink_precoder_net_scope`
- Completed runs: `1` / `1`

| Link | Method | Base config | Variant | Final total latency | Latency reduction % | Final avg SINR (dB) | Served bits | Highlighted configuration |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| downlink | monte_carlo | downlink_fixed_block_targets.yaml | scope-bs__loss-plain | 0.700000 | 30.0000 | 4.0000 | 100 | downlink_precoder_net_scope=bs_shared_net | constraint_loss_form=plain_lagrangian |

- Best final total latency: `scope-bs__loss-plain` -> `0.700000`
- Best latency configuration: `downlink_precoder_net_scope=bs_shared_net | constraint_loss_form=plain_lagrangian`
- Best latency reduction: `scope-bs__loss-plain` -> `30.0000%`
- Best reduction configuration: `downlink_precoder_net_scope=bs_shared_net | constraint_loss_form=plain_lagrangian`