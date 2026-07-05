# Decision Value Comparison: Downlink precoder scope = per_user_nets

- Decision path: `simulation.downlink_precoder_net_scope`
- Completed runs: `1` / `1`

| Link | Method | Base config | Variant | Final total latency | Latency reduction % | Final avg SINR (dB) | Served bits | Highlighted configuration |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| downlink | monte_carlo | downlink_fixed_block_targets.yaml | scope-user__loss-plain | 0.800000 | 20.0000 | 3.0000 | 90 | downlink_precoder_net_scope=per_user_nets | constraint_loss_form=plain_lagrangian |

- Best final total latency: `scope-user__loss-plain` -> `0.800000`
- Best latency configuration: `downlink_precoder_net_scope=per_user_nets | constraint_loss_form=plain_lagrangian`
- Best latency reduction: `scope-user__loss-plain` -> `20.0000%`
- Best reduction configuration: `downlink_precoder_net_scope=per_user_nets | constraint_loss_form=plain_lagrangian`