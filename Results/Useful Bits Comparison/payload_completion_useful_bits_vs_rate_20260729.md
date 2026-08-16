# Useful Bits vs Rate Comparison

Date: 2026-07-29
Scenario: `payload_completion`
Goal: compare the new `beam_reward_mode: useful_bits` runs against the existing saved `beam_reward_mode: rate` runs.

## Comparison table

| Link | Method | Reward | Final total latency | Latency reduction vs random (%) | Final asynchronality sum | Asynchronality reduction (%) | Final avg SNR (dB) | Final avg SINR / block SINR (dB) | Skipped blocks |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Uplink | Convergence per epoch | `rate` | 0.024667 | 19.7397 | 0.015333 | 8.0000 | 8.2627 | -3.2868 | 0 |
| Uplink | Convergence per epoch | `useful_bits` | 0.024667 | 19.7397 | 0.015333 | 8.0000 | 8.4480 | -3.2836 | 0 |
| Uplink | Monte Carlo | `rate` | 0.025200 | 18.0043 | 0.015333 | 8.0000 | 8.3355 | -3.4146 | 0 |
| Uplink | Monte Carlo | `useful_bits` | 0.024600 | 19.9566 | 0.015333 | 8.0000 | 8.1140 | -3.4033 | 0 |
| Downlink | Convergence per epoch | `rate` | 0.035800 | 88.2623 | 0.007867 | 94.0223 | 13.9912 | 12.5367 | 3 |
| Downlink | Convergence per epoch | `useful_bits` | 0.040267 | 86.7978 | 0.013467 | 89.7670 | 15.3206 | 18.6234 | 6 |
| Downlink | Monte Carlo | `rate` | 0.073533 | 75.8907 | 0.046400 | 64.7416 | 9.9357 | 12.5690 | 10 |
| Downlink | Monte Carlo | `useful_bits` | 0.775733 | -154.3388 | 0.261067 | -98.3789 | 5.5464 | 1.8109 | 150 |

## Main takeaways

- Uplink convergence: `useful_bits` did not change the chosen latency schedule at all. The final latency and asynchronality were identical to `rate`.
- Uplink Monte Carlo: `useful_bits` slightly improved final latency over the existing `rate` run, from `0.025200` to `0.024600`.
- Downlink convergence: `useful_bits` produced stronger per-block link quality, but worse end-to-end latency and worse asynchronality because it skipped more blocks.
- Downlink Monte Carlo: `useful_bits` is currently not viable for payload completion. It finished with much higher latency than even the random baseline and left a long tail of tiny remaining payload that took many extra blocks to clear.

## Important caveats

- The convergence comparisons are clean on the test side because both use the same test seed `3`.
- The existing saved Monte Carlo `rate` baselines were older runs with train seeds `[1, 2]`.
- The new Monte Carlo `useful_bits` runs used train seeds `[0, 1, 2]`.
- Because of that train-seed mismatch, the Monte Carlo comparisons are directionally useful but not perfectly controlled.
- Runtime and FLOP comparisons are also not perfectly apples-to-apples against the older saved runs, because the codebase has changed since the older `rate` artifacts were generated.

## Result folders used

- Uplink convergence `rate`:
  `Results/Uplink/Method-Convergence per epoch/conv_net__payload_completion__s3`
- Uplink convergence `useful_bits`:
  `Results/Uplink/Method-Convergence per epoch/conv_unwt_net__payload_completion_useful_bits__s3`
- Uplink Monte Carlo `rate`:
  `Results/Uplink/Method-Monte Carlo/mc__payload_completion__s3`
- Uplink Monte Carlo `useful_bits`:
  `Results/Uplink/Method-Monte Carlo/mc_unwt__payload_completion_useful_bits__s3`
- Downlink convergence `rate`:
  `Results/Downlink/Method-Convergence per epoch/conv_sum_user_net__payload_completion__s3`
- Downlink convergence `useful_bits`:
  `Results/Downlink/Method-Convergence per epoch/conv_unwt_user_net__payload_completion_useful_bits__s3`
- Downlink Monte Carlo `rate`:
  `Results/Downlink/Method-Monte Carlo/mc_user__payload_completion__s3`
- Downlink Monte Carlo `useful_bits`:
  `Results/Downlink/Method-Monte Carlo/mc_unwt_user__payload_completion_useful_bits__s3`
