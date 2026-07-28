# Payload Direction and Weighting Sweep Summary

- Sweep: `sweep_20260727_185736`
- Manifest: `C:\All Codes\Taiwan_Internship\UL_UPLINK_DOWNLINK_MONTE_CARLO\Experiment Configs\decision_sweep_payload_direction_weighting.yaml`
- Completed runs: `9/9`
- Objective names below use the cleaned public labels such as `unweighted_sum_rate` and `inverse_cnr_weighted_sum_rate`.
- Asynchronality is the sum of pairwise latency differences across users, so smaller is better.

## Downlink | convergence

| Variant               | Configuration                                                                                 | Final latency | Latency reduction % | Final asynchronality | Asynchronality reduction % | Final avg SNR (dB) | Final avg SINR (dB) |
| --------------------- | --------------------------------------------------------------------------------------------- | -------------:| -------------------:| --------------------:| --------------------------:| ------------------:| -------------------:|
| ndir-asc__obj-invcnr  | convergence_block_objective_mode=inverse_cnr_weighted_sum_rate; n_search_direction=ascending  | 0.040267      | 86.7978             | 0.013467             | 89.7670                    | 15.3203            | 18.6234             |
| ndir-desc__obj-invcnr | convergence_block_objective_mode=inverse_cnr_weighted_sum_rate; n_search_direction=descending | 0.040267      | 86.7978             | 0.013467             | 89.7670                    | 15.3203            | 18.6234             |
| ndir-asc__obj-unwt    | convergence_block_objective_mode=unweighted_sum_rate; n_search_direction=ascending            | 0.049333      | 83.8251             | 0.003733             | 97.1631                    | 14.1792            | 8.2492              |
| ndir-desc__obj-unwt   | convergence_block_objective_mode=unweighted_sum_rate; n_search_direction=descending           | 0.049333      | 83.8251             | 0.003733             | 97.1631                    | 14.1792            | 8.2492              |

- Best latency: `ndir-asc__obj-invcnr` -> `0.040267`
- Best latency configuration: `convergence_block_objective_mode=inverse_cnr_weighted_sum_rate; n_search_direction=ascending`
- Best asynchronality: `ndir-asc__obj-unwt` -> `0.003733`
- Best asynchronality configuration: `convergence_block_objective_mode=unweighted_sum_rate; n_search_direction=ascending`
- Interpretation: best latency and best asynchronality differ here, so this group shows a throughput-versus-fairness tradeoff.

## Downlink | monte_carlo

| Variant   | Configuration                 | Final latency | Latency reduction % | Final asynchronality | Asynchronality reduction % | Final avg SNR (dB) | Final avg SINR (dB) |
| --------- | ----------------------------- | -------------:| -------------------:| --------------------:| --------------------------:| ------------------:| -------------------:|
| ndir-desc | n_search_direction=descending | 0.075667      | 75.1913             | 0.048267             | 63.3232                    | 9.3594             | 11.9746             |

- Best latency: `ndir-desc` -> `0.075667`
- Best latency configuration: `n_search_direction=descending`
- Best asynchronality: `ndir-desc` -> `0.048267`
- Best asynchronality configuration: `n_search_direction=descending`

## Uplink | convergence

| Variant   | Configuration                 | Final latency | Latency reduction % | Final asynchronality | Asynchronality reduction % | Final avg SNR (dB) | Final avg SINR (dB) |
| --------- | ----------------------------- | -------------:| -------------------:| --------------------:| --------------------------:| ------------------:| -------------------:|
| ndir-asc  | n_search_direction=ascending  | 0.024667      | 19.7397             | 0.015333             | 8.0000                     | 8.3495             | -3.2712             |
| ndir-desc | n_search_direction=descending | 0.024667      | 19.7397             | 0.015333             | 8.0000                     | 8.3180             | -3.3018             |

- Best latency: `ndir-asc` -> `0.024667`
- Best latency configuration: `n_search_direction=ascending`
- Best asynchronality: `ndir-asc` -> `0.015333`
- Best asynchronality configuration: `n_search_direction=ascending`

## Uplink | monte_carlo

| Variant   | Configuration                 | Final latency | Latency reduction % | Final asynchronality | Asynchronality reduction % | Final avg SNR (dB) | Final avg SINR (dB) |
| --------- | ----------------------------- | -------------:| -------------------:| --------------------:| --------------------------:| ------------------:| -------------------:|
| ndir-desc | n_search_direction=descending | 0.024933      | 18.8720             | 0.015333             | 8.0000                     | 8.3865             | -3.3269             |
| ndir-asc  | n_search_direction=ascending  | 0.027467      | 10.6291             | 0.016267             | 2.4000                     | 8.1852             | -3.4335             |

- Best latency: `ndir-desc` -> `0.024933`
- Best latency configuration: `n_search_direction=descending`
- Best asynchronality: `ndir-desc` -> `0.015333`
- Best asynchronality configuration: `n_search_direction=descending`
