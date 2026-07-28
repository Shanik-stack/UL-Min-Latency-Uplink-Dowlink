# Decision Sweep Comparison

- Generated at: `2026-07-27T21:00:57+03:00`
- Manifest: `C:\All Codes\Taiwan_Internship\UL_UPLINK_DOWNLINK_MONTE_CARLO\Experiment Configs\decision_sweep_payload_direction_weighting.yaml`
- Run root: `C:\All Codes\Taiwan_Internship\UL_UPLINK_DOWNLINK_MONTE_CARLO\Decision Sweep Runs\sweep_20260727_185736`
- Total planned runs: `9`
- Completed runs: `9`
- Failed or missing runs: `0`

## downlink | convergence | downlink_payload_completion.yaml

- Scenario: `payload_completion`
- Seed: `3`
- Train seeds: `n/a`
- Decision axes: `Downlink convergence objective, n_kl search direction`
- Base configuration: `convergence_block_objective_mode=unweighted_sum_rate; n_search_direction=descending`

| Variant               | Base | Status    | Highlighted configuration                                                                     | Final total latency | Delta vs base | Latency reduction % | Delta vs base | Final asynchronality | Delta vs base | Asynchronality reduction % | Delta vs base | Final avg SINR (dB) | Served bits | Log                                                                            |
| --------------------- | ---- | --------- | --------------------------------------------------------------------------------------------- | -------------------:| -------------:| -------------------:| -------------:| --------------------:| -------------:| --------------------------:| -------------:| -------------------:| -----------:| ------------------------------------------------------------------------------ |
| ndir-asc__obj-invcnr  |      | completed | convergence_block_objective_mode=inverse_cnr_weighted_sum_rate; n_search_direction=ascending  | 0.040267            | -0.009067     | 86.7978             | 2.9727        | 0.013467             | 0.009733      | 89.7670                    | -7.3961       | 18.6234             | 6000        | `dl_payload_completion__sweep_20260727_185736__conv__ndir-asc__obj-prio.log`   |
| ndir-desc__obj-invcnr |      | completed | convergence_block_objective_mode=inverse_cnr_weighted_sum_rate; n_search_direction=descending | 0.040267            | -0.009067     | 86.7978             | 2.9727        | 0.013467             | 0.009733      | 89.7670                    | -7.3961       | 18.6234             | 6000        | `dl_payload_completion__sweep_20260727_185736__conv__ndir-desc__obj-prio.log`  |
| ndir-asc__obj-unwt    |      | completed | convergence_block_objective_mode=unweighted_sum_rate; n_search_direction=ascending            | 0.049333            | 0.000000      | 83.8251             | 0.0000        | 0.003733             | 0.000000      | 97.1631                    | 0.0000        | 8.2492              | 6000        | `dl_payload_completion__sweep_20260727_185736__conv__ndir-asc__obj-eqsum.log`  |
| ndir-desc__obj-unwt   | yes  | completed | convergence_block_objective_mode=unweighted_sum_rate; n_search_direction=descending           | 0.049333            | 0.000000      | 83.8251             | 0.0000        | 0.003733             | 0.000000      | 97.1631                    | 0.0000        | 8.2492              | 6000        | `dl_payload_completion__sweep_20260727_185736__conv__ndir-desc__obj-eqsum.log` |

- Best final total latency: `ndir-asc__obj-invcnr` -> `0.040267`
- Best-latency configuration: `convergence_block_objective_mode=inverse_cnr_weighted_sum_rate; n_search_direction=ascending`
- Best latency reduction: `ndir-asc__obj-invcnr` -> `86.7978%`
- Best-reduction configuration: `convergence_block_objective_mode=inverse_cnr_weighted_sum_rate; n_search_direction=ascending`
- Best final asynchronality: `ndir-asc__obj-unwt` -> `0.003733`
- Best-asynchronality configuration: `convergence_block_objective_mode=unweighted_sum_rate; n_search_direction=ascending`
- Best asynchronality reduction: `ndir-asc__obj-unwt` -> `97.1631%`
- Best asynchronality-reduction configuration: `convergence_block_objective_mode=unweighted_sum_rate; n_search_direction=ascending`

## downlink | monte_carlo | downlink_payload_completion.yaml

- Scenario: `payload_completion`
- Seed: `3`
- Train seeds: `1,2`
- Decision axes: `n_kl search direction`
- Base configuration: `n_search_direction=descending`

| Variant   | Base | Status    | Highlighted configuration     | Final total latency | Delta vs base | Latency reduction % | Delta vs base | Final asynchronality | Delta vs base | Asynchronality reduction % | Delta vs base | Final avg SINR (dB) | Served bits | Log                                                               |
| --------- | ---- | --------- | ----------------------------- | -------------------:| -------------:| -------------------:| -------------:| --------------------:| -------------:| --------------------------:| -------------:| -------------------:| -----------:| ----------------------------------------------------------------- |
| ndir-desc | yes  | completed | n_search_direction=descending | 0.075667            | 0.000000      | 75.1913             | 0.0000        | 0.048267             | 0.000000      | 63.3232                    | 0.0000        | 11.9746             | 6000        | `dl_payload_completion__sweep_20260727_185736__mc__ndir-desc.log` |

- Best final total latency: `ndir-desc` -> `0.075667`
- Best-latency configuration: `n_search_direction=descending`
- Best latency reduction: `ndir-desc` -> `75.1913%`
- Best-reduction configuration: `n_search_direction=descending`
- Best final asynchronality: `ndir-desc` -> `0.048267`
- Best-asynchronality configuration: `n_search_direction=descending`
- Best asynchronality reduction: `ndir-desc` -> `63.3232%`
- Best asynchronality-reduction configuration: `n_search_direction=descending`

## uplink | convergence | uplink_payload_completion.yaml

- Scenario: `payload_completion`
- Seed: `3`
- Train seeds: `n/a`
- Decision axes: `n_kl search direction`
- Base configuration: `n_search_direction=descending`

| Variant   | Base | Status    | Highlighted configuration     | Final total latency | Delta vs base | Latency reduction % | Delta vs base | Final asynchronality | Delta vs base | Asynchronality reduction % | Delta vs base | Final avg SINR (dB) | Served bits | Log                                                                 |
| --------- | ---- | --------- | ----------------------------- | -------------------:| -------------:| -------------------:| -------------:| --------------------:| -------------:| --------------------------:| -------------:| -------------------:| -----------:| ------------------------------------------------------------------- |
| ndir-asc  |      | completed | n_search_direction=ascending  | 0.024667            | 0.000000      | 19.7397             | 0.0000        | 0.015333             | 0.000000      | 8.0000                     | 0.0000        | -3.2712             | 6000        | `ul_payload_completion__sweep_20260727_185736__conv__ndir-asc.log`  |
| ndir-desc | yes  | completed | n_search_direction=descending | 0.024667            | 0.000000      | 19.7397             | 0.0000        | 0.015333             | 0.000000      | 8.0000                     | 0.0000        | -3.3018             | 6000        | `ul_payload_completion__sweep_20260727_185736__conv__ndir-desc.log` |

- Best final total latency: `ndir-asc` -> `0.024667`
- Best-latency configuration: `n_search_direction=ascending`
- Best latency reduction: `ndir-asc` -> `19.7397%`
- Best-reduction configuration: `n_search_direction=ascending`
- Best final asynchronality: `ndir-asc` -> `0.015333`
- Best-asynchronality configuration: `n_search_direction=ascending`
- Best asynchronality reduction: `ndir-asc` -> `8.0000%`
- Best asynchronality-reduction configuration: `n_search_direction=ascending`

## uplink | monte_carlo | uplink_payload_completion.yaml

- Scenario: `payload_completion`
- Seed: `3`
- Train seeds: `1,2`
- Decision axes: `n_kl search direction`
- Base configuration: `n_search_direction=descending`

| Variant   | Base | Status    | Highlighted configuration     | Final total latency | Delta vs base | Latency reduction % | Delta vs base | Final asynchronality | Delta vs base | Asynchronality reduction % | Delta vs base | Final avg SINR (dB) | Served bits | Log                                                               |
| --------- | ---- | --------- | ----------------------------- | -------------------:| -------------:| -------------------:| -------------:| --------------------:| -------------:| --------------------------:| -------------:| -------------------:| -----------:| ----------------------------------------------------------------- |
| ndir-desc | yes  | completed | n_search_direction=descending | 0.024933            | 0.000000      | 18.8720             | 0.0000        | 0.015333             | 0.000000      | 8.0000                     | 0.0000        | -3.3269             | 6000        | `ul_payload_completion__sweep_20260727_185736__mc__ndir-desc.log` |
| ndir-asc  |      | completed | n_search_direction=ascending  | 0.027467            | 0.002533      | 10.6291             | -8.2430       | 0.016267             | 0.000933      | 2.4000                     | -5.6000       | -3.4335             | 6000        | `ul_payload_completion__sweep_20260727_185736__mc__ndir-asc.log`  |

- Best final total latency: `ndir-desc` -> `0.024933`
- Best-latency configuration: `n_search_direction=descending`
- Best latency reduction: `ndir-desc` -> `18.8720%`
- Best-reduction configuration: `n_search_direction=descending`
- Best final asynchronality: `ndir-desc` -> `0.015333`
- Best-asynchronality configuration: `n_search_direction=descending`
- Best asynchronality reduction: `ndir-desc` -> `8.0000%`
- Best asynchronality-reduction configuration: `n_search_direction=descending`