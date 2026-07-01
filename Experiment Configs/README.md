# Experiment Configs

This folder contains the canonical ready-to-run YAML files for the cleaned
uplink/downlink experiment setup.

For the detailed parameter-by-parameter reference, read `PARAMETER_GUIDE.md`.

## Config Design

The configs now follow a simpler structure:

- `test`
  - Physical system dimensions, payload, SNR, symbol rate, and reliability.
- `simulation`
  - Optimization settings, one shared KKT stop budget `max_epochs`, Monte Carlo training-data settings, and scenario definition.

For convergence runs, `simulation.convergence_precoder_update_mode` selects
whether the solver updates a precoder net online (`precoder_net`) or directly
updates the complex precoder variables (`direct_precoder`).

The confusing legacy Monte Carlo names such as `precoder_net_train_*`,
`curriculum_*`, and `policy_train_*` are no longer used in the canonical YAML
files. The loaders still accept those old names as legacy aliases, but the
recommended names are the ones shown in the current files.

## Files

- `uplink_payload_completion.yaml`
  - Uplink payload-draining experiment.
  - Each user starts with `test.B[k]` total payload bits.
  - Transmission continues until each payload is drained.

- `uplink_fixed_block_targets.yaml`
  - Uplink fixed-bits-per-block experiment.
  - Each user tries to send `test.B[k]` bits in every block.
  - Unserved bits do not carry into the next block.

- `downlink_payload_completion.yaml`
  - Downlink payload-draining experiment.
  - Each user starts with `test.B[k]` total payload bits.

- `downlink_fixed_block_targets.yaml`
  - Downlink fixed-bits-per-block experiment.
  - Each user tries to send `test.B[k]` bits in every block over a fixed horizon.

## Running With Short Config Names

You only need the config filename. The loaders search:

1. The local method folder.
2. `UL_UPLINK_DOWNLINK_MONTE_CARLO/Experiment Configs`
3. `UL_UPLINK_DOWNLINK_MONTE_CARLO`
4. The current working directory.

## Example Commands

If your current directory is `C:\All Codes\Taiwan_Internship\UL_UPLINK_DOWNLINK_MONTE_CARLO`:

```powershell
python "Uplink\Methods\Convergence per epoch\main.py" --cfg_name uplink_payload_completion.yaml --seed 3
python "Uplink\Methods\Monte Carlo\main.py" --cfg_name uplink_payload_completion.yaml --train_seeds 0,1,2 --test_seed 3
python "Downlink\Methods\Convergence per epoch\main.py" --cfg_name downlink_payload_completion.yaml --seed 3
python "Downlink\Methods\Monte Carlo\main.py" --cfg_name downlink_payload_completion.yaml --train_seeds 0,1,2 --test_seed 3
```

## Scenario Meaning

- `payload_completion`
  - `test.B[k]` is the full payload for user `k`.
  - Later blocks may become tail blocks because only the remaining payload matters.

- `fixed_block_targets`
  - `test.B[k]` is the per-block target for user `k`.
  - Blocks are independent.
  - Unserved bits are recorded for that block only and do not carry forward.
