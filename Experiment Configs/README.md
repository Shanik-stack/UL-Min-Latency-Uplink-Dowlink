# Experiment Configs

This folder contains ready-to-run experiment configurations for the two scenario families used by the unified uplink/downlink Monte Carlo setup.

For a detailed explanation of every config parameter and how it affects the simulation, see `PARAMETER_GUIDE.md`.

The provided experiment YAML files are intended to define the canonical parameters explicitly instead of relying on hidden loader defaults.

## Files

- `uplink_payload_completion.yaml`
  - Uplink payload-draining scenario.
  - Each user starts with a payload budget from `test.B`.
  - The simulation continues until the payload is fully transmitted.

- `uplink_fixed_block_targets.yaml`
  - Uplink fixed-bits-per-block scenario.
  - Each user attempts to transmit `test.B[k]` bits in every block.
  - The current default uses `num_blocks: 10`, with per-user block targets taken directly from `B`.

- `downlink_payload_completion.yaml`
  - Downlink payload-draining scenario.
  - Each user starts with a payload budget from `test.B`.
  - The simulation continues until the payload is fully transmitted.

- `downlink_fixed_block_targets.yaml`
  - Downlink fixed-bits-per-block scenario.
  - Each user attempts to transmit `test.B[k]` bits in every block.
  - The current default uses `num_blocks: 10`, with per-user block targets taken directly from `B`.

## Running With Short Config Names

You do not need to pass the full config path. The uplink and downlink config loaders now automatically search:

1. Their local method folder.
2. `UL_UPLINK_DOWNLINK_MONTE_CARLO/Experiment Configs`
3. `UL_UPLINK_DOWNLINK_MONTE_CARLO`
4. The current working directory.

That means you can pass only the config filename, but the script path still depends on your current working directory.

### If your current directory is `C:\All Codes\Taiwan_Internship`

Use the project-prefixed script path:

```powershell
python "UL_UPLINK_DOWNLINK_MONTE_CARLO\Uplink\Methods\Monte Carlo\main.py" --cfg_name uplink_payload_completion.yaml --train_seeds 0,1,2 --test_seed 3
```

```powershell
python "UL_UPLINK_DOWNLINK_MONTE_CARLO\Uplink\Methods\Monte Carlo\main.py" --cfg_name uplink_fixed_block_targets.yaml --train_seeds 0,1,2 --test_seed 3
```

```powershell
python "UL_UPLINK_DOWNLINK_MONTE_CARLO\Downlink\Methods\Monte Carlo\main.py" --cfg_name downlink_payload_completion.yaml --train_seeds 0,1,2 --test_seed 3
```

```powershell
python "UL_UPLINK_DOWNLINK_MONTE_CARLO\Downlink\Methods\Monte Carlo\main.py" --cfg_name downlink_fixed_block_targets.yaml --train_seeds 0,1,2 --test_seed 3
```

### If your current directory is `C:\All Codes\Taiwan_Internship\UL_UPLINK_DOWNLINK_MONTE_CARLO`

Use the shorter script path:

### Uplink payload

```powershell
python "Uplink\Methods\Monte Carlo\main.py" --cfg_name uplink_payload_completion.yaml --train_seeds 0,1,2 --test_seed 3
```

### Uplink fixed block targets

```powershell
python "Uplink\Methods\Monte Carlo\main.py" --cfg_name uplink_fixed_block_targets.yaml --train_seeds 0,1,2 --test_seed 3
```

### Downlink payload

```powershell
python "Downlink\Methods\Monte Carlo\main.py" --cfg_name downlink_payload_completion.yaml --train_seeds 0,1,2 --test_seed 3
```

### Downlink fixed block targets

```powershell
python "Downlink\Methods\Monte Carlo\main.py" --cfg_name downlink_fixed_block_targets.yaml --train_seeds 0,1,2 --test_seed 3
```

## Scenario Meaning

- `payload_completion`
  - Total payload is fixed per user.
  - The final block may act like a tail block because only the remaining payload needs to be sent.

- `fixed_block_targets`
  - A fixed positive bit target is assigned to every block for every user.
  - In the canonical `generation_mode: constant` setup, that per-block target is `test.B[k]`.
  - This models continuous per-block transmission demand over a fixed block horizon.
