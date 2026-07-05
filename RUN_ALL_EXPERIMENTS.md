# Run All Experiments

This project now includes one batch runner:

`C:\All Codes\Taiwan_Internship\UL_UPLINK_DOWNLINK_MONTE_CARLO\run_all_experiments.py`

and one default manifest:

`C:\All Codes\Taiwan_Internship\UL_UPLINK_DOWNLINK_MONTE_CARLO\Experiment Configs\all_experiments.yaml`

## Default full run

From `C:\All Codes\Taiwan_Internship\UL_UPLINK_DOWNLINK_MONTE_CARLO`:

```powershell
python run_all_experiments.py
```

This launches all selected experiments in parallel, each in its own PowerShell window.

This default run starts:

- Uplink convergence on all `uplink_*.yaml` configs
- Uplink Monte Carlo on all `uplink_*.yaml` configs
- Downlink convergence on all `downlink_*.yaml` configs
- Downlink Monte Carlo on all `downlink_*.yaml` configs

For Monte Carlo runs, the runner now leaves training/test seed selection to each config unless you override it in the batch manifest.

## Dry run

```powershell
python run_all_experiments.py --dry_run
```

This prints the full command matrix without opening any new terminals.

## Useful filters

Only uplink:

```powershell
python run_all_experiments.py --links uplink
```

Only convergence:

```powershell
python run_all_experiments.py --methods convergence
```

Only one config stem:

```powershell
python run_all_experiments.py --configs uplink_payload_completion
```

Keep going after a failed run:

```powershell
python run_all_experiments.py --continue_on_error
```

Run everything sequentially in the current terminal instead:

```powershell
python run_all_experiments.py --launch_mode sequential
```

## Manifest format

The batch manifest is a YAML file with:

- `defaults`
  - shared fallback settings such as `seed`, `test_seed`, `train_seeds`, `quiet`
- `runs`
  - one run group per `(link, method)` combination

Example:

```yaml
defaults:
  seed: 3
  launch_mode: parallel_terminals

runs:
  - link: uplink
    method: convergence
    cfg_names: all
    seed: 3

  - link: downlink
    method: monte_carlo
    cfg_names:
      - downlink_payload_completion.yaml
    test_seed: 3
    num_train_seeds: 10
    quiet: true
```

Supported method names:

- `convergence`
- `monte_carlo`

Supported link names:

- `uplink`
- `downlink`

Supported launch modes:

- `parallel_terminals`
- `sequential`

`cfg_names: all` means:

- all `uplink_*.yaml` files for uplink runs
- all `downlink_*.yaml` files for downlink runs

For Monte Carlo entries in the manifest:

- `train_seeds` passes an explicit comma-separated seed list to the method.
- `num_train_seeds` passes `--num_train_seeds N`, so the method builds training seeds as `1..N` excluding `test_seed`.
- If neither is provided, the method uses the config defaults.

The runner uses the public method entry points:

- `Uplink/Methods/Convergence per epoch/main.py`
- `Uplink/Methods/Monte Carlo/main.py`
- `Downlink/Methods/Convergence per epoch/main.py`
- `Downlink/Methods/Monte Carlo/main.py`
