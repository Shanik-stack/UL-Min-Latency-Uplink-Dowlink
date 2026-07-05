# Run Decision Sweep

This project now includes a decision-combination sweep runner:

`C:\All Codes\Taiwan_Internship\UL_UPLINK_DOWNLINK_MONTE_CARLO\run_decision_sweep.py`

and one default manifest:

`C:\All Codes\Taiwan_Internship\UL_UPLINK_DOWNLINK_MONTE_CARLO\Experiment Configs\decision_sweep.yaml`

## What it does

This runner:

- starts from the canonical uplink and downlink configs,
- generates one temporary config for every allowed decision combination,
- runs the matching method entry point for each generated config,
- collects each run's `result.json`,
- writes comparison outputs across configurations.

The actual experiment results still go to the normal `Results\...` folders.

The sweep runner writes its control artifacts to:

`C:\All Codes\Taiwan_Internship\UL_UPLINK_DOWNLINK_MONTE_CARLO\Decision Sweep Runs\...`

including:

- generated configs,
- per-run logs,
- sweep plan files.

The detailed comparison results are written separately to:

`C:\All Codes\Taiwan_Internship\UL_UPLINK_DOWNLINK_MONTE_CARLO\Results\Decision Sweep Comparisons\...`

This comparison-results tree is where you inspect which configuration
combination performed best.

## Default run

From `C:\All Codes\Taiwan_Internship\UL_UPLINK_DOWNLINK_MONTE_CARLO`:

```powershell
python run_decision_sweep.py
```

By default this uses:

- all `uplink_*.yaml` base configs,
- all `downlink_*.yaml` base configs,
- the decision grids defined in `Experiment Configs\decision_sweep.yaml`,
- `max_parallel: auto`, which resolves to `min(planned_runs, logical_cpu_count)`.

## Dry run

```powershell
python run_decision_sweep.py --dry_run
```

This generates the derived configs and the run plan without executing the experiments.

## Useful filters

Only uplink:

```powershell
python run_decision_sweep.py --links uplink
```

Only Monte Carlo:

```powershell
python run_decision_sweep.py --methods monte_carlo
```

Only one base config:

```powershell
python run_decision_sweep.py --configs uplink_payload_completion
```

Run two child experiments at a time:

```powershell
python run_decision_sweep.py --max_parallel 2
```

Launch as many runs as the machine can schedule from logical CPU count:

```powershell
python run_decision_sweep.py --max_parallel auto
```

Launch every planned run immediately:

```powershell
python run_decision_sweep.py --max_parallel all
```

## Output files

Each sweep creates a timestamped folder under `Decision Sweep Runs`.

Each executed sweep also creates a timestamped folder under
`Results\Decision Sweep Comparisons`.

The main overall comparison outputs are:

- `overall\comparison_summary.csv`
  - one row per generated config run
- `overall\comparison_summary.json`
  - the same information in JSON form
- `overall\comparison_report.md`
  - grouped comparison tables with base-vs-variant deltas
- `overall\best_configurations_by_group.csv`
  - best final-latency and best latency-reduction combinations for each `(link, method, base_config)`
- `overall\best_completed_runs.csv`
  - all completed runs ranked so it is easy to find the strongest combinations
- `sweep_plan.json`
  - full generated run matrix, commands, and expected result paths

The detailed decision-sliced outputs are under:

- `by_decision\...`
  - one folder per decision variable
- `by_decision\<decision>\best_by_value.csv`
  - best combinations for each value of that decision
- `by_decision\<decision>\values\<value>\comparison_report.md`
  - a focused comparison where one decision value is fixed and the other decisions vary

This is the main place to answer questions like:

- "When `bs_shared_net` is fixed, which other choices give the best latency?"
- "For `augmented_lagrangian`, which objective or scope combination is best?"
- "Which complete combination is the strongest overall?"

## Base-vs-variant comparison

Inside each `(link, method, base_config)` group, the runner marks the one
generated combination that matches the original base config as:

- `is_base_configuration: true`

Then it computes deltas such as:

- `delta_final_total_latency_vs_base`
- `delta_total_latency_reduction_percent_vs_base`
- `delta_final_avg_snr_db_vs_base`
- `delta_final_avg_sinr_db_vs_base`
- `delta_total_served_bits_vs_base`

## Editing the decision grid

To add or remove decision combinations, edit:

`C:\All Codes\Taiwan_Internship\UL_UPLINK_DOWNLINK_MONTE_CARLO\Experiment Configs\decision_sweep.yaml`

The default manifest currently sweeps the main categorical decisions only:

- uplink rate model,
- plain vs augmented Lagrangian,
- convergence net vs direct-precoder updates,
- downlink objective mode,
- downlink per-user-net vs BS-shared-net,
- shared-BS fixed-target `n` handling mode.

If you want to include more public config decisions later, add them under the
relevant `decision_grids.<link>.<method>` section.

## Optional separate roots

If you want custom locations, you can set them independently:

```powershell
python run_decision_sweep.py --output_root "C:\path\to\control" --comparison_root "C:\path\to\comparison_results"
```

## Parallelism

The runner already launches experiments concurrently. The important change now is
that the default manifest no longer pins this to `1`.

- `auto`
  - uses `min(number_of_planned_runs, logical_cpu_count)`
- `all`
  - launches every planned run immediately
- integer value
  - launches up to that many runs at once

If you want the strongest default parallelism without manually setting anything,
leave the manifest at:

```yaml
defaults:
  max_parallel: auto
```
