# UL Min Latency: Uplink + Downlink

This repository contains the cleaned uplink and downlink finite-blocklength latency-reduction experiments built around a shared experiment structure, shared result formatting, and comparable training/testing workflows.

The current project keeps the two main method families side by side:

- convergence-based online precoder optimization
- Monte Carlo train-and-test precoder-net optimization

The repository also includes shared configuration guides, result summaries, and decision-sweep utilities for benchmarking design choices across both links.

## What Is In This Repository

- `Uplink/`
  - uplink system model, optimization code, plotting, and method entry points
- `Downlink/`
  - downlink system model, optimization code, plotting, and method entry points
- `Experiment Configs/`
  - experiment YAML files plus parameter guides
- `Results/`
  - saved experiment outputs, plots, summaries, and comparison artifacts
- `docs/figures/`
  - a small set of README-friendly latency figures copied from representative saved runs

## Main Methods

### Convergence per epoch

This is the direct optimization baseline. The precoder is optimized online during the experiment run using the configured constrained objective and stopping rule.

Primary entry points:

- `Uplink/Methods/Convergence per epoch/main.py`
- `Downlink/Methods/Convergence per epoch/main.py`

### Monte Carlo

This is the offline train-and-test pipeline. A base dataset of channel episodes is built from training seeds, the precoder net is trained offline, and the trained model is then evaluated on a held-out test seed.

Primary entry points:

- `Uplink/Methods/Monte Carlo/main.py`
- `Downlink/Methods/Monte Carlo/main.py`

## Representative Payload-Completion Results

The table below summarizes the saved representative payload-completion runs currently used for the README figures. All latency reductions are reported against the random-precoder baseline stored in the corresponding result summary.

| Link | Method | Saved run | Initial total latency | Final total latency | Latency reduction |
| --- | --- | --- | ---: | ---: | ---: |
| Uplink | Convergence per epoch | `conv_net__payload_completion__s3` | `0.030733` | `0.024667` | `19.7397%` |
| Uplink | Monte Carlo | `mc__payload_completion__s3` | `0.030733` | `0.025200` | `18.0043%` |
| Downlink | Convergence per epoch | `conv_sum_user_net__payload_completion__s3` | `0.305000` | `0.035800` | `88.2623%` |
| Downlink | Monte Carlo | `mc_user__payload_completion__s3` | `0.305000` | `0.073533` | `75.8907%` |

## Latency Gain Over ZF And RZF

The next table compares the same representative payload-completion method runs against the saved ZF and RZF benchmark runs with the same link, scenario, and test seed. The gain is computed from final total latency as:

`gain = (benchmark_final_latency - method_final_latency) / benchmark_final_latency`

Positive values mean the method achieved lower final latency than the benchmark. Negative values mean the benchmark remained better.

| Link | Method | Method final latency | ZF final latency | Gain vs ZF | RZF final latency | Gain vs RZF |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Uplink | Convergence per epoch | `0.024667` | `0.025600` | `3.64%` | `0.025400` | `2.89%` |
| Uplink | Monte Carlo | `0.025200` | `0.025600` | `1.56%` | `0.025400` | `0.79%` |
| Downlink | Convergence per epoch | `0.035800` | `0.042467` | `15.70%` | `0.034200` | `-4.68%` |
| Downlink | Monte Carlo | `0.073533` | `0.042467` | `-73.15%` | `0.034200` | `-115.01%` |

## Latency Improvement Figures

### Uplink

| Convergence per epoch | Monte Carlo |
| --- | --- |
| ![Uplink convergence latency improvement](docs/figures/uplink_convergence_latency.png) | ![Uplink Monte Carlo latency improvement](docs/figures/uplink_monte_carlo_latency.png) |

### Downlink

| Convergence per epoch | Monte Carlo |
| --- | --- |
| ![Downlink convergence latency improvement](docs/figures/downlink_convergence_latency.png) | ![Downlink Monte Carlo latency improvement](docs/figures/downlink_monte_carlo_latency.png) |

## Quick Start

Typical runs use the payload-completion configs in `Experiment Configs/`.

### Uplink convergence

```powershell
python "Uplink/Methods/Convergence per epoch/main.py" --cfg_name uplink_payload_completion.yaml --seed 3
```

### Uplink Monte Carlo

```powershell
python "Uplink/Methods/Monte Carlo/main.py" --cfg_name uplink_payload_completion.yaml --num_train_seeds 3 --test_seed 3
```

### Downlink convergence

```powershell
python "Downlink/Methods/Convergence per epoch/main.py" --cfg_name downlink_payload_completion.yaml --seed 3
```

### Downlink Monte Carlo

```powershell
python "Downlink/Methods/Monte Carlo/main.py" --cfg_name downlink_payload_completion.yaml --num_train_seeds 3 --test_seed 3
```

## Result Layout

Representative outputs are stored under:

- `Results/Uplink/Method-Convergence per epoch/<experiment_name>/`
- `Results/Uplink/Method-Monte Carlo/<experiment_name>/`
- `Results/Downlink/Method-Convergence per epoch/<experiment_name>/`
- `Results/Downlink/Method-Monte Carlo/<experiment_name>/`

Typical contents include:

- `data/result.json`
- `data/summary.txt`
- `latency_asynchronality/`
- `link_quality/`
- `optimization_history/`
- `schedule_details/`
- `interference/` for downlink

Monte Carlo runs also save training-side artifacts such as:

- `training/data/train_artifact.pt`
- `training/data/training_dataset_summary.json`
- `training/data/post_training_summary.json`

## Additional Documentation

- [Run all experiments](RUN_ALL_EXPERIMENTS.md)
- [Run decision sweeps](RUN_DECISION_SWEEP.md)
- [Experiment config guides](Experiment%20Configs/README/README.md)
- [Uplink Monte Carlo config guide](Experiment%20Configs/README/uplink_monte_carlo.md)
- [Downlink Monte Carlo config guide](Experiment%20Configs/README/downlink_monte_carlo.md)
- [Downlink weighting modes](Experiment%20Configs/README/downlink_weighting_modes.md)

## Notes

- The README figures are copied from saved result folders into `docs/figures/` so they render cleanly on GitHub.
- The representative numbers above come from the existing saved summaries already present in `Results/`.
- The ZF and RZF comparison table uses the matching saved payload-completion benchmark runs already present in `Results/`.
- If you rerun experiments and want the README to reflect a newer benchmark set, update the copied figures and the summary table together.
