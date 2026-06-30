Monte Carlo precoder-net train/test

This folder adds the direct offline downlink path:

- build a raw Monte Carlo dataset of channel blocks and block-level service targets
- train either one precoder net per user or one shared BS precoder net, selected by `simulation.downlink_precoder_net_scope`
- optimize all user precoder nets jointly with a Lagrangian objective based on sum finite-blocklength rate
- reduce the current `n_{k,l}` frontier during training only when the joint precoder nets make the 1-bit feasibility condition hold
- test with precoder-net forward passes plus the same outer linear search over `n_kl`

Notes:

- no expert-label collection or MSE imitation is used
- `per_user_nets` means one separate model instance for each user beam
- `bs_shared_net` means one shared model instance is reused across all downlink users and the full BS precoder is assembled from those user-specific forward passes
- the dataset stays simple; smaller `n_{k,l}` values are introduced by rollout frontier search during training rather than by a separate expert-label pass

The original online convergence baseline is kept separately:

`python "Downlink\\Methods\\Convergence per epoch\\main.py" --cfg_name config_downlink_example.yaml --seed 0`

Offline precoder-net entry point:

`python "Downlink\\Methods\\Monte Carlo\\main.py" --cfg_name config_downlink_example.yaml --train_seeds 0,1,2 --test_seed 3`

Results are written under:

`Results\\Downlink\\Monte Carlo\\<experiment_name>`

Standard files in `training/data/`:

- `train_artifact.pt`
- `training_dataset_summary.json`
- `training_dataset_summary.txt`
- `post_training_summary.json`
- `post_training_summary.txt`

Standard files in `testing/data/`:

- `result.json`
- `summary.txt`

Saved training history now includes:

- per-user Lagrangian traces
- epoch-wise sum FBL rate
- epoch-wise per-user FBL rate
- epoch-wise rate-violation and power-violation traces
