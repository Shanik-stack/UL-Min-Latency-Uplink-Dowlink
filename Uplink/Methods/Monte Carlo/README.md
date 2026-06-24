Monte Carlo precoder-net train/test

This folder implements the direct offline training path:

- build a raw Monte Carlo dataset of channel blocks and candidate `n_kl` values
- train one user-specific precoder net `F_{k,l} = f_{theta_k}(H_{k,l}, n_{k,l}, noise/interference, epsilon)`
- optimize the precoder net directly on the dataset with a Lagrangian loss
- enforce a fixed minimum transmitted-bit feasibility target during training with `min_bits_required = 1` by default
- test with only precoder-net forward passes plus the usual outer linear search over `n_kl`

Notes:

- no expert-label collection or MSE imitation is used
- `B_rem` stays in the scheduler; the training loss uses `min_bits_required / n_kl`, not a proportional `B_target`

Entry point:

`python "Uplink\\Methods\\Monte Carlo\\main.py" --cfg_name config_raw_T_exp1.yaml --train_seeds 0,1,2 --test_seed 3`

Results are written under:

`Results\\Uplink\\Monte Carlo\\<experiment_name>`

Standard files in `training/data/`:

- `train_artifact.pt`
- `training_dataset_summary.json`
- `training_dataset_summary.txt`
- `post_training_summary.json`
- `post_training_summary.txt`

Standard files in `testing/data/`:

- `result.json`
- `summary.txt`
