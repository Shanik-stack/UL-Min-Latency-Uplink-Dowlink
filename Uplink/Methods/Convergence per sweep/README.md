Converge-in-each-sweep baseline

This keeps the original slow expert style:

- each user/block gets a fresh precoder network
- that block is optimized to convergence across the local `n_kl` search
- no precoder parameters are shared across blocks

Entry point:

`python "Uplink\\Methods\\Convergence per sweep\\main.py" --cfg_name config_raw_T_exp1.yaml --seed 3`

If you want different seeds for baseline optimization and testing, you can still use
`--train_seed` and `--test_seed`.

Results are written under:

`Results\\Uplink\\Convergence per sweep\\<experiment_name>`

Standard files in `training/data/`:

- `train_artifact.pt`

Standard files in `testing/data/`:

- `result.json`
- `summary.txt`
