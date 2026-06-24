Monte Carlo shared-beam precoder-net train/test

This folder adds the second offline uplink Monte Carlo method:

- build a raw Monte Carlo dataset of channel blocks and candidate `n_kl` values
- train one user precoder net `F_{k,l} = f_{\theta_k}(H_{k,l}, noise/interference, epsilon)`
- predict one shared beam for each training scenario
- reuse that same predicted beam across the scenario's candidate `n_kl` grid
- optimize the net directly on average finite-blocklength rate over that shared-beam `n_kl` grid
- test with one shared forward pass per block, followed by the usual outer search over `n_kl`

Notes:

- no expert-label collection or MSE imitation is used
- unlike `Methods/Monte Carlo`, this method does not generate a fresh beam for each candidate `n_kl`

The original online convergence baseline is kept separately:

`python "Uplink\\Methods\\Convergence per sweep\\main.py" --cfg_name config_raw_T_exp1.yaml --seed 0`

The original per-`n_kl` Monte Carlo method is kept separately:

`python "Uplink\\Methods\\Monte Carlo\\main.py" --cfg_name config_raw_T_exp1.yaml --train_seeds 0,1,2 --test_seed 3`

Shared-beam entry point:

`python "Uplink\\Methods\\Monte Carlo Shared Beam\\main.py" --cfg_name config_raw_T_exp1.yaml --train_seeds 0,1,2 --test_seed 3`

Results are written under:

`Results\\Uplink\\Monte Carlo Shared Beam\\<experiment_name>`
