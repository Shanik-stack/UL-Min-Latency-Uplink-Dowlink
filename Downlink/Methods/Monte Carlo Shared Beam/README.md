Monte Carlo shared-beam precoder-net train/test

This folder adds the second offline downlink Monte Carlo method:

- build a raw Monte Carlo dataset of block contexts and active-user masks
- train one user precoder net `F_{k,l} = f_{\theta_k}(H_l, active_mask, noise/interference, epsilon)`
- predict one shared beam per active user for each training case
- reuse that same predicted beam across the candidate `n_kl` grid for the case
- optimize the nets directly on average sum finite-blocklength rate over that shared-beam `n_kl` grid
- test with one shared forward pass per active block, followed by the usual outer search over `n_kl`

Notes:

- no expert-label collection or MSE imitation is used
- unlike `Methods/Monte Carlo`, this method does not generate a fresh beam for each candidate `n_kl`
- the active-user mask is part of the input because the interference pattern depends on who is scheduled together

The original online safe sweep is kept separately as the baseline:

`python "Downlink\\Methods\\Convergence per sweep\\main.py" --cfg_name config_downlink_example.yaml --seed 0`

The original per-`n_kl` Monte Carlo method is kept separately:

`python "Downlink\\Methods\\Monte Carlo\\main.py" --cfg_name config_downlink_example.yaml --train_seeds 0,1,2 --test_seed 3`

Shared-beam entry point:

`python "Downlink\\Methods\\Monte Carlo Shared Beam\\main.py" --cfg_name config_downlink_example.yaml --train_seeds 0,1,2 --test_seed 3`

Results are written under:

`Results\\Downlink\\Monte Carlo Shared Beam\\<experiment_name>`
