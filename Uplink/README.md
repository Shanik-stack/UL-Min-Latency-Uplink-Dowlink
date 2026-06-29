Uplink

This uplink subset keeps only:

- `Methods/Convergence per epoch`: the original online convergence baseline
- `Methods/Monte Carlo`: the offline precoder-net path trained from baseline trajectories
- `Methods/Monte Carlo Shared Beam`: the offline shared-beam Monte Carlo variant

Shared files in this folder are the minimum common runtime needed by those two methods.

Shared experiment conventions in this cleaned folder:

- entry points use the shared result-tag helper from `experiment_utils.py`
- learned methods split results into `training/` and `testing/` subfolders inside each experiment folder
- the `Methods/Convergence per epoch` baseline writes one flat convergence result bundle with no training/testing split
- testing summaries live under `testing/data/`
- training artifacts and training summaries live under `training/data/`
