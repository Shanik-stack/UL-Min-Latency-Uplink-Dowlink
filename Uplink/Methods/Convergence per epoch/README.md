Convergence-per-epoch baseline

Canonical entry point:

`python "Uplink\\Methods\\Convergence per epoch\\main.py" --cfg_name uplink_payload_completion.yaml --seed 3`

This folder is the canonical entrypoint alias for the uplink convergence
baseline. It forwards to the legacy implementation so older scripts keep
working, while the experiment naming now uses `epoch` consistently.
