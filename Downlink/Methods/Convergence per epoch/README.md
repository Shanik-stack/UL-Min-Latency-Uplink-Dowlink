Convergence-per-epoch baseline

Canonical entry point:

`python "Downlink\\Methods\\Convergence per epoch\\main.py" --cfg_name downlink_payload_completion.yaml --seed 3`

This folder is the canonical entrypoint alias for the downlink convergence
baseline. It forwards to the legacy implementation so older scripts keep
working, while the experiment naming now uses `epoch` consistently.
