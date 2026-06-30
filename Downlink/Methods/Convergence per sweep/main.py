from __future__ import annotations

import argparse
import sys
from pathlib import Path


METHOD_DIR = Path(__file__).resolve().parent
LINK_ROOT = METHOD_DIR.parents[1]
PROJECT_ROOT = LINK_ROOT.parent
for path in (LINK_ROOT, PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from config_loader import load_config
from experiment_runner import build_result_tag, run_downlink_experiment
from optimizer import resolve_convergence_objective_mode
from project_paths import build_downlink_result_dirs

METHOD_NAME = "convergence_per_epoch_baseline"
METHOD_LABEL = "Convergence per epoch"


def main() -> None:
    parser = argparse.ArgumentParser(description="Downlink online convergence baseline")
    parser.add_argument("--cfg_name", type=str, default="config_downlink_example.yaml", help="Path to a YAML config")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic random seed")
    parser.add_argument("--quiet", action="store_true", help="Reduce console logging")
    args = parser.parse_args()

    _, sim_params, run_meta = load_config(args.cfg_name)
    objective_mode = resolve_convergence_objective_mode(sim_params)
    result_tag = build_result_tag(
        METHOD_NAME,
        run_meta["cfg_stem"],
        int(args.seed),
        objective_mode=objective_mode,
        model_scope=sim_params.get("downlink_precoder_net_scope"),
    )
    output_dirs = build_downlink_result_dirs(METHOD_LABEL, result_tag)
    run_downlink_experiment(
        METHOD_NAME,
        args.cfg_name,
        args.seed,
        verbose=not args.quiet,
        output_root=output_dirs["testing_root"],
    )


if __name__ == "__main__":
    main()
