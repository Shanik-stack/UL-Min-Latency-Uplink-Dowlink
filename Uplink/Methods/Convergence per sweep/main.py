import argparse
import os
import sys
from pathlib import Path

import torch


METHOD_DIR = Path(__file__).resolve().parent
LINK_ROOT = METHOD_DIR.parents[1]
PROJECT_ROOT = LINK_ROOT.parent
for path in (METHOD_DIR, LINK_ROOT, PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from UplinkSystem import UplinkSystem
from config_loader import get_config
from experiment_utils import make_method_result_tag
from main_per_block import test_simulate
from plotting import (
    initialize_plot_globals,
    plot_F_vs_n_for_all_subblocks,
    plot_optimization_result,
    plot_optimization_result_summary_dict,
)
from project_paths import build_uplink_result_dirs
from optimizer import dynamic_subblocklength_precoder_training_baseline


def train_converge_in_each_sweep_baseline(
    cfg_name: str,
    train_seed: int = 0,
    *,
    do_plots: bool = True,
):
    system_params, sim_cfg = get_config(cfg_name)
    uplinksystem = UplinkSystem(system_params, seed=train_seed)

    post = dynamic_subblocklength_precoder_training_baseline(
        uplinksystem=uplinksystem,
        sim_cfg=dict(sim_cfg),
        channel_norm=True,
    )

    if do_plots:
        plot_optimization_result(post["all_user_block_results_train"], train=True)
        plot_optimization_result_summary_dict(post, train=True)
        plot_F_vs_n_for_all_subblocks(post)

    return post


def main():
    parser = argparse.ArgumentParser(description="Original per-block convergence baseline")
    parser.add_argument("--cfg_name", type=str, default="config_raw_T_exp1.yaml", help="Configuration file name or path")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Shared seed for both baseline optimization and testing; matches the downlink baseline CLI.",
    )
    parser.add_argument("--train_seed", "--train-seed", dest="train_seed", type=int, default=None)
    parser.add_argument("--test_seed", "--test-seed", dest="test_seed", type=int, default=None)
    parser.add_argument("--skip_test", action="store_true")
    args = parser.parse_args()

    shared_seed = int(args.seed) if args.seed is not None else 0
    train_seed = int(args.train_seed) if args.train_seed is not None else int(shared_seed)
    test_seed = int(args.test_seed) if args.test_seed is not None else int(shared_seed)

    result_tag = make_method_result_tag("converge_in_each_sweep_baseline", args.cfg_name, seed=test_seed)
    result_dirs = build_uplink_result_dirs("Convergence per sweep", result_tag)
    initialize_plot_globals(result_tag, result_dirs)

    post_training_data = train_converge_in_each_sweep_baseline(
        cfg_name=args.cfg_name,
        train_seed=train_seed,
        do_plots=True,
    )
    post_training_data["train_seed"] = int(train_seed)
    post_training_data["test_seed"] = int(test_seed)

    torch.save(post_training_data, os.path.join(result_dirs["train_data"], "train_artifact.pt"))

    if not args.skip_test:
        test_simulate(
            post_training_data_dict=post_training_data,
            cfg_name=args.cfg_name,
            test_seed=test_seed,
            do_plots=True,
            result_tag=result_tag,
            result_dirs=result_dirs,
        )


if __name__ == "__main__":
    main()
