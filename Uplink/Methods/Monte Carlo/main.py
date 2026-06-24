import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch


METHOD_DIR = Path(__file__).resolve().parent
LINK_ROOT = METHOD_DIR.parents[1]
PROJECT_ROOT = LINK_ROOT.parent
for path in (METHOD_DIR, LINK_ROOT, PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from UplinkSystem import UplinkSystem
from advanced_methods_common import estimate_initial_random_precoder_schedule
from config_loader import _resolve_config_path, get_config
from experiment_utils import make_method_result_tag, parse_seed_list, save_json, save_text
from experiment_report import (
    build_post_training_summary_lines,
    build_precoder_net_result,
    build_summary_lines,
    build_training_dataset_summary_lines,
)
from policy_optimizer import evaluate_blocklength_precoder_net, train_blocklength_aware_precoder_net
from plotting import (
    initialize_plot_globals,
    plot_F_vs_n_for_all_subblocks,
    plot_optimization_result,
    plot_optimization_result_summary_dict,
    plot_latency_and_asynchronality_from_json,
    plot_link_quality_from_json,
    plot_user_config,
)
from precoder_models import load_user_precoder_models
from project_paths import build_uplink_result_dirs
from utils import save_test_results_to_txt


def _run_precoder_net_test(
    train_artifact: dict,
    cfg_name: str,
    test_seed: int,
    *,
    do_plots: bool,
    result_dirs: dict[str, str],
    train_seeds: list[int],
):
    system_params, sim_cfg = get_config(cfg_name)
    initial_baseline = estimate_initial_random_precoder_schedule(
        system_params,
        sim_cfg,
        seed=int(test_seed),
    )
    test_uplinksystem = UplinkSystem(system_params, seed=int(test_seed))
    _, initial_snr_db = test_uplinksystem.get_SNR()
    _, initial_sinr_db = test_uplinksystem.get_SINR()

    plot_params = dict(system_params)
    plot_params["initial_bits_per_symbol"] = np.asarray(initial_baseline["initial_bits_per_symbol"], dtype=float)
    plot_user_config(
        plot_params,
        extra_params={
            "measured_snr_db_k": np.asarray(initial_snr_db),
            "measured_sinr_db_k": np.asarray(initial_sinr_db),
        },
    )

    initial_Rfbl = [np.array(v, copy=True) for v in initial_baseline["initial_R_fbl"]]
    initial_latency = list(initial_baseline["initial_latency"])
    initial_bits_per_symbol = list(initial_baseline["initial_bits_per_symbol"])
    initial_bits_per_symbol_by_block = [
        list(values) for values in initial_baseline["initial_bits_per_symbol_by_block"]
    ]
    initial_n = list(initial_baseline["initial_n"])
    initial_n_kl = [list(values) for values in initial_baseline["initial_n_kl"]]
    initial_B_kl = [list(values) for values in initial_baseline["initial_B_kl"]]

    user_models = load_user_precoder_models(
        train_artifact["user_model_specs"],
        train_artifact["user_model_states"],
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    )
    post_test = evaluate_blocklength_precoder_net(
        uplinksystem=test_uplinksystem,
        user_models=user_models,
        sim_cfg=sim_cfg,
        method_name="monte_carlo_precoder_net_test",
    )

    test_data_dict = {
        "L_out_test": post_test["L_out"],
        "n_star_test": post_test["n_star"],
        "F_star_test": post_test["F_star"],
        "R_star_test": post_test["R_star"],
        "all_user_block_results_test": post_test["all_user_block_results_train"],
        "B_used_star_test": post_test["B_used_star"],
        "B_kl_star_test": post_test["B_kl_star"],
        "precoder_parameterization": train_artifact["precoder_parameterization"],
        "user_model_specs": train_artifact["user_model_specs"],
        "user_model_states": train_artifact["user_model_states"],
    }

    save_test_results_to_txt(
        test_uplinksystem=test_uplinksystem,
        test_data_dict=test_data_dict,
        initial_Rfbl=initial_Rfbl,
        initial_n_kl=initial_n_kl,
        initial_n=initial_n,
        initial_latency=initial_latency,
        initial_snr_db=initial_snr_db,
        initial_sinr_db=initial_sinr_db,
        save_dir=result_dirs["test_data"],
        filename="test_results.txt",
        initial_bits_per_symbol=initial_bits_per_symbol,
        initial_B_kl=initial_B_kl,
        initial_bits_per_symbol_by_block=initial_bits_per_symbol_by_block,
    )

    if do_plots:
        plot_optimization_result(test_data_dict["all_user_block_results_test"], train=False)
        plot_optimization_result_summary_dict(
            {"n_star": test_data_dict["n_star_test"], "R_star": test_data_dict["R_star_test"]},
            train=False,
        )
        if int(test_uplinksystem.K) > 1:
            plot_latency_and_asynchronality_from_json(
                json_path=os.path.join(result_dirs["test_data"], "test_results.json"),
                save_dir=result_dirs["latency_asynchronality"],
                prefix="test",
            )
        plot_link_quality_from_json(
            json_path=os.path.join(result_dirs["test_data"], "test_results.json"),
            save_dir=result_dirs["link_quality"],
            prefix="test",
        )

    result = build_precoder_net_result(
        test_uplinksystem,
        test_data_dict,
        method_name="monte_carlo_precoder_net_train_test",
        cfg_path=_resolve_config_path(cfg_name),
        test_seed=int(test_seed),
        train_seeds=train_seeds,
        train_artifact=train_artifact,
        initial_R_fbl=initial_Rfbl,
        initial_n_kl=initial_n_kl,
        initial_n=initial_n,
        initial_latency=initial_latency,
        initial_snr_db=initial_snr_db,
        initial_sinr_db=initial_sinr_db,
        initial_bits_per_symbol=initial_bits_per_symbol,
        initial_B_kl=initial_B_kl,
        initial_bits_per_symbol_by_block=initial_bits_per_symbol_by_block,
    )
    save_json(result, os.path.join(result_dirs["test_data"], "result.json"))
    save_text(build_summary_lines(result), os.path.join(result_dirs["test_data"], "summary.txt"))
    return result


def main():
    parser = argparse.ArgumentParser(description="Offline Monte Carlo precoder-net train/test")
    parser.add_argument("--cfg_name", type=str, default="config_raw_T_exp1.yaml", help="Configuration file name or path")
    parser.add_argument("--train_seeds", type=str, default="0,1,2")
    parser.add_argument("--test_seed", type=int, default=3)
    parser.add_argument("--precoder_net_epochs", "--precoder_epochs", "--policy_epochs", dest="precoder_net_epochs", type=int, default=40)
    parser.add_argument("--precoder_net_batch_size", "--precoder_batch_size", "--policy_batch_size", dest="precoder_net_batch_size", type=int, default=32)
    parser.add_argument("--precoder_net_lr", "--precoder_lr", "--policy_lr", dest="precoder_net_lr", type=float, default=1e-3)
    parser.add_argument("--skip_test", action="store_true")
    args = parser.parse_args()

    train_seeds = parse_seed_list(args.train_seeds)
    result_tag = make_method_result_tag("monte_carlo_precoder_net_train_test", args.cfg_name, seed=args.test_seed)
    result_dirs = build_uplink_result_dirs("Monte Carlo", result_tag)
    initialize_plot_globals(result_tag, result_dirs)

    train_artifact = train_blocklength_aware_precoder_net(
        cfg_name=args.cfg_name,
        train_seeds=train_seeds,
        epochs=args.precoder_net_epochs,
        batch_size=args.precoder_net_batch_size,
        lr=args.precoder_net_lr,
    )
    train_artifact["cfg_path"] = _resolve_config_path(args.cfg_name)
    train_artifact["method_name"] = "monte_carlo_precoder_net_train_test"
    train_artifact["test_seed"] = int(args.test_seed)

    plot_optimization_result(train_artifact["all_user_block_results_train"], train=True)
    plot_optimization_result_summary_dict(train_artifact, train=True)
    plot_F_vs_n_for_all_subblocks(train_artifact)

    torch.save(train_artifact, os.path.join(result_dirs["train_data"], "train_artifact.pt"))
    save_json(
        train_artifact.get("training_dataset_summary", {}),
        os.path.join(result_dirs["train_data"], "training_dataset_summary.json"),
    )
    save_text(
        build_training_dataset_summary_lines(train_artifact.get("training_dataset_summary", {})),
        os.path.join(result_dirs["train_data"], "training_dataset_summary.txt"),
    )
    save_json(
        train_artifact.get("post_training_summary", {}),
        os.path.join(result_dirs["train_data"], "post_training_summary.json"),
    )
    save_text(
        build_post_training_summary_lines(train_artifact.get("post_training_summary", {})),
        os.path.join(result_dirs["train_data"], "post_training_summary.txt"),
    )

    if not args.skip_test:
        _run_precoder_net_test(
            train_artifact=train_artifact,
            cfg_name=args.cfg_name,
            test_seed=args.test_seed,
            do_plots=True,
            result_dirs=result_dirs,
            train_seeds=train_seeds,
        )
    print(f"Saved uplink precoder-net results to: {result_dirs['experiment_root']}")


if __name__ == "__main__":
    main()
