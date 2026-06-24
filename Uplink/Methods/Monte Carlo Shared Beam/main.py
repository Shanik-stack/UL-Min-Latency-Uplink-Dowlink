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
from experiment_report import build_precoder_net_result
from experiment_utils import make_method_result_tag, parse_seed_list, save_json, save_text
from plotting import (
    initialize_plot_globals,
    plot_F_vs_n_for_all_subblocks,
    plot_latency_and_asynchronality_from_json,
    plot_link_quality_from_json,
    plot_optimization_result,
    plot_optimization_result_summary_dict,
    plot_user_config,
)
from policy_optimizer import evaluate_shared_beam_precoder_net, train_shared_beam_precoder_net
from precoder_models import load_user_precoder_models
from project_paths import build_uplink_result_dirs
from utils import save_test_results_to_txt


def _build_dataset_summary_lines(dataset_summary: dict[str, object]) -> list[str]:
    lines = [
        "Uplink shared-beam training dataset summary",
        f"Total training scenarios: {int(dataset_summary.get('total_scenarios', 0))}",
        f"Total n evaluations across scenarios: {int(dataset_summary.get('total_n_evaluations', 0))}",
        f"Training scenarios by seed: {dataset_summary.get('scenarios_by_seed', {})}",
        f"Training scenarios by block: {dataset_summary.get('scenarios_by_block', {})}",
        f"Global n evaluations by n_kl: {dataset_summary.get('global_n_evaluations_by_n_kl', {})}",
        "",
        "Per-user training scenario details",
    ]
    for user_summary in dataset_summary.get("per_user", []):
        lines.append(
            " | ".join(
                [
                    f"User {int(user_summary.get('user', 0))}",
                    f"training_scenarios={int(user_summary.get('total_scenarios', 0))}",
                    f"scenarios_by_seed={user_summary.get('scenarios_by_seed', {})}",
                    f"scenarios_by_block={user_summary.get('scenarios_by_block', {})}",
                    f"n_evaluations_by_n_kl={user_summary.get('n_evaluations_by_n_kl', {})}",
                ]
            )
        )
    lines.extend(
        [
            "",
            "Terminology",
            "- training scenario: one (seed, user, block) item that reuses one beam across its n_kl grid",
            "- n evaluation: one rate evaluation of that shared beam at one candidate n_kl",
        ]
    )
    return lines


def _build_post_training_summary_lines(post_training_summary: dict[str, object]) -> list[str]:
    return [
        "Uplink shared-beam post-training summary",
        f"Train-eval seed: {int(post_training_summary.get('train_eval_seed', 0))}",
        f"Epochs requested: {int(post_training_summary.get('epochs_requested', 0))}",
        f"Per-user num epochs: {post_training_summary.get('per_user_num_epochs', [])}",
        f"Final avg loss: {float(post_training_summary.get('final_avg_loss', 0.0)):.6f}",
        f"Best avg loss: {float(post_training_summary.get('best_avg_loss', 0.0)):.6f}",
        f"Final avg user rate: {float(post_training_summary.get('final_avg_user_rate', 0.0)):.6f}",
        f"Best avg user rate: {float(post_training_summary.get('best_avg_user_rate', 0.0)):.6f}",
        f"Per-user final loss: {post_training_summary.get('per_user_final_loss', [])}",
        f"Per-user best loss: {post_training_summary.get('per_user_best_loss', [])}",
        f"Per-user final rate: {post_training_summary.get('per_user_final_rate', [])}",
        f"Per-user best rate: {post_training_summary.get('per_user_best_rate', [])}",
        f"Train-eval initial latency: {post_training_summary.get('train_eval_initial_latency', [])}",
        f"Train-eval final latency: {post_training_summary.get('train_eval_final_latency', [])}",
        f"Train-eval initial blocks per user: {post_training_summary.get('train_eval_initial_blocks_per_user', [])}",
        f"Train-eval final blocks per user: {post_training_summary.get('train_eval_blocks_per_user', [])}",
        f"Train-eval initial total n per user: {post_training_summary.get('train_eval_initial_total_n_per_user', [])}",
        f"Train-eval final total n per user: {post_training_summary.get('train_eval_total_n_per_user', [])}",
        f"Train-eval initial served bits per user: {post_training_summary.get('train_eval_initial_served_bits_per_user', [])}",
        f"Train-eval final served bits per user: {post_training_summary.get('train_eval_served_bits_per_user', [])}",
        (
            "Train-eval total latency reduction (%): "
            f"{float(post_training_summary.get('train_eval_total_latency_reduction_percent', 0.0)):.4f}"
        ),
        "Train-eval initial selected n_kl summary:",
        f"{post_training_summary.get('train_eval_initial_selected_n_kl_summary', {})}",
        "Train-eval selected n_kl summary:",
        f"{post_training_summary.get('train_eval_selected_n_kl_summary', {})}",
        "",
        "Terminology",
        "- train-eval: evaluation of the trained shared-beam precoder nets on the first training seed",
    ]


def _build_summary_lines(result: dict[str, object]) -> list[str]:
    metrics = result["summary_metrics"]
    assert isinstance(metrics, dict)
    dataset_summary = result.get("training_dataset_summary", {})
    post_training_summary = result.get("post_training_summary", {})
    lines = [
        "Uplink optimizer summary",
        f"Method: {result.get('method_name', 'unknown')}",
        f"Config: {result.get('cfg_path', 'unknown')}",
        f"Test seed: {int(result.get('seed', 0))}",
        f"Train seeds: {result.get('train_seeds', [])}",
        f"Training scenario counts per user: {result.get('training_sample_counts_per_user', result.get('training_dataset_sizes', []))}",
        f"Training dataset total scenarios: {int(dataset_summary.get('total_scenarios', 0)) if isinstance(dataset_summary, dict) else 0}",
        f"Precoder parameterization: {result.get('precoder_parameterization', 'unknown')}",
        f"Training objective: {result.get('training_objective', 'unknown')}",
        f"Initial schedule source: {result.get('initial_schedule_source', 'unknown')}",
        "",
    ]
    if isinstance(post_training_summary, dict) and len(post_training_summary) > 0:
        lines.extend(
            [
                "Training summary",
                f"Final avg loss: {float(post_training_summary.get('final_avg_loss', 0.0)):.6f}",
                f"Best avg loss: {float(post_training_summary.get('best_avg_loss', 0.0)):.6f}",
                f"Final avg user rate: {float(post_training_summary.get('final_avg_user_rate', 0.0)):.6f}",
                f"Best avg user rate: {float(post_training_summary.get('best_avg_user_rate', 0.0)):.6f}",
                f"Per-user final loss: {post_training_summary.get('per_user_final_loss', [])}",
                f"Per-user final rate: {post_training_summary.get('per_user_final_rate', [])}",
                f"Train-eval initial blocks per user: {post_training_summary.get('train_eval_initial_blocks_per_user', [])}",
                f"Train-eval final blocks per user: {post_training_summary.get('train_eval_blocks_per_user', [])}",
                f"Train-eval initial total n per user: {post_training_summary.get('train_eval_initial_total_n_per_user', [])}",
                f"Train-eval final total n per user: {post_training_summary.get('train_eval_total_n_per_user', [])}",
                (
                    "Train-eval total latency reduction (%): "
                    f"{float(post_training_summary.get('train_eval_total_latency_reduction_percent', 0.0)):.4f}"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "Testing summary",
            "",
            "Latency summary",
            f"Initial total latency: {metrics['initial_total_latency']:.6f}",
            f"Final total latency: {metrics['final_total_latency']:.6f}",
            f"Total latency reduction (%): {metrics['total_latency_reduction_percent']:.4f}",
            f"Initial avg latency: {metrics['initial_avg_latency']:.6f}",
            f"Final avg latency: {metrics['final_avg_latency']:.6f}",
            f"Initial min/max latency: {metrics['initial_min_latency']:.6f} / {metrics['initial_max_latency']:.6f}",
            f"Final min/max latency: {metrics['final_min_latency']:.6f} / {metrics['final_max_latency']:.6f}",
            "",
            "Asynchronality summary",
            f"Initial asynchronality sum: {metrics['initial_asynchronality_sum']:.6f}",
            f"Final asynchronality sum: {metrics['final_asynchronality_sum']:.6f}",
            f"Asynchronality reduction (%): {metrics['asynchronality_reduction_percent']:.4f}",
            "",
            "Link quality summary",
            f"Initial avg SNR (dB): {metrics['initial_avg_snr_db']:.4f}",
            f"Final avg SNR (dB): {metrics['final_avg_snr_db']:.4f}",
            f"Initial avg SINR (dB): {metrics['initial_avg_sinr_db']:.4f}",
            f"Final avg SINR (dB): {metrics['final_avg_sinr_db']:.4f}",
            "",
            "Per-user details",
        ]
    )
    for row in metrics["per_user_summary"]:
        lines.append(
            " | ".join(
                [
                    f"User {row['user']}",
                    f"init_lat={row['initial_latency']:.6f}",
                    f"final_lat={row['final_latency']:.6f}",
                    f"lat_red={row['latency_reduction_percent']:.4f}%",
                    f"init_snr={row['initial_snr_db']:.4f} dB",
                    f"final_snr={row['final_snr_db']:.4f} dB",
                    f"init_sinr={row['initial_sinr_db']:.4f} dB",
                    f"final_sinr={row['final_sinr_db']:.4f} dB",
                    f"blocks={row['blocks']}",
                    f"total_n={row['total_n']}",
                    f"served_bits={row['served_bits']}",
                ]
            )
        )
    if metrics["initial_asynchronality_pairs"]:
        lines.extend(["", "Per-pair asynchronality"])
        for init_pair, final_pair in zip(metrics["initial_asynchronality_pairs"], metrics["final_asynchronality_pairs"]):
            lines.append(
                " | ".join(
                    [
                        f"Users {init_pair['user_i']}-{init_pair['user_j']}",
                        f"initial_diff={init_pair['abs_latency_diff']:.6f}",
                        f"final_diff={final_pair['abs_latency_diff']:.6f}",
                    ]
                )
            )
    lines.extend(
        [
            "",
            "Terminology",
            "- training scenario: one (seed, user, block) item that reuses one beam across its n_kl grid",
            "- initial schedule: the random-precoder baseline used for the before-optimization uplink latency",
        ]
    )
    return lines


def _run_shared_beam_test(
    train_artifact: dict[str, object],
    cfg_name: str,
    test_seed: int,
    *,
    do_plots: bool,
    result_dirs: dict[str, str],
    train_seeds: list[int],
) -> dict[str, object]:
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
    post_test = evaluate_shared_beam_precoder_net(
        test_uplinksystem,
        user_models,
        sim_cfg,
        method_name="monte_carlo_shared_beam_test",
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
        method_name="monte_carlo_shared_beam_train_test",
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
    save_text(_build_summary_lines(result), os.path.join(result_dirs["test_data"], "summary.txt"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Monte Carlo shared-beam train/test")
    parser.add_argument("--cfg_name", type=str, default="config_raw_T_exp1.yaml", help="Configuration file name or path")
    parser.add_argument("--train_seeds", type=str, default="0,1,2")
    parser.add_argument("--test_seed", type=int, default=3)
    parser.add_argument("--precoder_net_epochs", "--precoder_epochs", dest="precoder_net_epochs", type=int, default=40)
    parser.add_argument("--precoder_net_batch_size", "--precoder_batch_size", dest="precoder_net_batch_size", type=int, default=32)
    parser.add_argument("--precoder_net_lr", "--precoder_lr", dest="precoder_net_lr", type=float, default=1e-3)
    parser.add_argument("--skip_test", action="store_true")
    args = parser.parse_args()

    train_seeds = parse_seed_list(args.train_seeds)
    result_tag = make_method_result_tag("monte_carlo_shared_beam_train_test", args.cfg_name, seed=args.test_seed)
    result_dirs = build_uplink_result_dirs("Monte Carlo Shared Beam", result_tag)
    initialize_plot_globals(result_tag, result_dirs)

    train_artifact = train_shared_beam_precoder_net(
        cfg_name=args.cfg_name,
        train_seeds=train_seeds,
        epochs=args.precoder_net_epochs,
        batch_size=args.precoder_net_batch_size,
        lr=args.precoder_net_lr,
    )
    train_artifact["cfg_path"] = _resolve_config_path(args.cfg_name)
    train_artifact["method_name"] = "monte_carlo_shared_beam_train_test"
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
        _build_dataset_summary_lines(train_artifact.get("training_dataset_summary", {})),
        os.path.join(result_dirs["train_data"], "training_dataset_summary.txt"),
    )
    save_json(
        train_artifact.get("post_training_summary", {}),
        os.path.join(result_dirs["train_data"], "post_training_summary.json"),
    )
    save_text(
        _build_post_training_summary_lines(train_artifact.get("post_training_summary", {})),
        os.path.join(result_dirs["train_data"], "post_training_summary.txt"),
    )

    if not args.skip_test:
        _run_shared_beam_test(
            train_artifact=train_artifact,
            cfg_name=args.cfg_name,
            test_seed=args.test_seed,
            do_plots=True,
            result_dirs=result_dirs,
            train_seeds=train_seeds,
        )
    print(f"Saved uplink shared-beam results to: {result_dirs['experiment_root']}")


if __name__ == "__main__":
    main()
