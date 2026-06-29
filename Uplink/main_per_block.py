import os

import numpy as np

from advanced_methods_common import estimate_initial_random_precoder_schedule
from UplinkSystem import UplinkSystem
from Optimizer_per_block import dynamic_subblocklength_precoder_testing
from config_loader import _resolve_config_path, get_config
from experiment_report import build_precoder_net_result, build_summary_lines
from experiment_utils import save_json, save_text
from plotting import (
    plot_latency_and_asynchronality_from_json,
    plot_link_quality_from_json,
    plot_optimization_result,
    plot_optimization_result_summary_dict,
    plot_user_config,
)
from utils import save_test_results_to_txt


def test_simulate(
    post_training_data_dict,
    cfg_name,
    test_seed=1,
    do_plots=True,
    result_tag=None,
    result_dirs=None,
):
    system_test_params, simulation_test_params = get_config(cfg_name)
    save_tag = result_tag or cfg_name

    print("Starting Test")
    print("SYSTEM PARAMS")
    print(system_test_params)
    print("SIMULATION PARAMS")
    print(simulation_test_params)

    sim_cfg = dict(simulation_test_params)
    initial_baseline = estimate_initial_random_precoder_schedule(
        system_test_params,
        sim_cfg,
        seed=int(test_seed),
    )

    data_dir = os.path.join("data_saves", save_tag)
    latency_dir = os.path.join("figs", save_tag, "test_result")
    link_dir = latency_dir
    if result_dirs is not None:
        data_dir = result_dirs.get("test_data", result_dirs["data"])
        latency_dir = result_dirs["latency_asynchronality"]
        link_dir = result_dirs["link_quality"]

    test_uplinksystem = UplinkSystem(system_test_params, seed=test_seed)
    _, initial_snr_db = test_uplinksystem.get_SNR()
    _, initial_sinr_db = test_uplinksystem.get_SINR()

    plot_params = dict(system_test_params)
    plot_params["initial_bits_per_symbol"] = np.asarray(initial_baseline["initial_bits_per_symbol"], dtype=float)
    plot_user_config(
        plot_params,
        extra_params={
            "measured_snr_db_k": np.asarray(initial_snr_db),
            "measured_sinr_db_k": np.asarray(initial_sinr_db),
        },
    )

    initial_Rfbl = [np.array(v, copy=True) for v in initial_baseline["initial_R_fbl"]]
    test_uplinksystem.R_fbl[:] = []

    initial_latency = list(initial_baseline["initial_latency"])
    initial_bits_per_symbol = list(initial_baseline["initial_bits_per_symbol"])
    initial_bits_per_symbol_by_block = [
        list(values) for values in initial_baseline["initial_bits_per_symbol_by_block"]
    ]
    initial_n = list(initial_baseline["initial_n"])
    initial_n_kl = [list(values) for values in initial_baseline["initial_n_kl"]]
    initial_B_kl = [list(values) for values in initial_baseline["initial_B_kl"]]

    test_data_dict = dynamic_subblocklength_precoder_testing(
        uplinksystem=test_uplinksystem,
        post_training_data_dict=post_training_data_dict,
        sim_cfg=sim_cfg,
        channel_norm=True,
    )

    save_test_results_to_txt(
        test_uplinksystem=test_uplinksystem,
        test_data_dict=test_data_dict,
        initial_Rfbl=initial_Rfbl,
        initial_n_kl=initial_n_kl,
        initial_n=initial_n,
        initial_latency=initial_latency,
        initial_snr_db=initial_snr_db,
        initial_sinr_db=initial_sinr_db,
        save_dir=data_dir,
        filename="test_results.txt",
        initial_bits_per_symbol=initial_bits_per_symbol,
        initial_B_kl=initial_B_kl,
        initial_bits_per_symbol_by_block=initial_bits_per_symbol_by_block,
    )

    if do_plots:
        plot_optimization_result(test_data_dict["all_user_block_results_test"], train=False)
        summary_dict = {
            "n_star": test_data_dict["n_star_test"],
            "R_star": test_data_dict["R_star_test"],
        }
        plot_optimization_result_summary_dict(summary_dict, train=False)
        plot_latency_and_asynchronality_from_json(
            json_path=os.path.join(data_dir, "test_results.json"),
            save_dir=latency_dir,
            prefix="test",
        )
        plot_link_quality_from_json(
            json_path=os.path.join(data_dir, "test_results.json"),
            save_dir=link_dir,
            prefix="test",
        )

    result = build_precoder_net_result(
        test_uplinksystem,
        test_data_dict,
        method_name="convergence_per_epoch_baseline",
        cfg_path=_resolve_config_path(cfg_name),
        test_seed=int(test_seed),
        train_seeds=[],
        train_artifact=post_training_data_dict,
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
    save_json(result, os.path.join(data_dir, "result.json"))
    save_text(build_summary_lines(result), os.path.join(data_dir, "summary.txt"))
    return test_uplinksystem, test_data_dict, result
