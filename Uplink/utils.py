import json
import os
from dataclasses import dataclass, asdict
from typing import Optional, Sequence

import numpy as np
import torch

from experiment_scenarios import FIXED_BLOCK_TARGETS_MODE

def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj
    
def save_test_results_to_txt(
    test_uplinksystem,
    test_data_dict,
    initial_Rfbl,
    initial_n_kl,
    initial_n,
    initial_latency,
    initial_snr_db,
    initial_sinr_db,
    initial_bits_per_symbol,
    save_dir,
    filename,
    initial_B_kl=None,
    initial_bits_per_symbol_by_block=None,
):
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, filename)

    final_bits_per_symbol = [
        test_data_dict["B_kl_star_test"][user] /
        np.array(test_uplinksystem.n_kl[user], dtype=np.float64)
        for user in range(test_uplinksystem.K)
    ]
    _, final_snr_db = test_uplinksystem.get_SNR()
    _, final_sinr_db = test_uplinksystem.get_SINR()
    with open(file_path, "w", encoding="utf-8") as f:

        for user_idx in range(test_uplinksystem.K):
            user_initial_b_kl = (
                list(initial_B_kl[user_idx])
                if initial_B_kl is not None and user_idx < len(initial_B_kl)
                else [int(test_uplinksystem.B[user_idx])]
            )
            user_initial_bps_by_block = (
                list(initial_bits_per_symbol_by_block[user_idx])
                if initial_bits_per_symbol_by_block is not None and user_idx < len(initial_bits_per_symbol_by_block)
                else [float(initial_bits_per_symbol[user_idx])]
            )
            print("logging rfbl: ", initial_Rfbl[user_idx])
            f.write(f"\n|| ---------------- USER {user_idx} ---------------- ||\n")
            f.write(
                f"R-fsbl_kl:    Initial {initial_Rfbl[user_idx]} "
                f"->      Final {test_uplinksystem.R_fbl[user_idx]}\n"
            )
            f.write(
                f"n_kl:   Initial {initial_n_kl[user_idx]} "
                f"->   Final {test_uplinksystem.n_kl[user_idx]}\n"
            )
            f.write(
                f"N_k:     Initial {initial_n[user_idx]} "
                f"->  Final {test_uplinksystem.n[user_idx]}\n"
            )
            f.write(
                f"B^tx_kl:    Initial {user_initial_b_kl}"
                f"->   Final  {test_data_dict['B_kl_star_test'][user_idx]} \n"
            )
            f.write(
                f"Bits per Symbol:    Initial avg {initial_bits_per_symbol[user_idx]}"
                f", per block {user_initial_bps_by_block}"
                f"->   Final {final_bits_per_symbol[user_idx]} \n"
                )
            f.write(
                f"SNR_dB: Initial {initial_snr_db[user_idx]:.4f} -> Final {final_snr_db[user_idx]:.4f}\n"
            )
            f.write(
                f"SINR_dB: Initial {initial_sinr_db[user_idx]:.4f} -> Final {final_sinr_db[user_idx]:.4f}\n"
            )

            if str(test_data_dict.get("scenario_mode", "")) == FIXED_BLOCK_TARGETS_MODE:
                scenario_block_targets = np.asarray(test_data_dict.get("scenario_block_targets", []), dtype=int)
                if scenario_block_targets.ndim == 2 and user_idx < scenario_block_targets.shape[0]:
                    target_bits = scenario_block_targets[user_idx].tolist()
                    final_bits = list(map(int, test_data_dict["B_kl_star_test"][user_idx]))
                    initial_bits = (
                        list(map(int, initial_B_kl[user_idx]))
                        if initial_B_kl is not None and user_idx < len(initial_B_kl)
                        else [0 for _ in range(len(target_bits))]
                    )
                    initial_unserved = [max(int(t) - int(b), 0) for t, b in zip(target_bits, initial_bits)]
                    final_unserved = [max(int(t) - int(b), 0) for t, b in zip(target_bits, final_bits)]
                    f.write(f"Target bits per block: {target_bits}\n")
                    f.write(f"Initial unserved bits per block: {initial_unserved}\n")
                    f.write(f"Final unserved bits per block: {final_unserved}\n")

        f.write("\nInitial Latency per user:\n")
        f.write(f"{initial_latency}\n")

        f.write("\nFinal Latency per user:\n")
        f.write(f"{test_uplinksystem.latency}\n")
        
        # -------------------------------------------------
        # Latency Reduction %
        # (Using sum of latencies)
        # -------------------------------------------------
        initial_total_latency = sum(initial_latency)
        final_total_latency = sum(test_uplinksystem.latency)
        
        latency_reduction_per_user = (np.array(initial_latency) - np.array(test_uplinksystem.latency))/np.array(initial_latency)*100
        if initial_total_latency > 0:
            latency_reduction_percent = (
                (initial_total_latency - final_total_latency)
                / initial_total_latency
            ) * 100
        else:
            latency_reduction_percent = 0.0
        
        f.write(f"Latency reduction per user (%): {latency_reduction_per_user}%\n ")
        f.write(f"Total Latency Reduction(total_initial_latency/total_final_latency) (%): {latency_reduction_percent}%\n")
        
        f.write("\nInitial Asynchronality:\n")
        sum_initial_asynchronality = 0

        for i in range(len(initial_latency)):
            for j in range(i + 1, len(initial_latency)):
                diff = abs(initial_latency[i] - initial_latency[j])
                sum_initial_asynchronality += diff
                f.write(f"User {i} - User {j}: {diff}\n")

        f.write(f"\nInitial Sum of Asynchronality: {sum_initial_asynchronality}\n")


        f.write("\nFinal Asynchronality:\n")
        sum_final_asynchronality = 0

        for i in range(len(test_uplinksystem.latency)):
            for j in range(i + 1, len(test_uplinksystem.latency)):
                diff = abs(test_uplinksystem.latency[i] - test_uplinksystem.latency[j])
                sum_final_asynchronality += diff
                f.write(f"User {i} - User {j}: {diff}\n")

        f.write(f"\nFinal Sum of Asynchronality: {sum_final_asynchronality}\n")


        # -------------------------------------------------
        # Asynchronality Reduction %
        # -------------------------------------------------
        if sum_initial_asynchronality > 0:
            async_reduction_percent = (
                (sum_initial_asynchronality - sum_final_asynchronality)
                / sum_initial_asynchronality
            ) * 100
        else:
            async_reduction_percent = 0.0

        f.write(f"\nAsynchronality Reduction (%): {async_reduction_percent}%\n")

        print(f"Test results saved to: {file_path}")
    json_data = {
        "initial_latency": list(initial_latency),
        "final_latency": list(test_uplinksystem.latency),
        "initial_n": list(initial_n),
        "final_n": list(test_uplinksystem.n),
        "initial_n_kl": initial_n_kl,
        "final_n_kl": test_uplinksystem.n_kl,
        "initial_Rfbl": initial_Rfbl,
        "final_Rfbl": test_uplinksystem.R_fbl,
        "initial_B_kl": initial_B_kl,
        "target_snr_db": list(test_uplinksystem.snr_db),
        "initial_snr_db": list(initial_snr_db),
        "final_snr_db": list(final_snr_db),
        "initial_sinr_db": list(initial_sinr_db),
        "final_sinr_db": list(final_sinr_db),
        "initial_bits_per_symbol": initial_bits_per_symbol,
        "initial_bits_per_symbol_by_block": initial_bits_per_symbol_by_block,
        "final_bits_per_symbol": [list(x) for x in final_bits_per_symbol],
        "B_tx_final": test_data_dict["B_kl_star_test"],
        "scenario_mode": test_data_dict.get("scenario_mode", ""),
        "scenario_block_targets": test_data_dict.get("scenario_block_targets", []),
        "latency_reduction_per_user_percent": latency_reduction_per_user.tolist(),
        "total_latency_reduction_percent": latency_reduction_percent,
        "initial_asynchronality_sum": sum_initial_asynchronality,
        "final_asynchronality_sum": sum_final_asynchronality,
        "asynchronality_reduction_percent": async_reduction_percent
    }
    serializable_data = make_json_serializable(json_data)

    json_path = os.path.join(save_dir, filename.replace(".txt", ".json"))

    with open(json_path, "w") as jf:
        json.dump(serializable_data, jf, indent=4)
        
@dataclass()
class SystemParams:
    K: int
    B: np.ndarray              # (K,)  channel uses per coherence block / sub-block
    fs: np.ndarray             # (K,)  symbols/sec (or channel uses/sec)
    f_carrier: None            # (K,)  carrier frequency (Hz)
    v: None             # (K,)  user speed (m/s)
    P: np.ndarray             # (K,)  transmit power (linear)
    NR: np.ndarray             # (K,)  receiver antennas at BS
    NT: np.ndarray             # (K,)  transmit antennas at UE
    snr_db: np.ndarray         # (K,)  SNR in dB (if you’re using it elsewhere)
    desired_CNR: np.ndarray    # (K,)  desired CNR (kept as provided)
    epsilon: np.ndarray        # (K,)  target error probability

    # Derived
    D_s: None            # (K,)  doppler shift (Hz)
    C_T: None            # (K,)  coherence time (s)
    T: None              # (K,)  coherence blocklength in symbols (C_T * fs)
    L: np.ndarray              # (K,)  number of sub-blocks (currently all ones)
    n_kl: list[list[int]]       # per-user per-subblock channel uses
    n: np.ndarray              # (K,)  total channel uses (sum over n_kl)
    latency: np.ndarray        # (K,)  seconds (n / fs)
    dk: np.ndarray             # (K,)  min(NR, NT)
    initial_latency: np.ndarray
    initial_bits_per_symbol: list


def _as_1d_array(x: Sequence, K: int, name: str, dtype=None) -> np.ndarray:
    arr = np.asarray(x, dtype=dtype)
    if arr.shape != (K,):
        raise ValueError(f"{name} must have shape (K,) = ({K},), got {arr.shape}")
    return arr


def initialize_system_params(
    K: int,
    *,
    B: Sequence[int],
    fs: Sequence[float],
    f_carrier: Sequence[float]|None = None,
    v: Sequence[float]|None = None,
    P: Sequence[float],
    Nr: Sequence[int],
    Nt: Sequence[int],
    snr_db: Sequence[float],
    desired_CNR: Optional[Sequence[float]] = None,
    epsilon: Sequence[float],
    c: float = 299_792_458.0,  # speed of light (m/s)
    initial_bits_per_symbol=None,
    T = None
) -> dict:
    """
    Build and validate all system parameters.

    """

    if desired_CNR is None:
        desired_CNR_arr = np.zeros(K, dtype=float)
    else:
        desired_CNR_arr = _as_1d_array(desired_CNR, K, "desired_CNR", dtype=float)
    # --- Validate + convert inputs to (K,) arrays ---
    B_arr = _as_1d_array(B, K, "B", dtype=int)
    fs_arr = _as_1d_array(fs, K, "fs", dtype=float)
    Pt_arr = _as_1d_array(P, K, "P", dtype=float)
    NR_arr = _as_1d_array(Nr, K, "Nr", dtype=int)
    NT_arr = _as_1d_array(Nt, K, "Nt", dtype=int)
    snr_db_arr = _as_1d_array(snr_db, K, "snr_db", dtype=float)
    eps_arr = _as_1d_array(epsilon, K, "epsilon", dtype=float)



    
    if T is None:
        fc_arr = _as_1d_array(f_carrier, K, "f_carrier", dtype=float)
        v_arr = _as_1d_array(v, K, "v", dtype=float)
        # --- Derived quantities ---
        # Doppler spread (Hz): D_s = 2*f_carrier*v/c

        D_s = fc_arr * v_arr / c
        if np.any(D_s <= 0):
            raise ValueError("All Doppler shifts must be > 0 (check f_carrier and v).")
        # Coherence time (s): C_T = 1/(4*D_s)
        # C_T = 1.0 / (4.0 * D_s)
        C_T = 1.0 / (D_s)

        # Coherence blocklength in symbols: T = C_T * fs
        T = C_T * fs_arr
        T = np.array(T, dtype = int)
    else:
        D_s = None
        C_T = None
        fc_arr = None
        v_arr = None
        
        T = np.array(T, dtype = int)
        
    # Number of sub-blocks (currently fixed to 1 for each user)
    L = np.ones(K, dtype=int)

    # Per-subblock channel uses: n_kl[k] is a list of length L[k]
    # Assumes
    n_kl = [[T[k]] for k in range(K)]

    # Total channel uses per user
    n = np.array([sum(n_kl[k]) for k in range(K)], dtype=int)
    # Latency in seconds: n / fs
    latency = n / fs_arr
    if initial_bits_per_symbol is None:
        initial_bits_per_symbol_arr = B_arr.astype(float) / np.maximum(np.asarray(T, dtype=float), 1.0)
    else:
        initial_bits_per_symbol_arr = _as_1d_array(
            initial_bits_per_symbol,
            K,
            "initial_bits_per_symbol",
            dtype=float,
        )

    initial_latency = (B_arr / np.maximum(initial_bits_per_symbol_arr, 1e-12)) / fs_arr
    

    # Spatial streams (or min dimension): dk = min(NR, NT)
    dk = np.minimum(NR_arr, NT_arr)

    params = SystemParams(
        K=K,
        B=B_arr,
        fs=fs_arr,
        f_carrier=fc_arr,
        v=v_arr,
        P=Pt_arr,
        NR=NR_arr,
        NT=NT_arr,
        snr_db=snr_db_arr,
        desired_CNR=desired_CNR_arr,
        epsilon=eps_arr,
        D_s=D_s,
        C_T=C_T,
        T=T,
        L=L,
        n_kl=n_kl,
        n=n,
        latency=latency,
        dk=dk,
        initial_latency = initial_latency,
        initial_bits_per_symbol = initial_bits_per_symbol_arr.tolist()
    )

    # Keep your original return style (dict of arrays)
    return asdict(params)


#------------------------ Folder Utilis ---------------------------
import os
import torch


import os

def set_up_folders(cfg_name):
    structure = {
        "figs": ["train_result", "test_result", "user_cfg"],
        "data_saves": []
    }

    for base, subs in structure.items():
        exp_base = os.path.join(base, cfg_name)

        if not subs:
            os.makedirs(exp_base, exist_ok=True)
            print(f"Directory ready: {exp_base}")
        else:
            for sub in subs:
                path = os.path.join(exp_base, sub)
                os.makedirs(path, exist_ok=True)
                print(f"Directory ready: {path}")


def clear_folders(cfg_name, base_folders=None, extension=".png"):
    """
    Delete files with given extension inside the experiment folders.
    """
    if base_folders is None:
        base_folders = ["figs", "data_saves"]

    for base in base_folders:

        exp_path = os.path.join(base, cfg_name)

        if not os.path.isdir(exp_path):
            print(f"Skipping (not found): {exp_path}")
            continue

        for root, _, files in os.walk(exp_path):
            for fname in files:
                if fname.lower().endswith(extension.lower()):
                    file_path = os.path.join(root, fname)
                    try:
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")
                    except OSError as e:
                        print(f"Failed to delete {file_path}: {e}")
                        
def save_simulation_data(data_dict, train_seed, folder="data_saves"):
    """
    Save simulation dictionary using torch.save.
    """
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"train_data_seed_{train_seed}.pt")

    torch.save(data_dict, path)
    print(f"Simulation data saved to: {path}")
