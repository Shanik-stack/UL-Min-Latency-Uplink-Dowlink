import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from advanced_methods_common import collect_uplink_interference_diagnostics

# ============================================================
# Module-level plotting globals
# These should be initialized once by initialize_plot_globals(cfg_name)
# ============================================================
home_path = os.path.dirname(os.path.abspath(__file__))

cfg_name = None
train_result_path = None
test_result_path = None
user_cfg_result_path = None
def to_numpy_safe(x):
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):  # torch tensor
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _get_result_save_dir(train: bool, save_dir=None):
    global train_result_path, test_result_path

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    result_dir = train_result_path if train else test_result_path
    if result_dir is None:
        raise ValueError(
            "Plot result path is None. Call initialize_plot_globals(cfg_name) first."
        )

    os.makedirs(result_dir, exist_ok=True)
    return result_dir


def initialize_plot_globals(cfg_name_in, result_dirs=None):
    global cfg_name, train_result_path, test_result_path, user_cfg_result_path

    cfg_name = cfg_name_in
    if result_dirs is None:
        train_result_path = os.path.join(home_path, "figs", cfg_name, "train_result")
        test_result_path = os.path.join(home_path, "figs", cfg_name, "test_result")
        user_cfg_result_path = os.path.join(home_path, "figs", cfg_name, "user_cfg")
    else:
        train_result_path = result_dirs["train_result"]
        test_result_path = result_dirs["test_result"]
        user_cfg_result_path = result_dirs["user_config"]

    os.makedirs(train_result_path, exist_ok=True)
    os.makedirs(test_result_path, exist_ok=True)
    os.makedirs(user_cfg_result_path, exist_ok=True)

    print("Initialized plotting globals:")
    print("train_result_path:", train_result_path)
    print("test_result_path:", test_result_path)
    print("user_cfg_result_path:", user_cfg_result_path)




def plot_optimization_result_summary_dict(
    post_training_data_dict,
    train=True,
    save_dir=None,
    phase_label=None,
    filename_prefix=None,
):
    """
    Uses module global experiment folders unless save_dir is explicitly given.

    Expected keys:
      - training: 'n_star', 'R_star'
      - testing summary remap should also provide same keys

    Saves:
      training_user{user}_summary.png   if train=True
      testing_user{user}_summary.png    if train=False
    """
    save_dir = _get_result_save_dir(train=train, save_dir=save_dir)

    n_star = post_training_data_dict.get("n_star", None)
    R_star = post_training_data_dict.get("R_star", None)

    if n_star is None or R_star is None:
        raise KeyError("post_training_data_dict must contain keys: 'n_star' and 'R_star'")

    if len(n_star) != len(R_star):
        raise ValueError(
            f"Length mismatch: len(n_star)={len(n_star)} but len(R_star)={len(R_star)}"
        )

    title_prefix = phase_label or ("Training" if train else "Testing")
    file_prefix = filename_prefix or ("training" if train else "testing")

    for user_idx in range(len(n_star)):
        L = len(n_star[user_idx])
        if L == 0:
            continue

        blocks = np.arange(L)
        t_vals = np.asarray(n_star[user_idx], dtype=np.float64)
        r_vals = np.asarray(R_star[user_idx], dtype=np.float64)

        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax2 = ax1.twinx()

        ax1.plot(blocks, t_vals, marker="o", label="n_kl")
        ax1.set_xlabel("Block index (l)")
        ax1.set_ylabel("Chosen sub-blocklength n_kl")
        ax1.grid(True)

        ax2.plot(blocks, r_vals, marker="s", label="R_fbl")
        ax2.set_ylabel("Chosen R_fbl")

        ax1.set_title(f"{title_prefix} Result – User {user_idx}")

        # combined legend
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="best")

        fig.tight_layout()

        save_path = os.path.join(
            save_dir,
            f"{file_prefix}_user{user_idx}_summary.png"
        )

        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


import os
import numpy as np
import matplotlib.pyplot as plt


def to_numpy_safe(x):
    if isinstance(x, np.ndarray):
        return x
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def plot_F_vs_n_for_all_subblocks(
    post_training_data_dict,
    save_dir="F_vs_n",   # only subfolder name
    base_dir=None,
):
    """
    Uses global cfg paths (must call initialize_plot_globals first)
    Saves under:
        figs/<cfg_name>/train_result/F_vs_n/
    """

    # base train_result path
    if base_dir is None:
        base_dir = _get_result_save_dir(train=True)
    else:
        os.makedirs(base_dir, exist_ok=True)

    # append subfolder
    save_dir = os.path.join(base_dir, save_dir)
    os.makedirs(save_dir, exist_ok=True)

    all_user_block_results = post_training_data_dict["all_user_block_results_train"]

    for user_idx, user_blocks in enumerate(all_user_block_results):
        if not user_blocks:
            continue

        # ==================================================
        # FIG 1: ||F||^2 vs n
        # ==================================================
        plt.figure(figsize=(8, 5))

        for block_idx, S in enumerate(user_blocks):
            if not S:
                continue

            n_vals, F_power_vals = [], []

            for item in S:
                if "n_kl" not in item or "F" not in item:
                    continue

                n_vals.append(int(item["n_kl"]))

                if "F_power" in item:
                    F_power_vals.append(float(item["F_power"]))
                else:
                    F_arr = to_numpy_safe(item["F"])
                    F_power_vals.append(np.sum(np.abs(F_arr) ** 2))

            if len(n_vals) == 0:
                continue

            order = np.argsort(n_vals)
            n_vals = np.array(n_vals)[order]
            F_power_vals = np.array(F_power_vals)[order]

            plt.plot(n_vals, F_power_vals, marker="o", label=f"block {block_idx}")

        plt.xlabel(r"$n_{k,\ell}$")
        plt.ylabel(r"$\|F\|_F^2$")
        plt.title(f"User {user_idx}: Precoder power vs n")
        plt.grid(True)
        plt.legend()

        plt.gca().invert_xaxis()   # correct position

        plt.savefig(
            os.path.join(save_dir, f"user{user_idx}_Fpower_vs_n.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

        # ==================================================
        # FIG 2: ||F(n) - F(n_max)|| vs n
        # ==================================================
        plt.figure(figsize=(8, 5))

        for block_idx, S in enumerate(user_blocks):
            if not S:
                continue

            n_vals, F_list = [], []

            for item in S:
                if "n_kl" not in item or "F" not in item:
                    continue
                n_vals.append(int(item["n_kl"]))
                F_list.append(to_numpy_safe(item["F"]))

            if len(n_vals) == 0:
                continue

            order_desc = np.argsort(n_vals)[::-1]
            n_desc = np.array(n_vals)[order_desc]
            F_desc = [F_list[i] for i in order_desc]

            F_ref = F_desc[0]

            delta = [
                np.linalg.norm((F - F_ref).reshape(-1))
                for F in F_desc
            ]

            order_inc = np.argsort(n_desc)
            n_plot = n_desc[order_inc]
            d_plot = np.array(delta)[order_inc]

            plt.plot(n_plot, d_plot, marker="o", label=f"block {block_idx}")

        plt.xlabel(r"$n_{k,\ell}$")
        plt.ylabel(r"$\|F(n) - F(n_{\max})\|_F$")
        plt.title(f"User {user_idx}: Precoder drift vs n")
        plt.grid(True)
        plt.legend()

        plt.gca().invert_xaxis()

        plt.savefig(
            os.path.join(save_dir, f"user{user_idx}_Fchange_vs_n.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

        # ==================================================
        # FIG 3: incremental change
        # ==================================================
        plt.figure(figsize=(8, 5))

        for block_idx, S in enumerate(user_blocks):
            if not S:
                continue

            n_vals, F_list = [], []

            for item in S:
                if "n_kl" not in item or "F" not in item:
                    continue
                n_vals.append(int(item["n_kl"]))
                F_list.append(to_numpy_safe(item["F"]))

            if len(n_vals) == 0:
                continue

            order_desc = np.argsort(n_vals)[::-1]
            n_desc = np.array(n_vals)[order_desc]
            F_desc = [F_list[i] for i in order_desc]

            delta_prev = [0.0]
            for i in range(1, len(F_desc)):
                delta_prev.append(
                    np.linalg.norm((F_desc[i] - F_desc[i - 1]).reshape(-1))
                )

            order_inc = np.argsort(n_desc)
            n_plot = n_desc[order_inc]
            d_plot = np.array(delta_prev)[order_inc]

            plt.plot(n_plot, d_plot, marker="o", label=f"block {block_idx}")

        plt.xlabel(r"$n_{k,\ell}$")
        plt.ylabel(r"$\|F_i - F_{i-1}\|_F$")
        plt.title(f"User {user_idx}: incremental F change")
        plt.grid(True)
        plt.legend()

        plt.gca().invert_xaxis()

        plt.savefig(
            os.path.join(save_dir, f"user{user_idx}_Fincrement_vs_n.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

        # ==================================================
        # FIG 4: R_fbl vs n
        # ==================================================
        plt.figure(figsize=(8, 5))

        for block_idx, S in enumerate(user_blocks):
            if not S:
                continue

            n_vals, R_vals = [], []

            for item in S:
                if "n_kl" not in item or "R_fbl" not in item:
                    continue
                n_vals.append(int(item["n_kl"]))
                R_vals.append(float(item["R_fbl"]))

            if len(n_vals) == 0:
                continue

            order = np.argsort(n_vals)
            n_vals = np.array(n_vals)[order]
            R_vals = np.array(R_vals)[order]

            plt.plot(n_vals, R_vals, marker="o", label=f"block {block_idx}")

        plt.xlabel(r"$n_{k,\ell}$")
        plt.ylabel(r"$R_{\mathrm{fbl}}$")
        plt.title(f"User {user_idx}: Rate vs n")
        plt.grid(True)
        plt.legend()

        plt.gca().invert_xaxis()

        plt.savefig(
            os.path.join(save_dir, f"user{user_idx}_Rfbl_vs_n.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

def plot_optimization_result(
    all_user_results,
    train=True,
    save_dir=None,
    phase_label=None,
    filename_prefix=None,
):
    """
    all_user_results[user][block] = list_of_dicts_over_n_kl

    Uses module global experiment folders unless save_dir is explicitly given.

    Saves:
      training_optimization_user{u}_block{b}.png   if train=True
      testing_optimization_user{u}_block{b}.png    if train=False
    """
    save_dir = _get_result_save_dir(train=train, save_dir=save_dir)
    title_prefix = phase_label or ("Training" if train else "Testing")
    file_prefix = filename_prefix or ("training" if train else "testing")

    for user_idx, user_result in enumerate(all_user_results):
        for block_idx, block_result in enumerate(user_result):
            if block_result is None or len(block_result) == 0:
                continue

            t_vals = np.asarray([d["n_kl"] for d in block_result], dtype=np.float64)
            bits_per_blk = np.asarray(
                [d["Bits per sub-block length B/n_kl"] for d in block_result],
                dtype=np.float64
            )
            r_fbl_vals = np.asarray([d["R_fbl"] for d in block_result], dtype=np.float64)

            fig = plt.figure(figsize=(6, 4))
            plt.plot(t_vals, bits_per_blk, marker="o", label="B/n_kl")
            plt.plot(t_vals, r_fbl_vals, marker="s", label="R_fbl")

            t_final = t_vals[-1]
            b_final = bits_per_blk[-1]
            r_final = r_fbl_vals[-1]

            plt.annotate(
                f"B/n_kl: {b_final:.4f}",
                (t_final, b_final),
                xytext=(0, -15),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black"),
            )

            plt.annotate(
                f"R_fbl: {r_final:.4f}",
                (t_final, r_final),
                xytext=(0, 15),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black"),
            )

            plt.xlabel("Blocklength n_kl")
            plt.ylabel("Rate (bits / symbol)")

            plt.title(f"{title_prefix} Result – User {user_idx}, Block {block_idx}")

            plt.legend()
            plt.grid(True)
            plt.gca().invert_xaxis()
            plt.tight_layout()

            save_path = os.path.join(
                save_dir,
                f"{file_prefix}_optimization_user{user_idx}_block{block_idx}.png"
            )

            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

def plot_user_config(system_params, extra_params=None, max_ticks=12):
    save_path=user_cfg_result_path
    """
    Save adaptive user configuration plots.

    Features:
    - Adapts automatically to number of users K
    - Saves standard per-user plots
    - Saves reverse scatter plots
    - Saves summary grid of standard plots
    - Saves summary grid of reverse plots

    Expected keys in system_params:
        K
        snr_db
        P
        B
        T
        initial_bits_per_symbol
    """

    os.makedirs(save_path, exist_ok=True)

    K = int(system_params["K"])
    users = np.arange(K)

    params = {
        "snr_db_k": np.asarray(system_params["snr_db"]),
        "P_k": np.asarray(system_params["P"]),
        "B_k": np.asarray(system_params["B"]),
        "T_k": np.asarray(system_params["T"]),
        "initial_bits_per_symbol": np.asarray(system_params["initial_bits_per_symbol"]),
    }
    if extra_params is not None:
        for name, values in extra_params.items():
            arr = np.asarray(values)
            if arr.shape != (K,):
                raise ValueError(f"{name} must have shape ({K},), got {arr.shape}")
            params[name] = arr

    def get_sparse_ticks(arr, max_ticks=12):
        arr = np.asarray(arr)
        n = len(arr)
        if n <= max_ticks:
            return arr
        idx = np.linspace(0, n - 1, max_ticks, dtype=int)
        return arr[idx]

    def get_user_figsize(K, base_width=8):
        height = max(4.5, min(8.0, 4.0 + 0.02 * K))
        return (base_width, height)

    def get_summary_figsize(K):
        height = max(8, min(14, 8 + 0.015 * K))
        width = 12
        return (width, height)

    def get_marker_size(K):
        if K <= 20:
            return 7
        elif K <= 100:
            return 5
        elif K <= 300:
            return 3
        else:
            return 2

    def get_scatter_size(K):
        if K <= 20:
            return 40
        elif K <= 100:
            return 20
        elif K <= 300:
            return 10
        else:
            return 6

    def padded_limits(values):
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        if np.isclose(vmin, vmax):
            pad = 0.5 if vmin == 0 else 0.05 * abs(vmin)
            return vmin - pad, vmax + pad
        pad = 0.05 * (vmax - vmin)
        return vmin - pad, vmax + pad

    x_ticks_sparse = get_sparse_ticks(users, max_ticks=max_ticks)
    y_ticks_sparse = get_sparse_ticks(users, max_ticks=max_ticks)
    marker_size = get_marker_size(K)
    scatter_size = get_scatter_size(K)

    # -------------------------------------------------
    # 1. Standard plots (User on x-axis)
    # -------------------------------------------------
    for name, values in params.items():
        plt.figure(figsize=get_user_figsize(K))
        plt.plot(users, values, marker="o", markersize=marker_size)
        plt.xlabel("User Index")
        plt.ylabel(name)
        plt.title(f"{name} per User")
        plt.xticks(x_ticks_sparse)
        plt.grid(True)

        save_file = os.path.join(save_path, f"{name}_per_user.png")
        plt.savefig(save_file, dpi=300, bbox_inches="tight")
        plt.close()

    # -------------------------------------------------
    # 2. Reverse plots (NO GRID)
    # -------------------------------------------------
    for name, values in params.items():
        order = np.argsort(values)
        xvals = values[order]
        yvals = users[order]

        plt.figure(figsize=get_user_figsize(K))
        plt.scatter(xvals, yvals, s=scatter_size, alpha=0.7)

        plt.xlabel(name)
        plt.ylabel("User Index")
        plt.title(f"Users vs {name}")
        plt.xlim(*padded_limits(xvals))
        plt.yticks(y_ticks_sparse)

        save_file = os.path.join(save_path, f"{name}_users_scatter.png")
        plt.savefig(save_file, dpi=300, bbox_inches="tight")
        plt.close()

    # -------------------------------------------------
    # 3. Standard summary grid
    # -------------------------------------------------
    n_params = len(params)
    ncols = 2
    nrows = int(np.ceil(n_params / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, max(8, 4 * nrows)))
    axes = axes.flatten()

    for i, (name, values) in enumerate(params.items()):
        axes[i].plot(users, values, marker="o", markersize=marker_size)
        axes[i].set_title(name)
        axes[i].set_xlabel("User")
        axes[i].set_ylabel(name)
        axes[i].set_xticks(x_ticks_sparse)
        axes[i].grid(True)

    for i in range(n_params, len(axes)):
        fig.delaxes(axes[i])
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "user_config_summary.png"), dpi=300)
    plt.close()

    # -------------------------------------------------
    # 4. Reverse summary grid (NO GRID)
    # -------------------------------------------------
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, max(8, 4 * nrows)))
    axes = axes.flatten()

    for i, (name, values) in enumerate(params.items()):
        order = np.argsort(values)
        xvals = values[order]
        yvals = users[order]

        axes[i].scatter(xvals, yvals, s=scatter_size, alpha=0.7)
        axes[i].set_title(f"Users vs {name}")
        axes[i].set_xlabel(name)
        axes[i].set_ylabel("User")
        axes[i].set_xlim(*padded_limits(xvals))
        axes[i].set_yticks(y_ticks_sparse)

    for i in range(n_params, len(axes)):
        fig.delaxes(axes[i])
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "user_config_reverse_summary.png"), dpi=300)
    plt.close()

def plot_link_quality_from_json(json_path, save_dir=None, prefix="test"):
    """
    Plot target SNR together with measured SNR/SINR before and after optimization.
    """
    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(json_path))
    os.makedirs(save_dir, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    target_snr_db = np.asarray(data.get("target_snr_db", []), dtype=np.float64)
    initial_snr_db = np.asarray(data.get("initial_snr_db", []), dtype=np.float64)
    final_snr_db = np.asarray(data.get("final_snr_db", []), dtype=np.float64)
    initial_sinr_db = np.asarray(data.get("initial_sinr_db", []), dtype=np.float64)
    final_sinr_db = np.asarray(data.get("final_sinr_db", []), dtype=np.float64)

    if target_snr_db.size == 0 or initial_sinr_db.size == 0 or final_sinr_db.size == 0:
        return

    K = len(target_snr_db)
    user_idx = np.arange(K)
    tick_spacing = 1 if K <= 20 else (5 if K <= 50 else 10)
    sparse_ticks = user_idx[::tick_spacing]
    width = max(10, K * 0.18)

    plt.figure(figsize=(width, 6))
    plt.plot(user_idx, target_snr_db, label="Target SNR (cfg)", linewidth=2)
    if initial_snr_db.size == K:
        plt.plot(user_idx, initial_snr_db, label="Measured SNR (initial)", linestyle="--", alpha=0.85)
    if final_snr_db.size == K:
        plt.plot(user_idx, final_snr_db, label="Measured SNR (final)", linestyle=":", alpha=0.85)
    plt.plot(user_idx, initial_sinr_db, label="Measured SINR (initial)", linewidth=2)
    plt.plot(user_idx, final_sinr_db, label="Measured SINR (final)", linewidth=2)
    plt.xlabel("User Index")
    plt.ylabel("dB")
    plt.title("Per-User Link Quality: Target SNR vs Measured SNR/SINR")
    plt.xticks(sparse_ticks)
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_link_quality_comparison.png"), dpi=300)
    plt.close()
import os
import numpy as np
import matplotlib.pyplot as plt

import os
import json
import numpy as np
import matplotlib.pyplot as plt

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import numpy as np
import matplotlib.pyplot as plt

def plot_latency_and_asynchronality_from_json(json_path, save_dir=None, prefix="test"):
    """
    Generates a full suite of visualizations for latency and asynchronality:
    1. Bar Comparison 2. Population Hist+CDF 3. Async Distribution 4. Heatmaps
    """
    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(json_path))
    os.makedirs(save_dir, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # --- Data Extraction ---
    initial_latency = np.array(data["initial_latency"], dtype=np.float64)
    final_latency = np.array(data.get("final_latency", data.get("latency")), dtype=np.float64)
    K = len(final_latency)
    user_idx = np.arange(K)

    # --- Adaptive Scaling Logic ---
    dynamic_width = max(10, K * 0.2) 
    tick_spacing = 1 if K <= 20 else (5 if K <= 50 else 10)
    sparse_ticks = user_idx[::tick_spacing]

    # 1) PER-USER BAR COMPARISON
    plt.figure(figsize=(dynamic_width, 6))
    if K > 150:
        plt.plot(user_idx, initial_latency, label="Initial", alpha=0.7, marker='o', markersize=2)
        plt.plot(user_idx, final_latency, label="Final", alpha=0.9, marker='x', markersize=2)
    else:
        bar_width = 0.35 if K <= 50 else 0.3 
        plt.bar(user_idx - bar_width/2, initial_latency, width=bar_width, label="Initial", color='#1f77b4', zorder=3)
        plt.bar(user_idx + bar_width/2, final_latency, width=bar_width, label="Final", color='#ff7f0e', zorder=3)
    plt.xlabel("User Index"); plt.ylabel("Latency"); plt.legend()
    plt.title(f"Per-User Latency Optimization (K={K})")
    plt.xticks(sparse_ticks); plt.grid(True, axis='y', linestyle='--', alpha=0.4, zorder=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_latency_comparison.png"), dpi=300); plt.close()

    # 2) POPULATION HISTOGRAM + CDF
    fig, ax1 = plt.subplots(figsize=(10, 6))
    combined_data = np.concatenate([initial_latency, final_latency])
    bins = np.linspace(np.min(combined_data), np.max(combined_data), 40)
    ax1.hist(initial_latency, bins=bins, alpha=0.3, color='gray', label="Initial (Freq)")
    ax1.hist(final_latency, bins=bins, alpha=0.5, color='forestgreen', label="Final (Freq)")
    ax1.set_xlabel("Latency Value"); ax1.set_ylabel("Number of Users")
    
    ax2 = ax1.twinx()
    for d, c, ls in zip([initial_latency, final_latency], ['red', 'darkgreen'], ['--', '-']):
        sorted_d = np.sort(d)
        y = np.arange(len(sorted_d)) / float(len(sorted_d) - 1)
        ax2.plot(sorted_d, y, color=c, linestyle=ls, linewidth=2, label=f"CDF {'Init' if ls=='--' else 'Final'}")
    ax2.set_ylabel("Cumulative Probability"); ax2.set_ylim(0, 1.05)
    plt.title("Latency Stats (Hist + CDF)")
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9)); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_latency_histogram.png"), dpi=300); plt.close()

    # 3) ASYNCHRONALITY DISTRIBUTION (Pairs Hist)
    init_diffs = np.abs(initial_latency[:, None] - initial_latency[None, :])
    final_diffs = np.abs(final_latency[:, None] - final_latency[None, :])
    mask = ~np.eye(K, dtype=bool)
    init_vals, final_vals = init_diffs[mask], final_diffs[mask]

    plt.figure(figsize=(10, 6))
    plt.hist(init_vals, bins=50, density=True, alpha=0.4, color='red', label='Initial Async')
    plt.hist(final_vals, bins=50, density=True, alpha=0.6, color='skyblue', label='Final Async')
    plt.title(f"Global Asynchronality Distribution (K={K} users, {len(init_vals)} pairs)")
    plt.xlabel("Latency Difference "); plt.ylabel("Probability Density"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_async_distribution.png"), dpi=300); plt.close()

    # 4) HEATMAPS
    vmin = min(np.min(init_vals), np.min(final_vals))
    vmax = max(np.max(init_vals), np.max(final_vals))
    
    def save_hm(m, title, fname):
        plt.figure(figsize=(max(8, K*0.12), max(6, K*0.1)))
        m_masked = np.ma.masked_where(~mask, m)
        cmap = plt.cm.plasma.copy(); cmap.set_bad(color='white')
        im = plt.imshow(m_masked, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
        plt.colorbar(im, label="|Li - Lj|"); plt.xticks(sparse_ticks); plt.yticks(sparse_ticks)
        plt.title(title); plt.xlabel("User j"); plt.ylabel("User i")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, fname), dpi=300); plt.close()

    save_hm(init_diffs, "Initial Asynchronality Heatmap", f"{prefix}_initial_async_heatmap.png")
    save_hm(final_diffs, "Final Asynchronality Heatmap", f"{prefix}_final_async_heatmap.png")
    
    
def adapt_training_dict_to_plot_format(post_training_data_dict):
    """
    Your plot_optimization_result_train expects:
        train_all_user_results[user][block] = list_of_dicts_over_n_kl

    The cleaned optimizer returns:
        post_training_data_dict["n_star"][user]   : list of chosen n_kl per block
        post_training_data_dict["R_star"][user]   : list of chosen R_fbl per block
        post_training_data_dict["F_star"][user]   : list of chosen F per block
    and does NOT store intermediate trajectories for each block unless you save them.

    So this adapter builds a minimal structure with ONE point per block:
        list_of_dicts_over_n_kl = [ {final point} ]
    """
    n_star = post_training_data_dict["n_star"]
    R_star = post_training_data_dict["R_star"]
    F_star = post_training_data_dict["F_star"]

    train_all_user_results = []

    K = len(n_star)
    for user in range(K):
        user_blocks = []
        L = len(n_star[user])
        for b in range(L):
            n_kl = int(n_star[user][b])
            R = float(R_star[user][b])
            Fmat = F_star[user][b]

            # Cannot reconstruct B_l reliably unless you saved it during training.
            # Keep it as None; plotting uses "Bits per sub-block length B/n_kl" and "R_fbl".
            # You can set it if you stored B_l in training results.
            point = {
                "n_kl": n_kl,
                "Bits per sub-block length B/n_kl": np.nan,  # fill if you have B_l
                "R_fbl": R,
                "F": Fmat,
            }
            user_blocks.append([point])  # list over "iterations"; here single point
        train_all_user_results.append(user_blocks)

    return train_all_user_results

# ============================================================
# Channel plots (unchanged; just fix variable name bugs)
# ============================================================
def plot_channel_magnitude1(channelsystem, users=None):
    if users is None or len(users) == 0:
        users = list(range(channelsystem.K))

    K = len(users)
    nrows = (K + 1) // 2
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, 3 * nrows))
    axes = axes.flatten()

    for idx, k in enumerate(users):
        H_mag_k = np.abs(channelsystem.H[k])  # (L, NR, NT)
        for rx in range(channelsystem.NR[k]):
            for tx in range(channelsystem.NT[k]):
                axes[idx].plot(range(channelsystem.L[k]), H_mag_k[:, rx, tx], label=f'Rx{rx+1}-Tx{tx+1}')

        axes[idx].set_title(f'User {k+1}')
        axes[idx].set_xlabel('Coherence block index (l)')
        axes[idx].set_ylabel('|H| magnitude')
        axes[idx].grid(True)
        axes[idx].legend(fontsize='x-small')

    for j in range(K, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def plot_channel_magnitude2(channelsystem, users=None):
    if users is None or len(users) == 0:
        users = list(range(channelsystem.K))

    K = len(users)
    nrows = (K + 1) // 2
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, 3 * nrows))
    axes = axes.flatten()

    for idx, k in enumerate(users):
        H_mag_k = np.abs(channelsystem.H[k])  # (L, NR, NT)
        H_plot_k = []
        for l in range(len(H_mag_k)):
            H_plot_k.append([H_mag_k[l]] * channelsystem.n_kl[k][l])  # n_kl is per block (list)

        H_plot_k = np.array(H_plot_k).reshape(-1, channelsystem.NR[k], channelsystem.NT[k])

        for rx in range(channelsystem.NR[k]):
            for tx in range(channelsystem.NT[k]):
                axes[idx].plot(range(channelsystem.n[k]), H_plot_k[:, rx, tx], label=f'Rx{rx+1}-Tx{tx+1}')

        axes[idx].set_title(f'User {k+1}')
        axes[idx].set_xlabel('Blocklength (n)')
        axes[idx].set_ylabel('|H| magnitude')
        axes[idx].grid(True)
        axes[idx].legend(fontsize='x-small')

    for j in range(K, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

# ============================================================
# Fix bugs: "system" -> "uplinksystem"
# ============================================================
def plot_capacity1(uplinksystem, users=None):
    if users is None or len(users) == 0:
        users = list(range(uplinksystem.K))

    K = len(users)
    nrows = (K + 1) // 2
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, 3 * nrows))
    axes = axes.flatten()

    for idx, k in enumerate(users):
        Ck = uplinksystem.C[k]
        axes[idx].plot(range(uplinksystem.L[k]), Ck)
        axes[idx].set_title(f'User {k+1}')
        axes[idx].set_xlabel('Coherence block index (l)')
        axes[idx].set_ylabel('Capacity')
        axes[idx].grid(True)

    for j in range(K, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def plot_capacity2(uplinksystem, users=None):
    if users is None or len(users) == 0:
        users = list(range(uplinksystem.K))

    K = len(users)
    nrows = (K + 1) // 2
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, 3 * nrows))
    axes = axes.flatten()

    for idx, k in enumerate(users):
        # Repeat per block by its n_kl
        Ck = np.concatenate([np.repeat([uplinksystem.C[k][l]], uplinksystem.n_kl[k][l]) for l in range(uplinksystem.L[k])])
        axes[idx].plot(range(uplinksystem.n[k]), Ck)
        axes[idx].set_title(f'User {k+1}')
        axes[idx].set_xlabel('Blocklength (n)')
        axes[idx].set_ylabel('Capacity')
        axes[idx].grid(True)

    for j in range(K, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def plot_dispersion1(uplinksystem, users=None):
    if users is None or len(users) == 0:
        users = list(range(uplinksystem.K))

    K = len(users)
    nrows = (K + 1) // 2
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, 3 * nrows))
    axes = axes.flatten()

    for idx, k in enumerate(users):
        Vk = uplinksystem.V[k]
        axes[idx].plot(range(uplinksystem.L[k]), Vk)
        axes[idx].set_title(f'User {k+1}')
        axes[idx].set_xlabel('Coherence block index (l)')
        axes[idx].set_ylabel('Dispersion')
        axes[idx].grid(True)

    for j in range(K, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def plot_dispersion2(uplinksystem, users=None):
    if users is None or len(users) == 0:
        users = list(range(uplinksystem.K))

    K = len(users)
    nrows = (K + 1) // 2
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, 3 * nrows))
    axes = axes.flatten()

    for idx, k in enumerate(users):
        Vk = np.concatenate([np.repeat([uplinksystem.V[k][l]], uplinksystem.n_kl[k][l]) for l in range(uplinksystem.L[k])])
        axes[idx].plot(range(uplinksystem.n[k]), Vk)
        axes[idx].set_title(f'User {k+1}')
        axes[idx].set_xlabel('Blocklength (n)')
        axes[idx].set_ylabel('Dispersion')
        axes[idx].grid(True)

    for j in range(K, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def plot_rate_fbl1(uplinksystem, users=None):
    if users is None or len(users) == 0:
        users = list(range(uplinksystem.K))

    K = len(users)
    nrows = (K + 1) // 2
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, 3 * nrows))
    axes = axes.flatten()

    for idx, k in enumerate(users):
        Rk = uplinksystem.R_fbl[k]
        axes[idx].plot(range(uplinksystem.L[k]), Rk)
        axes[idx].set_title(f'User {k+1}')
        axes[idx].set_xlabel('Coherence block index (l)')
        axes[idx].set_ylabel('Rate (FBL)')
        axes[idx].grid(True)

    for j in range(K, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def plot_rate_fbl2(uplinksystem, users=None):
    if users is None or len(users) == 0:
        users = list(range(uplinksystem.K))

    K = len(users)
    nrows = (K + 1) // 2
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, 3 * nrows))
    axes = axes.flatten()

    for idx, k in enumerate(users):
        Rk = np.concatenate([np.repeat([uplinksystem.R_fbl[k][l]], uplinksystem.n_kl[k][l]) for l in range(uplinksystem.L[k])])
        axes[idx].plot(range(uplinksystem.n[k]), Rk)
        axes[idx].set_title(f'User {k+1}')
        axes[idx].set_xlabel('Blocklength (n)')
        axes[idx].set_ylabel('Rate (FBL)')
        axes[idx].grid(True)

    for j in range(K, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def plot_2(uplinksystem):
    plot_capacity2(uplinksystem)
    plot_dispersion2(uplinksystem)
    plot_rate_fbl2(uplinksystem)


def _safe_db(values):
    arr = np.asarray(values, dtype=float)
    out = 10.0 * np.log10(np.maximum(arr, 1e-30))
    if np.isscalar(values):
        return float(out)
    return out


def _load_interference_diag(payload):
    if not payload:
        return None
    return {
        "blocks_per_user": [int(v) for v in payload.get("blocks_per_user", [])],
        "signal": np.asarray(payload.get("signal", []), dtype=float),
        "total_interference": np.asarray(payload.get("total_interference", []), dtype=float),
        "noise": np.asarray(payload.get("noise", []), dtype=float),
        "sinr_db": np.asarray(payload.get("sinr_db", []), dtype=float),
        "pairwise_block": np.asarray(payload.get("pairwise_block", []), dtype=float),
        "avg_pairwise_power": np.asarray(payload.get("avg_pairwise_power", []), dtype=float),
        "avg_pairwise_inr_db": np.asarray(payload.get("avg_pairwise_inr_db", []), dtype=float),
        "avg_pairwise_share": np.asarray(payload.get("avg_pairwise_share", []), dtype=float),
        "worst_block": int(payload.get("worst_block", -1)),
    }


def _combined_finite_limits(*matrices):
    finite_parts = []
    for matrix in matrices:
        arr = np.asarray(matrix, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size > 0:
            finite_parts.append(finite.reshape(-1))
    if not finite_parts:
        return None, None
    stacked = np.concatenate(finite_parts)
    return float(np.min(stacked)), float(np.max(stacked))


def _imshow_with_shared_scale(ax, matrix, title, cbar_label, *, cmap="viridis", center_zero=False, vmin=None, vmax=None):
    mat = np.asarray(matrix, dtype=float)
    masked = np.ma.masked_invalid(mat)
    norm = None
    if center_zero:
        finite = mat[np.isfinite(mat)]
        if finite.size > 0:
            bound = max(abs(float(np.min(finite))), abs(float(np.max(finite))), 1e-12)
            norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
    im = ax.imshow(masked, aspect="auto", interpolation="none", cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Interferer user")
    ax.set_ylabel("Victim user")
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_yticks(np.arange(mat.shape[0]))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)


def plot_interference_heatmaps(uplinksystem, figs_dir):
    os.makedirs(figs_dir, exist_ok=True)
    diag = collect_uplink_interference_diagnostics(uplinksystem)
    avg_power_db = _safe_db(diag["avg_pairwise_power"])
    avg_inr_db = np.asarray(diag["avg_pairwise_inr_db"], dtype=float)
    share_pct = 100.0 * np.asarray(diag["avg_pairwise_share"], dtype=float)
    worst_block = int(diag["worst_block"])
    worst_block_mat = None
    if worst_block >= 0:
        worst_block_mat = _safe_db(np.asarray(diag["pairwise_block"][worst_block], dtype=float))

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    _imshow_with_shared_scale(axes[0, 0], avg_power_db, "Average pairwise interference power", "dB")
    _imshow_with_shared_scale(axes[0, 1], avg_inr_db, "Average interference-to-noise ratio", "INR (dB)")
    _imshow_with_shared_scale(axes[1, 0], share_pct, "Average interference share", "% of victim interference")
    if worst_block_mat is not None:
        _imshow_with_shared_scale(
            axes[1, 1],
            worst_block_mat,
            f"Worst block interference map (block {worst_block})",
            "dB",
        )
    else:
        axes[1, 1].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "interference_heatmaps.png"), dpi=250)
    plt.close(fig)


def plot_interference_before_after_heatmaps(result, figs_dir):
    os.makedirs(figs_dir, exist_ok=True)
    initial_diag = _load_interference_diag(result.get("initial_interference_diag"))
    final_diag = _load_interference_diag(result.get("final_interference_diag"))
    if initial_diag is None or final_diag is None:
        return

    initial_power_db = _safe_db(initial_diag["avg_pairwise_power"])
    final_power_db = _safe_db(final_diag["avg_pairwise_power"])
    delta_power_db = final_power_db - initial_power_db
    initial_inr_db = np.asarray(initial_diag["avg_pairwise_inr_db"], dtype=float)
    final_inr_db = np.asarray(final_diag["avg_pairwise_inr_db"], dtype=float)
    delta_inr_db = final_inr_db - initial_inr_db
    power_vmin, power_vmax = _combined_finite_limits(initial_power_db, final_power_db)
    inr_vmin, inr_vmax = _combined_finite_limits(initial_inr_db, final_inr_db)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    _imshow_with_shared_scale(
        axes[0, 0],
        initial_power_db,
        "Avg interference power before optimization",
        "dB",
        vmin=power_vmin,
        vmax=power_vmax,
    )
    _imshow_with_shared_scale(
        axes[0, 1],
        final_power_db,
        "Avg interference power after optimization",
        "dB",
        vmin=power_vmin,
        vmax=power_vmax,
    )
    _imshow_with_shared_scale(
        axes[0, 2],
        delta_power_db,
        "Interference power change (after - before)",
        "dB",
        cmap="RdBu_r",
        center_zero=True,
    )
    _imshow_with_shared_scale(
        axes[1, 0],
        initial_inr_db,
        "Avg INR before optimization",
        "INR (dB)",
        vmin=inr_vmin,
        vmax=inr_vmax,
    )
    _imshow_with_shared_scale(
        axes[1, 1],
        final_inr_db,
        "Avg INR after optimization",
        "INR (dB)",
        vmin=inr_vmin,
        vmax=inr_vmax,
    )
    _imshow_with_shared_scale(
        axes[1, 2],
        delta_inr_db,
        "INR change (after - before)",
        "dB",
        cmap="RdBu_r",
        center_zero=True,
    )

    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "interference_before_after_heatmaps.png"), dpi=250)
    plt.close(fig)


def plot_per_user_interference_profiles(uplinksystem, figs_dir):
    os.makedirs(figs_dir, exist_ok=True)
    diag = collect_uplink_interference_diagnostics(uplinksystem)
    signal_db = _safe_db(diag["signal"])
    interference_db = _safe_db(diag["total_interference"])
    noise_db = _safe_db(diag["noise"])
    sinr_db = np.asarray(diag["sinr_db"], dtype=float)
    pairwise_block = np.asarray(diag["pairwise_block"], dtype=float)
    K = int(uplinksystem.K)

    fig, axes = plt.subplots(K, 2, figsize=(15, max(4 * K, 5)), squeeze=False)
    for k in range(K):
        blocks = np.arange(len(uplinksystem.n_kl[k]))
        ax_left = axes[k, 0]
        ax_left.plot(blocks, signal_db[k, : len(blocks)], "o-", label="Signal", color="tab:blue")
        ax_left.plot(blocks, interference_db[k, : len(blocks)], "o-", label="Total interference", color="tab:red")
        ax_left.plot(blocks, noise_db[k, : len(blocks)], "o--", label="Noise", color="tab:gray")
        ax_left.set_title(f"User {k} signal/interference/noise profile")
        ax_left.set_xlabel("Block index")
        ax_left.set_ylabel("Power (dB)")
        ax_left.grid(True, alpha=0.3)

        ax_left_r = ax_left.twinx()
        ax_left_r.plot(blocks, sinr_db[k, : len(blocks)], "s-", color="tab:green", label="SINR")
        ax_left_r.set_ylabel("SINR (dB)")

        handles_l, labels_l = ax_left.get_legend_handles_labels()
        handles_r, labels_r = ax_left_r.get_legend_handles_labels()
        ax_left.legend(handles_l + handles_r, labels_l + labels_r, fontsize=8, loc="best")

        ax_right = axes[k, 1]
        contrib = pairwise_block[: len(blocks), k, :]
        bottom = np.zeros(len(blocks), dtype=float)
        for j in range(K):
            if j == k:
                continue
            vals = np.nan_to_num(contrib[:, j], nan=0.0)
            if np.allclose(vals, 0.0):
                continue
            ax_right.bar(blocks, vals, bottom=bottom, alpha=0.75, label=f"Interferer {j}")
            bottom += vals
        ax_right.set_title(f"User {k} interference contributors")
        ax_right.set_xlabel("Block index")
        ax_right.set_ylabel("Interference power")
        ax_right.grid(True, axis="y", alpha=0.3)
        if K <= 6:
            ax_right.legend(fontsize=8, loc="best")

    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "per_user_interference_profiles.png"), dpi=250)
    plt.close(fig)


def plot_per_user_interference_before_after(result, figs_dir):
    os.makedirs(figs_dir, exist_ok=True)
    initial_diag = _load_interference_diag(result.get("initial_interference_diag"))
    final_diag = _load_interference_diag(result.get("final_interference_diag"))
    if initial_diag is None or final_diag is None:
        return

    K = max(
        len(initial_diag.get("blocks_per_user", [])),
        len(final_diag.get("blocks_per_user", [])),
        len(result.get("n_kl", result.get("final_n_kl", []))),
    )
    if K == 0:
        return

    initial_interf_db = _safe_db(initial_diag["total_interference"])
    final_interf_db = _safe_db(final_diag["total_interference"])
    initial_sinr_db = np.asarray(initial_diag["sinr_db"], dtype=float)
    final_sinr_db = np.asarray(final_diag["sinr_db"], dtype=float)
    initial_signal_db = _safe_db(initial_diag["signal"])
    final_signal_db = _safe_db(final_diag["signal"])

    fig, axes = plt.subplots(K, 2, figsize=(16, max(4 * K, 5)), squeeze=False)
    for k in range(K):
        init_blocks = np.arange(int(initial_diag["blocks_per_user"][k])) if k < len(initial_diag["blocks_per_user"]) else np.asarray([], dtype=int)
        final_blocks = np.arange(int(final_diag["blocks_per_user"][k])) if k < len(final_diag["blocks_per_user"]) else np.asarray([], dtype=int)

        ax_left = axes[k, 0]
        if len(init_blocks) > 0:
            ax_left.plot(init_blocks, initial_interf_db[k, : len(init_blocks)], "o--", color="tab:red", alpha=0.8, label="Initial interference")
            ax_left.plot(init_blocks, initial_signal_db[k, : len(init_blocks)], "o--", color="tab:blue", alpha=0.5, label="Initial signal")
        if len(final_blocks) > 0:
            ax_left.plot(final_blocks, final_interf_db[k, : len(final_blocks)], "o-", color="tab:red", label="Final interference")
            ax_left.plot(final_blocks, final_signal_db[k, : len(final_blocks)], "o-", color="tab:blue", alpha=0.7, label="Final signal")
        ax_left.set_title(f"User {k} interference before vs after")
        ax_left.set_xlabel("Block index")
        ax_left.set_ylabel("Power (dB)")
        ax_left.grid(True, alpha=0.3)
        ax_left.legend(fontsize=8, loc="best")

        ax_right = axes[k, 1]
        if len(init_blocks) > 0:
            ax_right.plot(init_blocks, initial_sinr_db[k, : len(init_blocks)], "s--", color="tab:green", alpha=0.8, label="Initial SINR")
        if len(final_blocks) > 0:
            ax_right.plot(final_blocks, final_sinr_db[k, : len(final_blocks)], "s-", color="tab:green", label="Final SINR")
        ax_right.set_title(f"User {k} SINR before vs after")
        ax_right.set_xlabel("Block index")
        ax_right.set_ylabel("SINR (dB)")
        ax_right.grid(True, alpha=0.3)
        ax_right.legend(fontsize=8, loc="best")

    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "per_user_interference_before_after.png"), dpi=250)
    plt.close(fig)

# ============================================================
# Optimization plots (make them consistent with NEW pipeline)
# ============
# ============================================================
# Your existing per-block curve plots can stay,
# but they REQUIRE per-block trajectories.
#
# If you want those exact plots, you must save `results` from
# optimize_subblocklength_precoder for each block during training.
# ============================================================

if __name__ == "__main__":
    print(train_result_path)
