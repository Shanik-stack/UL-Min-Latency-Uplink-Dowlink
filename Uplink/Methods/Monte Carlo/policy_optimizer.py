import copy
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

METHOD_DIR = Path(__file__).resolve().parent
LINK_ROOT = METHOD_DIR.parents[1]
BASELINE_DIR = METHOD_DIR.parent / "Convergence per sweep"
for path in (METHOD_DIR, LINK_ROOT, BASELINE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from UplinkSystem import UplinkSystem
from advanced_methods_common import (
    apply_training_solution,
    ensure_blocks_up_to,
    estimate_initial_random_precoder_schedule,
)
from config_loader import get_config
from precoder_models import (
    DEVICE,
    build_user_precoder_net_with_blocklength,
    export_user_model_specs,
    export_user_model_states,
    infer_precoder_numpy_with_blocklength,
    infer_precoder_torch_with_blocklength,
)


def _to_complex_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.astype(np.complex64, copy=False)
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(np.complex64, copy=False)
    return np.asarray(x, dtype=np.complex64)


def _q_inv_torch(epsilon: float, device: torch.device = DEVICE) -> torch.Tensor:
    normal = torch.distributions.Normal(
        torch.tensor(0.0, device=device, dtype=torch.float64),
        torch.tensor(1.0, device=device, dtype=torch.float64),
    )
    p = torch.tensor(1.0 - float(epsilon), device=device, dtype=torch.float64)
    p = torch.clamp(p, 1e-12, 1.0 - 1e-12)
    return normal.icdf(p).to(dtype=torch.float32)


def _compute_r_fbl_torch(
    H: torch.Tensor,
    Fmat: torch.Tensor,
    epsilon: float,
    n_kl: int,
    noise_plus_interference_cov: torch.Tensor,
) -> torch.Tensor:
    Nr = H.shape[0]
    I = torch.eye(Nr, dtype=torch.complex64, device=H.device)
    noise_cov = noise_plus_interference_cov.to(device=H.device, dtype=torch.complex64)
    noise_cov = 0.5 * (noise_cov + noise_cov.conj().transpose(1, 0))
    noise_cov = noise_cov + (1e-6 * I)

    HF = H @ Fmat
    chol = torch.linalg.cholesky(noise_cov)
    G = torch.linalg.solve(chol, HF)
    A = G @ G.conj().transpose(1, 0)
    A = 0.5 * (A + A.conj().transpose(1, 0))

    sign, logdet = torch.linalg.slogdet(I + A)
    if torch.any(torch.abs(sign) <= 1e-12):
        raise RuntimeError("Non-positive logdet sign while evaluating uplink Monte Carlo rate.")

    C = (logdet / np.log(2.0)).real
    eigvals = torch.linalg.eigvalsh(A)
    V = torch.sum(eigvals * (eigvals + 2.0) / (eigvals + 1.0) ** 2).real * (np.log2(np.e) ** 2)
    R = C - torch.sqrt(V / float(max(int(n_kl), 1))) * _q_inv_torch(float(epsilon), device=H.device)
    return R.real


def _compute_r_fbl_np(
    H: np.ndarray,
    Fmat: np.ndarray,
    sigma2: float,
    epsilon: float,
    n_kl: int,
    noise_plus_interference_cov: np.ndarray,
) -> float:
    from Optimizer_per_block import _compute_R_fbl_np

    return _compute_R_fbl_np(
        H=np.asarray(H, dtype=np.complex64),
        F=np.asarray(Fmat, dtype=np.complex64),
        sigma2=float(sigma2),
        epsilon=float(epsilon),
        n_kl=int(n_kl),
        noise_plus_interference_cov=np.asarray(noise_plus_interference_cov, dtype=np.complex128),
    )


def _build_training_n_kl_values(
    T_ref: int,
    n_min: int,
    fine_step: int,
    coarse_step: int,
) -> list[int]:
    T_ref = int(T_ref)
    n_min = min(int(n_min), T_ref)
    fine_step = max(1, int(fine_step))
    coarse_step = max(fine_step, int(coarse_step))
    if T_ref <= n_min:
        return [T_ref]
    if (T_ref - n_min) < coarse_step:
        return list(range(T_ref, n_min - 1, -fine_step))

    n_values = [T_ref]
    current = int(T_ref)
    while current - coarse_step >= n_min:
        current -= coarse_step
        n_values.append(int(current))

    if n_values[-1] != n_min:
        current = int(n_values[-1]) - fine_step
        while current >= n_min:
            n_values.append(int(current))
            current -= fine_step

    seen: set[int] = set()
    ordered_values: list[int] = []
    for n_kl in n_values:
        if int(n_kl) in seen:
            continue
        seen.add(int(n_kl))
        ordered_values.append(int(n_kl))
    return ordered_values


def build_training_dataset(
    cfg_name: str,
    train_seeds: Sequence[int],
) -> list[list[dict]]:
    system_params, sim_cfg = get_config(cfg_name)
    K = int(system_params["K"])
    samples_by_user: list[list[dict]] = [[] for _ in range(K)]
    n_min = int(sim_cfg["n_kl_min"])
    n_step = int(sim_cfg["n_kl_step"])
    min_bits_required = max(1, int(sim_cfg.get("precoder_net_train_min_bits_required", 1)))
    blocks_per_seed = max(1, int(sim_cfg.get("precoder_net_train_blocks_per_seed", 1)))
    coarse_step = max(int(n_step), int(sim_cfg.get("precoder_net_train_n_kl_coarse_step", 5)))

    for seed in train_seeds:
        print(f"\n================ RAW TRAINING DATA seed={seed} ================")
        uplinksystem = UplinkSystem(system_params, seed=int(seed))
        ensure_blocks_up_to(uplinksystem, int(blocks_per_seed) - 1)

        for k in range(K):
            T_ref = int(uplinksystem.T[k])
            P_user = float(uplinksystem.P[k])
            sigma2 = float(uplinksystem.sigma2[k])
            epsilon = float(uplinksystem.epsilon[k])
            n_values = _build_training_n_kl_values(
                T_ref=int(T_ref),
                n_min=int(n_min),
                fine_step=int(n_step),
                coarse_step=int(coarse_step),
            )
            max_blocks = min(int(blocks_per_seed), len(uplinksystem.H[k]))
            for block in range(max_blocks):
                H_block = np.asarray(uplinksystem.H[k][int(block)], dtype=np.complex64)
                noise_plus_interference_cov = uplinksystem.get_interference_plus_noise_covariance(k, int(block))

                for n_kl in n_values:
                    samples_by_user[k].append(
                        {
                            "seed": int(seed),
                            "user": int(k),
                            "block": int(block),
                            "H": H_block,
                            "n_kl": int(n_kl),
                            "T_ref": int(T_ref),
                            "min_bits_required": int(min_bits_required),
                            "P": float(P_user),
                            "sigma2": float(sigma2),
                            "epsilon": float(epsilon),
                            "noise_plus_interference_cov": np.asarray(noise_plus_interference_cov, dtype=np.complex128),
                        }
                    )

    return samples_by_user


def summarize_training_dataset(samples_by_user: Sequence[Sequence[dict]]) -> dict:
    total_examples = int(sum(len(samples) for samples in samples_by_user))
    global_examples_by_n_kl: dict[int, int] = {}
    global_examples_by_min_bits_required: dict[int, int] = {}
    examples_by_seed: dict[int, int] = {}
    global_examples_by_block: dict[int, int] = {}
    per_user_summary = []

    for user_idx, samples in enumerate(samples_by_user):
        user_examples_by_n_kl: dict[int, int] = {}
        user_examples_by_min_bits_required: dict[int, int] = {}
        user_examples_by_seed: dict[int, int] = {}
        user_examples_by_block: dict[int, int] = {}
        for sample in samples:
            n_kl = int(sample["n_kl"])
            min_bits_required = int(sample.get("min_bits_required", 1))
            seed = int(sample["seed"])
            block = int(sample.get("block", 0))
            user_examples_by_n_kl[n_kl] = user_examples_by_n_kl.get(n_kl, 0) + 1
            user_examples_by_min_bits_required[min_bits_required] = (
                user_examples_by_min_bits_required.get(min_bits_required, 0) + 1
            )
            user_examples_by_seed[seed] = user_examples_by_seed.get(seed, 0) + 1
            user_examples_by_block[block] = user_examples_by_block.get(block, 0) + 1
            global_examples_by_n_kl[n_kl] = global_examples_by_n_kl.get(n_kl, 0) + 1
            global_examples_by_min_bits_required[min_bits_required] = (
                global_examples_by_min_bits_required.get(min_bits_required, 0) + 1
            )
            examples_by_seed[seed] = examples_by_seed.get(seed, 0) + 1
            global_examples_by_block[block] = global_examples_by_block.get(block, 0) + 1

        per_user_summary.append(
            {
                "user": int(user_idx),
                "total_examples": int(len(samples)),
                "unique_n_kl": [int(v) for v in sorted(user_examples_by_n_kl)],
                "examples_by_n_kl": {str(int(k)): int(v) for k, v in sorted(user_examples_by_n_kl.items())},
                "examples_by_min_bits_required": {
                    str(int(k)): int(v) for k, v in sorted(user_examples_by_min_bits_required.items())
                },
                "examples_by_seed": {str(int(k)): int(v) for k, v in sorted(user_examples_by_seed.items())},
                "examples_by_block": {str(int(k)): int(v) for k, v in sorted(user_examples_by_block.items())},
            }
        )

    return {
        "total_examples": int(total_examples),
        "num_users": int(len(samples_by_user)),
        "examples_by_seed": {str(int(k)): int(v) for k, v in sorted(examples_by_seed.items())},
        "global_examples_by_block": {str(int(k)): int(v) for k, v in sorted(global_examples_by_block.items())},
        "global_examples_by_n_kl": {str(int(k)): int(v) for k, v in sorted(global_examples_by_n_kl.items())},
        "global_examples_by_min_bits_required": {
            str(int(k)): int(v) for k, v in sorted(global_examples_by_min_bits_required.items())
        },
        "per_user": per_user_summary,
    }


def _summarize_selected_n_kl(n_star: Sequence[Sequence[int]]) -> dict[str, object]:
    global_counts: dict[int, int] = {}
    per_user = []
    for user_idx, user_n in enumerate(n_star):
        user_counts: dict[int, int] = {}
        for n_kl in user_n:
            n_val = int(n_kl)
            user_counts[n_val] = user_counts.get(n_val, 0) + 1
            global_counts[n_val] = global_counts.get(n_val, 0) + 1
        per_user.append(
            {
                "user": int(user_idx),
                "selected_examples_by_n_kl": {str(int(k)): int(v) for k, v in sorted(user_counts.items())},
            }
        )
    return {
        "global_selected_examples_by_n_kl": {str(int(k)): int(v) for k, v in sorted(global_counts.items())},
        "per_user": per_user,
    }


def _aggregate_epoch_means(per_user_histories: Sequence[Sequence[float]]) -> list[float]:
    max_len = max((len(history) for history in per_user_histories), default=0)
    aggregated: list[float] = []
    for epoch_idx in range(max_len):
        values = [float(history[epoch_idx]) for history in per_user_histories if epoch_idx < len(history)]
        aggregated.append(float(np.mean(values)) if values else 0.0)
    return aggregated


def _build_post_training_summary(
    train_eval_system: UplinkSystem,
    train_eval_post: dict,
    training_history: dict,
    *,
    train_eval_seed: int,
    epochs: int,
    min_bits_required: int,
    initial_baseline: dict | None = None,
) -> dict:
    per_user_lagrangian = training_history.get("per_user_lagrangian", [])
    per_user_rate = training_history.get("per_user_rate", [])
    per_user_rate_violation = training_history.get("avg_rate_violation", [])
    per_user_power_violation = training_history.get("avg_power_violation", [])
    avg_lagrangian = training_history.get("avg_lagrangian", [])
    avg_user_rate = training_history.get("avg_user_rate", [])
    avg_rate_violation = training_history.get("avg_rate_violation_over_users", [])
    avg_power_violation = training_history.get("avg_power_violation_over_users", [])
    initial_latency = (
        [float(v) for v in initial_baseline.get("initial_latency", [])]
        if isinstance(initial_baseline, dict)
        else [float(v) for v in train_eval_system.initial_latency]
    )
    initial_n = (
        [int(v) for v in initial_baseline.get("initial_n", [])]
        if isinstance(initial_baseline, dict)
        else [int(v) for v in train_eval_system.n]
    )
    initial_n_kl = (
        [[int(x) for x in user_blocks] for user_blocks in initial_baseline.get("initial_n_kl", [])]
        if isinstance(initial_baseline, dict)
        else [list(map(int, user_blocks)) for user_blocks in train_eval_system.n_kl]
    )
    initial_B_kl = (
        [[int(x) for x in user_bits] for user_bits in initial_baseline.get("initial_B_kl", [])]
        if isinstance(initial_baseline, dict)
        else [[int(train_eval_system.B[k])] for k in range(int(train_eval_system.K))]
    )
    final_latency = [float(v) for v in train_eval_system.latency]
    initial_total_latency = float(sum(initial_latency))
    final_total_latency = float(sum(final_latency))
    total_latency_reduction_percent = (
        float(((initial_total_latency - final_total_latency) / initial_total_latency) * 100.0)
        if initial_total_latency > 0.0
        else 0.0
    )
    initial_selected_n_summary = _summarize_selected_n_kl(initial_n_kl)
    selected_n_summary = _summarize_selected_n_kl(train_eval_post.get("n_star", []))

    return {
        "train_eval_seed": int(train_eval_seed),
        "epochs_requested": int(epochs),
        "train_min_bits_required": int(min_bits_required),
        "per_user_num_epochs": [int(len(history)) for history in per_user_lagrangian],
        "per_user_final_lagrangian": [
            float(history[-1]) if len(history) > 0 else 0.0 for history in per_user_lagrangian
        ],
        "per_user_best_lagrangian": [
            float(min(history)) if len(history) > 0 else 0.0 for history in per_user_lagrangian
        ],
        "per_user_final_rate": [float(history[-1]) if len(history) > 0 else 0.0 for history in per_user_rate],
        "per_user_final_rate_violation": [
            float(history[-1]) if len(history) > 0 else 0.0 for history in per_user_rate_violation
        ],
        "per_user_final_power_violation": [
            float(history[-1]) if len(history) > 0 else 0.0 for history in per_user_power_violation
        ],
        "final_avg_lagrangian": float(avg_lagrangian[-1]) if len(avg_lagrangian) > 0 else 0.0,
        "best_avg_lagrangian": float(min(avg_lagrangian)) if len(avg_lagrangian) > 0 else 0.0,
        "final_avg_user_rate": float(avg_user_rate[-1]) if len(avg_user_rate) > 0 else 0.0,
        "best_avg_user_rate": float(max(avg_user_rate)) if len(avg_user_rate) > 0 else 0.0,
        "final_avg_rate_violation": float(avg_rate_violation[-1]) if len(avg_rate_violation) > 0 else 0.0,
        "best_avg_rate_violation": float(min(avg_rate_violation)) if len(avg_rate_violation) > 0 else 0.0,
        "final_avg_power_violation": float(avg_power_violation[-1]) if len(avg_power_violation) > 0 else 0.0,
        "best_avg_power_violation": float(min(avg_power_violation)) if len(avg_power_violation) > 0 else 0.0,
        "per_user_final_loss": [
            float(history[-1]) if len(history) > 0 else 0.0 for history in per_user_lagrangian
        ],
        "per_user_best_loss": [
            float(min(history)) if len(history) > 0 else 0.0 for history in per_user_lagrangian
        ],
        "train_eval_initial_latency": initial_latency,
        "train_eval_final_latency": final_latency,
        "train_eval_initial_total_latency": float(initial_total_latency),
        "train_eval_final_total_latency": float(final_total_latency),
        "train_eval_total_latency_reduction_percent": float(total_latency_reduction_percent),
        "train_eval_initial_blocks_per_user": [int(len(v)) for v in initial_n_kl],
        "train_eval_initial_total_n_per_user": initial_n,
        "train_eval_initial_served_bits_per_user": [int(sum(bits)) for bits in initial_B_kl],
        "train_eval_initial_selected_n_kl_summary": initial_selected_n_summary,
        "train_eval_blocks_per_user": [int(v) for v in train_eval_post.get("L_out", [])],
        "train_eval_total_n_per_user": [int(v) for v in train_eval_system.n],
        "train_eval_served_bits_per_user": [
            int(sum(block_bits))
            for block_bits in train_eval_post.get("B_kl_star", [[] for _ in range(int(train_eval_system.K))])
        ],
        "train_eval_selected_n_kl_summary": selected_n_summary,
    }


def train_blocklength_aware_precoder_net(
    cfg_name: str,
    train_seeds: Sequence[int],
    *,
    epochs: int =20,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> dict:
    system_params, sim_cfg = get_config(cfg_name)
    K = int(system_params["K"])
    samples_by_user = build_training_dataset(cfg_name, train_seeds)
    dataset_summary = summarize_training_dataset(samples_by_user)
    min_bits_required = max(1, int(sim_cfg.get("precoder_net_train_min_bits_required", 1)))
    training_history = {
        "per_user_lagrangian": [[] for _ in range(K)],
        "per_user_rate": [[] for _ in range(K)],
        "avg_rate_violation": [[] for _ in range(K)],
        "avg_power_violation": [[] for _ in range(K)],
        "avg_lagrangian": [],
        "avg_user_rate": [],
        "avg_rate_violation_over_users": [],
        "avg_power_violation_over_users": [],
        "dataset_summary": dataset_summary,
        "training_objective": "lagrangian_user_finite_blocklength_rate_with_fixed_min_bits_objective",
    }

    user_models = []

    for k in range(K):
        Nr = int(system_params["NR"][k])
        Nt = int(system_params["NT"][k])
        dk = int(system_params["dk"][k])

        model = build_user_precoder_net_with_blocklength(Nr=Nr, Nt=Nt, dk=dk, device=DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
        samples = samples_by_user[k]
        user_lagrangian_history = training_history["per_user_lagrangian"][k]
        user_rate_history = training_history["per_user_rate"][k]
        user_rate_violation_history = training_history["avg_rate_violation"][k]
        user_power_violation_history = training_history["avg_power_violation"][k]
        lambda_rate = float(sim_cfg.get("initial_lambda_rate_constraint", 0.1))
        lambda_power = float(sim_cfg.get("initial_lambda_power_constraint", 0.01))
        lr_rate = float(sim_cfg.get("lr_rate_constraint", 1e-2))
        lr_power = float(sim_cfg.get("lr_power_constraint", 1e-3))

        if len(samples) == 0:
            user_models.append(model.eval())
            continue

        rng = np.random.default_rng(int(train_seeds[0]) + 17 * (k + 1))
        indices = np.arange(len(samples))

        print(
            f"\n================ PRECODER NET TRAIN USER {k} ================\n"
            f"Samples: {len(samples)} | epochs: {epochs} | batch_size: {batch_size}"
        )

        for epoch in range(int(epochs)):
            rng.shuffle(indices)
            epoch_term_sum = 0.0
            epoch_rate_sum = 0.0
            epoch_rate_violation_sum = 0.0
            epoch_power_violation_sum = 0.0
            epoch_sample_count = 0.0

            for start in range(0, len(indices), max(int(batch_size), 1)):
                batch_idx = indices[start:start + max(int(batch_size), 1)]
                optimizer.zero_grad()
                loss = torch.zeros((), dtype=torch.float32, device=DEVICE)
                batch_rate_violation = 0.0
                batch_power_violation = 0.0

                for idx in batch_idx:
                    sample = samples[int(idx)]
                    H_t = torch.tensor(sample["H"], dtype=torch.complex64, device=DEVICE)
                    noise_cov_t = torch.tensor(
                        sample["noise_plus_interference_cov"],
                        dtype=torch.complex64,
                        device=DEVICE,
                    )

                    pred_t = infer_precoder_torch_with_blocklength(
                        model,
                        H_t,
                        sample["n_kl"],
                        noise_cov_t,
                        float(sample["epsilon"]),
                        Nt=Nt,
                        dk=dk,
                        P=sample["P"],
                    )
                    rate = _compute_r_fbl_torch(
                        H_t,
                        pred_t,
                        epsilon=float(sample["epsilon"]),
                        n_kl=int(sample["n_kl"]),
                        noise_plus_interference_cov=noise_cov_t,
                    )
                    power = (torch.linalg.norm(pred_t, ord="fro") ** 2).real
                    required_rate = float(sample["min_bits_required"]) / float(max(int(sample["n_kl"]), 1))
                    rate_violation = torch.tensor(required_rate, dtype=torch.float32, device=DEVICE) - rate
                    power_violation = power - float(sample["P"])
                    rate_violation_pos = F.relu(rate_violation)
                    power_violation_pos = F.relu(power_violation)
                    term = (
                        -rate
                        + float(lambda_rate) * rate_violation_pos
                        + float(lambda_power) * power_violation_pos
                    )
                    loss = loss + term
                    batch_rate_violation += float(rate_violation_pos.detach().cpu())
                    batch_power_violation += float(power_violation_pos.detach().cpu())
                    epoch_term_sum += float(term.detach().cpu())
                    epoch_rate_sum += float(rate.detach().cpu())
                    epoch_rate_violation_sum += float(rate_violation_pos.detach().cpu())
                    epoch_power_violation_sum += float(power_violation_pos.detach().cpu())
                    epoch_sample_count += 1.0

                loss = loss / float(len(batch_idx))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                lambda_rate = max(0.0, lambda_rate + lr_rate * (batch_rate_violation / float(len(batch_idx))))
                lambda_power = max(0.0, lambda_power + lr_power * (batch_power_violation / float(len(batch_idx))))

            avg_lagrangian = float(epoch_term_sum / max(epoch_sample_count, 1.0))
            avg_rate = float(epoch_rate_sum / max(epoch_sample_count, 1.0))
            avg_rate_violation = float(epoch_rate_violation_sum / max(epoch_sample_count, 1.0))
            avg_power_violation = float(epoch_power_violation_sum / max(epoch_sample_count, 1.0))
            user_lagrangian_history.append(avg_lagrangian)
            user_rate_history.append(avg_rate)
            user_rate_violation_history.append(avg_rate_violation)
            user_power_violation_history.append(avg_power_violation)
            print(
                f"Precoder-net user {k} epoch {epoch + 1}/{epochs}: "
                f"avg_rate={avg_rate:.6e} | "
                f"avg_rate_violation={avg_rate_violation:.6e} | "
                f"avg_power_violation={avg_power_violation:.6e} | "
                f"lagrangian={avg_lagrangian:.6e}"
            )

        user_models.append(model.eval())

    training_history["avg_lagrangian"] = _aggregate_epoch_means(training_history["per_user_lagrangian"])
    training_history["avg_user_rate"] = _aggregate_epoch_means(training_history["per_user_rate"])
    training_history["avg_rate_violation_over_users"] = _aggregate_epoch_means(training_history["avg_rate_violation"])
    training_history["avg_power_violation_over_users"] = _aggregate_epoch_means(training_history["avg_power_violation"])

    train_eval_seed = int(train_seeds[0]) if len(train_seeds) > 0 else 0
    train_eval_initial_baseline = estimate_initial_random_precoder_schedule(
        system_params,
        sim_cfg,
        seed=train_eval_seed,
    )
    train_eval_system = UplinkSystem(system_params, seed=train_eval_seed)
    train_eval_post = evaluate_blocklength_precoder_net(
        uplinksystem=train_eval_system,
        user_models=user_models,
        sim_cfg=get_config(cfg_name)[1],
        method_name="monte_carlo_precoder_net_train_eval",
    )
    post_training_summary = _build_post_training_summary(
        train_eval_system,
        train_eval_post,
        training_history,
        train_eval_seed=train_eval_seed,
        epochs=int(epochs),
        min_bits_required=int(min_bits_required),
        initial_baseline=train_eval_initial_baseline,
    )

    train_eval_post.update(
        {
            "train_seeds": [int(s) for s in train_seeds],
            "training_dataset_sizes": [len(v) for v in samples_by_user],
            "training_sample_counts_per_user": [len(v) for v in samples_by_user],
            "training_dataset_summary": dataset_summary,
            "post_training_summary": post_training_summary,
            "precoder_net_training_losses": [
                list(map(float, history)) for history in training_history["per_user_lagrangian"]
            ],
            "precoder_net_training_history": training_history,
            "user_model_specs": export_user_model_specs(
                system_params["NR"],
                system_params["NT"],
                system_params["dk"],
                uses_blocklength_input=True,
            ),
            "user_model_states": export_user_model_states(user_models),
            "precoder_parameterization": "shared_user_channel_n_interference_epsilon_to_precoder_mlp",
            "training_objective": "lagrangian_user_finite_blocklength_rate_with_fixed_min_bits_objective",
        }
    )
    return train_eval_post


def _build_precoder_net_snapshot(
    uplinksystem: UplinkSystem,
    user_models: Sequence[torch.nn.Module],
    block_idx: int,
) -> list[list[np.ndarray]]:
    ensure_blocks_up_to(uplinksystem, int(block_idx))
    snapshot: list[list[np.ndarray]] = []

    for k in range(int(uplinksystem.K)):
        user_blocks: list[np.ndarray] = []
        for l in range(int(block_idx) + 1):
            noise_cov = uplinksystem.get_interference_plus_noise_covariance(k, l)
            user_blocks.append(
                infer_precoder_numpy_with_blocklength(
                    user_models[k],
                    np.asarray(uplinksystem.H[k][l], dtype=np.complex64),
                    n_kl=int(uplinksystem.T[k]),
                    noise_plus_interference_cov=np.asarray(noise_cov, dtype=np.complex128),
                    epsilon=float(uplinksystem.epsilon[k]),
                    Nt=int(uplinksystem.NT[k]),
                    dk=int(uplinksystem.dk[k]),
                    P=float(uplinksystem.P[k]),
                    device=DEVICE,
                )
            )
        snapshot.append(user_blocks)

    return snapshot


def evaluate_blocklength_precoder_net(
    uplinksystem: UplinkSystem,
    user_models: Sequence[torch.nn.Module],
    sim_cfg: dict,
    *,
    method_name: str,
) -> dict:
    K = int(uplinksystem.K)

    L_out = [1] * K
    n_star = [[] for _ in range(K)]
    F_star = [[] for _ in range(K)]
    R_star = [[] for _ in range(K)]
    all_user_block_results = [[] for _ in range(K)]
    B_used_star = [[] for _ in range(K)]
    B_kl_star = [[] for _ in range(K)]

    n_kl_min = int(sim_cfg["n_kl_min"])
    n_kl_step = int(sim_cfg["n_kl_step"])

    for k in range(K):
        print(f"\n================ PRECODER NET EVAL USER {k} ================")
        B_rem = int(uplinksystem.B[k])
        ell = 0

        while B_rem > 0:
            ensure_blocks_up_to(uplinksystem, ell)

            H_kl = np.asarray(uplinksystem.H[k][ell], dtype=np.complex64)
            T_ref = int(uplinksystem.T[k])
            P = float(uplinksystem.P[k])
            sigma2 = float(uplinksystem.sigma2[k])
            epsilon = float(uplinksystem.epsilon[k])

            snapshot_full = _build_precoder_net_snapshot(uplinksystem, user_models, ell)

            print(f"\n--- PRECODER NET User {k}, Block {ell}, B_rem={B_rem} ---")

            B_try = int(B_rem)
            B_used = None
            S_block = []

            for attempt in range(12):
                cov_T_input = uplinksystem.get_interference_plus_noise_covariance(k, ell, F_override=snapshot_full)
                F_T = infer_precoder_numpy_with_blocklength(
                    user_models[k],
                    H_kl,
                    n_kl=T_ref,
                    noise_plus_interference_cov=np.asarray(cov_T_input, dtype=np.complex128),
                    epsilon=epsilon,
                    Nt=int(uplinksystem.NT[k]),
                    dk=int(uplinksystem.dk[k]),
                    P=P,
                    device=DEVICE,
                )
                snapshot_candidate = copy.deepcopy(snapshot_full)
                snapshot_candidate[k][ell] = F_T
                cov_T = uplinksystem.get_interference_plus_noise_covariance(k, ell, F_override=snapshot_candidate)
                R_T = _compute_r_fbl_np(H_kl, F_T, sigma2, epsilon, T_ref, cov_T)
                rate_violation = (B_try / float(T_ref)) - R_T

                print(
                    f"Attempt {attempt + 1}: n=T={T_ref}, B={B_try}, "
                    f"R_fbl={R_T}, rate_violation={max(0.0, rate_violation)}"
                )

                if rate_violation <= 0.0:
                    B_used = int(B_try)
                    S_block.append(
                        {
                            "n_kl": int(T_ref),
                            "n": int(T_ref),
                            "B_l": int(B_used),
                            "Bits per sub-block length B/n_kl": float(B_used) / float(T_ref),
                            "F": torch.tensor(F_T, dtype=torch.complex64),
                            "R_fbl": float(R_T),
                            "F_power": float(np.linalg.norm(F_T, "fro") ** 2),
                            "lambda_rate": 0.0,
                            "lambda_power": 0.0,
                            "loss_curve": [],
                            "method": method_name,
                        }
                    )
                    break

                B_new = int(np.floor(T_ref * R_T))
                B_new = max(0, min(B_new, B_try))
                if B_new == B_try:
                    B_used = 0
                    break
                B_try = B_new

            if B_used is None or B_used <= 0:
                print(f">>> STOP precoder-net user {k} at block {ell}: no feasible T-point.")
                break

            best_n = int(T_ref)
            best_R = float(S_block[-1]["R_fbl"])
            best_F = S_block[-1]["F"]

            n_kl = int(T_ref) - int(n_kl_step)
            while n_kl >= int(n_kl_min):
                cov_n_input = uplinksystem.get_interference_plus_noise_covariance(k, ell, F_override=snapshot_full)
                F_n = infer_precoder_numpy_with_blocklength(
                    user_models[k],
                    H_kl,
                    n_kl=n_kl,
                    noise_plus_interference_cov=np.asarray(cov_n_input, dtype=np.complex128),
                    epsilon=epsilon,
                    Nt=int(uplinksystem.NT[k]),
                    dk=int(uplinksystem.dk[k]),
                    P=P,
                    device=DEVICE,
                )
                snapshot_candidate = copy.deepcopy(snapshot_full)
                snapshot_candidate[k][ell] = F_n
                cov_n = uplinksystem.get_interference_plus_noise_covariance(k, ell, F_override=snapshot_candidate)
                R_n = _compute_r_fbl_np(H_kl, F_n, sigma2, epsilon, n_kl, cov_n)
                rate_violation = (B_used / float(n_kl)) - R_n

                print(f"Precoder-net test n_kl={n_kl}: R_fbl={R_n}, rate_violation={rate_violation}")

                if rate_violation > 0.0:
                    break

                best_n = int(n_kl)
                best_R = float(R_n)
                best_F = torch.tensor(F_n, dtype=torch.complex64)
                S_block.append(
                    {
                        "n_kl": int(n_kl),
                        "n": int(n_kl),
                        "B_l": int(B_used),
                        "Bits per sub-block length B/n_kl": float(B_used) / float(n_kl),
                        "F": torch.tensor(F_n, dtype=torch.complex64),
                        "R_fbl": float(R_n),
                        "F_power": float(np.linalg.norm(F_n, "fro") ** 2),
                        "lambda_rate": 0.0,
                        "lambda_power": 0.0,
                        "loss_curve": [],
                        "method": method_name,
                    }
                )
                n_kl -= int(n_kl_step)

            all_user_block_results[k].append(S_block)
            n_star[k].append(best_n)
            F_star[k].append(best_F)
            R_star[k].append(best_R)
            B_used_star[k].append(int(B_used))

            B_kl = min(B_rem, int(B_used))
            B_kl_star[k].append(int(B_kl))
            B_rem -= B_kl
            print(f">>> Precoder-net chose n_kl={best_n}, B_used={B_used}, B_kl={B_kl}, remaining B_rem={B_rem}")

            if B_rem > 0:
                ell += 1
                L_out[k] = ell + 1

    apply_training_solution(uplinksystem, n_star, F_star)

    return {
        "L_out": L_out,
        "n_star": n_star,
        "F_star": F_star,
        "R_star": R_star,
        "all_user_block_results_train": all_user_block_results,
        "B_used_star": B_used_star,
        "B_kl_star": B_kl_star,
        "norm_stats": [(0.0 + 0.0j, 1.0) for _ in range(K)],
        "method_name": method_name,
    }


train_blocklength_aware_precoder = train_blocklength_aware_precoder_net
train_blocklength_aware_policy = train_blocklength_aware_precoder_net
_build_precoder_snapshot = _build_precoder_net_snapshot
_build_policy_snapshot = _build_precoder_net_snapshot
evaluate_blocklength_precoder = evaluate_blocklength_precoder_net
evaluate_blocklength_policy = evaluate_blocklength_precoder_net
