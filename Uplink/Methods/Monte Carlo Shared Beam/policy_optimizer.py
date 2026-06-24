import copy
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


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
    build_user_precoder_net_with_interference_context,
    export_user_model_specs,
    export_user_model_states,
    infer_precoder_numpy_with_interference_context,
    infer_precoder_torch_with_interference_context,
)


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
        raise RuntimeError("Non-positive logdet sign while evaluating uplink shared-beam rate.")

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
        n_val = int(n_kl)
        if n_val in seen:
            continue
        seen.add(n_val)
        ordered_values.append(n_val)
    return ordered_values


def build_training_dataset(
    cfg_name: str,
    train_seeds: Sequence[int],
) -> list[list[dict[str, Any]]]:
    system_params, sim_cfg = get_config(cfg_name)
    K = int(system_params["K"])
    scenarios_by_user: list[list[dict[str, Any]]] = [[] for _ in range(K)]
    n_min = int(sim_cfg["n_kl_min"])
    n_step = int(sim_cfg["n_kl_step"])
    blocks_per_seed = max(1, int(sim_cfg.get("precoder_net_train_blocks_per_seed", 1)))
    coarse_step = max(int(n_step), int(sim_cfg.get("precoder_net_train_n_kl_coarse_step", 5)))

    for seed in train_seeds:
        print(f"\n================ RAW TRAINING DATA seed={seed} ================")
        uplinksystem = UplinkSystem(system_params, seed=int(seed))
        ensure_blocks_up_to(uplinksystem, int(blocks_per_seed) - 1)

        for k in range(K):
            T_ref = int(uplinksystem.T[k])
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
                scenarios_by_user[k].append(
                    {
                        "seed": int(seed),
                        "user": int(k),
                        "block": int(block),
                        "H": H_block,
                        "n_values": [int(v) for v in n_values],
                        "T_ref": int(T_ref),
                        "P": float(uplinksystem.P[k]),
                        "sigma2": float(uplinksystem.sigma2[k]),
                        "epsilon": float(uplinksystem.epsilon[k]),
                        "noise_plus_interference_cov": np.asarray(
                            noise_plus_interference_cov,
                            dtype=np.complex128,
                        ),
                    }
                )

    return scenarios_by_user


def summarize_training_dataset(scenarios_by_user: Sequence[Sequence[dict[str, Any]]]) -> dict[str, Any]:
    total_scenarios = int(sum(len(scenarios) for scenarios in scenarios_by_user))
    total_n_evaluations = 0
    scenarios_by_seed: dict[int, int] = {}
    scenarios_by_block: dict[int, int] = {}
    global_n_evaluations_by_n_kl: dict[int, int] = {}
    per_user: list[dict[str, Any]] = []

    for user_idx, scenarios in enumerate(scenarios_by_user):
        user_seed_counts: dict[int, int] = {}
        user_block_counts: dict[int, int] = {}
        user_n_counts: dict[int, int] = {}
        for scenario in scenarios:
            seed = int(scenario["seed"])
            block = int(scenario["block"])
            user_seed_counts[seed] = user_seed_counts.get(seed, 0) + 1
            user_block_counts[block] = user_block_counts.get(block, 0) + 1
            scenarios_by_seed[seed] = scenarios_by_seed.get(seed, 0) + 1
            scenarios_by_block[block] = scenarios_by_block.get(block, 0) + 1
            for n_kl in scenario["n_values"]:
                n_val = int(n_kl)
                user_n_counts[n_val] = user_n_counts.get(n_val, 0) + 1
                global_n_evaluations_by_n_kl[n_val] = global_n_evaluations_by_n_kl.get(n_val, 0) + 1
                total_n_evaluations += 1
        per_user.append(
            {
                "user": int(user_idx),
                "total_scenarios": int(len(scenarios)),
                "n_evaluations_by_n_kl": {
                    str(int(k)): int(v) for k, v in sorted(user_n_counts.items())
                },
                "scenarios_by_seed": {str(int(k)): int(v) for k, v in sorted(user_seed_counts.items())},
                "scenarios_by_block": {str(int(k)): int(v) for k, v in sorted(user_block_counts.items())},
            }
        )

    return {
        "total_scenarios": int(total_scenarios),
        "total_n_evaluations": int(total_n_evaluations),
        "num_users": int(len(scenarios_by_user)),
        "scenarios_by_seed": {str(int(k)): int(v) for k, v in sorted(scenarios_by_seed.items())},
        "scenarios_by_block": {str(int(k)): int(v) for k, v in sorted(scenarios_by_block.items())},
        "global_n_evaluations_by_n_kl": {
            str(int(k)): int(v) for k, v in sorted(global_n_evaluations_by_n_kl.items())
        },
        "per_user": per_user,
    }


def _summarize_selected_n_kl(n_star: Sequence[Sequence[int]]) -> dict[str, Any]:
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
        "global_selected_examples_by_n_kl": {
            str(int(k)): int(v) for k, v in sorted(global_counts.items())
        },
        "per_user": per_user,
    }


def evaluate_shared_beam_precoder_net(
    uplinksystem: UplinkSystem,
    user_models: Sequence[torch.nn.Module],
    sim_cfg: dict[str, Any],
    *,
    method_name: str = "monte_carlo_shared_beam_train_test",
) -> dict[str, Any]:
    K = int(uplinksystem.K)
    L_out = [1] * K
    n_star = [[] for _ in range(K)]
    F_star = [[] for _ in range(K)]
    R_star = [[] for _ in range(K)]
    B_used_star = [[] for _ in range(K)]
    B_kl_star = [[] for _ in range(K)]
    all_user_block_results = [[] for _ in range(K)]

    n_kl_min = int(sim_cfg["n_kl_min"])
    n_kl_step = int(sim_cfg["n_kl_step"])

    for k in range(K):
        B_rem = int(uplinksystem.B[k])
        ell = 0

        while B_rem > 0:
            if ell >= len(uplinksystem.H[k]):
                uplinksystem.add_block(k)

            H_kl = np.asarray(uplinksystem.H[k][ell], dtype=np.complex64)
            T_ref = int(uplinksystem.T[k])
            P_user = float(uplinksystem.P[k])
            sigma2 = float(uplinksystem.sigma2[k])
            epsilon = float(uplinksystem.epsilon[k])

            snapshot_full = copy.deepcopy(uplinksystem.F)
            cov_input = uplinksystem.get_interference_plus_noise_covariance(k, ell, F_override=snapshot_full)
            F_shared = infer_precoder_numpy_with_interference_context(
                user_models[k],
                H_kl,
                np.asarray(cov_input, dtype=np.complex128),
                epsilon,
                Nt=int(uplinksystem.NT[k]),
                dk=int(uplinksystem.dk[k]),
                P=P_user,
                device=DEVICE,
            )
            snapshot_candidate = copy.deepcopy(snapshot_full)
            snapshot_candidate[k][ell] = F_shared
            cov_shared = uplinksystem.get_interference_plus_noise_covariance(k, ell, F_override=snapshot_candidate)

            B_try = int(B_rem)
            B_used = None
            S_block = []
            R_T = _compute_r_fbl_np(H_kl, F_shared, sigma2, epsilon, T_ref, cov_shared)
            for _ in range(12):
                rate_violation = (B_try / float(max(T_ref, 1))) - R_T
                if rate_violation <= 0.0:
                    B_used = int(B_try)
                    break
                B_new = int(np.floor(float(T_ref) * float(R_T)))
                B_new = max(0, min(B_new, B_try))
                if B_new == B_try or B_new <= 0:
                    B_used = 0
                    break
                B_try = B_new

            if B_used is None or B_used <= 0:
                break

            best_n = int(T_ref)
            best_R = float(R_T)
            shared_F_tensor = torch.tensor(F_shared, dtype=torch.complex64)
            S_block.append(
                {
                    "n_kl": int(T_ref),
                    "n": int(T_ref),
                    "B_l": int(B_used),
                    "Bits per sub-block length B/n_kl": float(B_used) / float(max(int(T_ref), 1)),
                    "F": shared_F_tensor,
                    "R_fbl": float(R_T),
                    "F_power": float(np.linalg.norm(F_shared, "fro") ** 2),
                    "loss_curve": [],
                    "method": method_name,
                }
            )

            n_kl = int(T_ref) - int(n_kl_step)
            while n_kl >= int(n_kl_min):
                R_n = _compute_r_fbl_np(H_kl, F_shared, sigma2, epsilon, n_kl, cov_shared)
                rate_violation = (B_used / float(n_kl)) - R_n
                if rate_violation > 0.0:
                    break
                best_n = int(n_kl)
                best_R = float(R_n)
                S_block.append(
                    {
                        "n_kl": int(n_kl),
                        "n": int(n_kl),
                        "B_l": int(B_used),
                        "Bits per sub-block length B/n_kl": float(B_used) / float(max(int(n_kl), 1)),
                        "F": shared_F_tensor,
                        "R_fbl": float(R_n),
                        "F_power": float(np.linalg.norm(F_shared, "fro") ** 2),
                        "loss_curve": [],
                        "method": method_name,
                    }
                )
                n_kl -= int(n_kl_step)

            all_user_block_results[k].append(S_block)
            n_star[k].append(int(best_n))
            F_star[k].append(shared_F_tensor)
            R_star[k].append(float(best_R))
            B_used_star[k].append(int(B_used))

            B_kl = min(B_rem, int(B_used))
            B_kl_star[k].append(int(B_kl))
            B_rem -= B_kl

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
        "final_latency": [float(v) for v in uplinksystem.latency],
        "final_n": [int(v) for v in uplinksystem.n],
        "final_served_bits_per_user": [int(sum(v)) for v in B_kl_star],
        "method_name": method_name,
        "precoder_parameterization": "shared_user_channel_noise_epsilon_to_shared_beam_mlp",
    }


def _build_post_training_summary(
    training_history: dict[str, Any],
    train_eval_post: dict[str, Any],
    initial_baseline: dict[str, Any],
    *,
    train_eval_seed: int,
    epochs: int,
) -> dict[str, Any]:
    per_user_loss = training_history.get("per_user_loss", [])
    per_user_rate = training_history.get("per_user_rate", [])
    avg_loss = training_history.get("avg_loss", [])
    avg_user_rate = training_history.get("avg_user_rate", [])

    initial_latency = [float(v) for v in initial_baseline["initial_latency"]]
    final_latency = [float(v) for v in train_eval_post.get("final_latency", [])]
    initial_total_latency = float(sum(initial_latency))
    final_total_latency = float(sum(final_latency))
    latency_reduction_percent = (
        ((initial_total_latency - final_total_latency) / initial_total_latency) * 100.0
        if initial_total_latency > 0.0
        else 0.0
    )

    return {
        "train_eval_seed": int(train_eval_seed),
        "epochs_requested": int(epochs),
        "per_user_num_epochs": [int(len(history)) for history in per_user_loss],
        "per_user_final_loss": [float(history[-1]) if history else 0.0 for history in per_user_loss],
        "per_user_best_loss": [float(min(history)) if history else 0.0 for history in per_user_loss],
        "per_user_final_rate": [float(history[-1]) if history else 0.0 for history in per_user_rate],
        "per_user_best_rate": [float(max(history)) if history else 0.0 for history in per_user_rate],
        "final_avg_loss": float(avg_loss[-1]) if avg_loss else 0.0,
        "best_avg_loss": float(min(avg_loss)) if avg_loss else 0.0,
        "final_avg_user_rate": float(avg_user_rate[-1]) if avg_user_rate else 0.0,
        "best_avg_user_rate": float(max(avg_user_rate)) if avg_user_rate else 0.0,
        "train_eval_initial_latency": initial_latency,
        "train_eval_final_latency": final_latency,
        "train_eval_initial_blocks_per_user": [len(v) for v in initial_baseline["initial_n_kl"]],
        "train_eval_blocks_per_user": [len(v) for v in train_eval_post["n_star"]],
        "train_eval_initial_total_n_per_user": [int(v) for v in initial_baseline["initial_n"]],
        "train_eval_total_n_per_user": [int(sum(v)) for v in train_eval_post["n_star"]],
        "train_eval_initial_served_bits_per_user": [int(sum(v)) for v in initial_baseline["initial_B_kl"]],
        "train_eval_served_bits_per_user": [int(sum(v)) for v in train_eval_post["B_kl_star"]],
        "train_eval_total_latency_reduction_percent": float(latency_reduction_percent),
        "train_eval_initial_selected_n_kl_summary": _summarize_selected_n_kl(initial_baseline["initial_n_kl"]),
        "train_eval_selected_n_kl_summary": _summarize_selected_n_kl(train_eval_post["n_star"]),
    }


def train_shared_beam_precoder_net(
    cfg_name: str,
    train_seeds: Sequence[int],
    *,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    verbose: bool = True,
) -> dict[str, Any]:
    system_params, sim_cfg = get_config(cfg_name)
    K = int(system_params["K"])
    scenarios_by_user = build_training_dataset(cfg_name, train_seeds)
    dataset_summary = summarize_training_dataset(scenarios_by_user)
    training_history = {
        "per_user_loss": [[] for _ in range(K)],
        "per_user_rate": [[] for _ in range(K)],
        "avg_loss": [],
        "avg_user_rate": [],
        "dataset_summary": dataset_summary,
        "training_objective": "average_shared_beam_fbl_rate_over_candidate_n_grid",
    }

    user_models: list[torch.nn.Module] = []
    training_dataset_sizes = [int(len(scenarios)) for scenarios in scenarios_by_user]

    for k in range(K):
        model = build_user_precoder_net_with_interference_context(
            int(system_params["NR"][k]),
            int(system_params["NT"][k]),
            int(system_params["dk"][k]),
            device=DEVICE,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
        scenarios = scenarios_by_user[k]
        loss_history = training_history["per_user_loss"][k]
        rate_history = training_history["per_user_rate"][k]

        if len(scenarios) == 0:
            user_models.append(model.eval())
            continue

        if verbose:
            print(
                f"\n================ SHARED-BEAM TRAIN USER {k} ================\n"
                f"Scenarios: {len(scenarios)} | epochs: {int(epochs)} | batch_size: {int(batch_size)}"
            )

        rng = np.random.default_rng(int(train_seeds[0]) + 97 * (k + 1))
        indices = np.arange(len(scenarios))
        for epoch in range(int(epochs)):
            model.train()
            rng.shuffle(indices)
            epoch_scenario_rate_sum = 0.0
            epoch_scenario_count = 0
            epoch_n_rate_sum = 0.0
            epoch_n_count = 0

            for start in range(0, len(indices), max(int(batch_size), 1)):
                batch_idx = indices[start:start + max(int(batch_size), 1)]
                optimizer.zero_grad()
                loss = torch.zeros((), dtype=torch.float32, device=DEVICE)
                batch_scenarios = 0

                for idx in batch_idx:
                    scenario = scenarios[int(idx)]
                    H_t = torch.tensor(scenario["H"], dtype=torch.complex64, device=DEVICE)
                    noise_cov_t = torch.tensor(
                        scenario["noise_plus_interference_cov"],
                        dtype=torch.complex64,
                        device=DEVICE,
                    )
                    pred_t = infer_precoder_torch_with_interference_context(
                        model,
                        H_t,
                        noise_cov_t,
                        float(scenario["epsilon"]),
                        int(system_params["NT"][k]),
                        int(system_params["dk"][k]),
                        float(scenario["P"]),
                    )
                    rate_terms = []
                    for n_kl in scenario["n_values"]:
                        rate = _compute_r_fbl_torch(
                            H_t,
                            pred_t,
                            epsilon=float(scenario["epsilon"]),
                            n_kl=int(n_kl),
                            noise_plus_interference_cov=noise_cov_t,
                        )
                        rate_terms.append(rate)
                        epoch_n_rate_sum += float(rate.detach().cpu())
                        epoch_n_count += 1
                    scenario_avg_rate = torch.stack(rate_terms).mean()
                    loss = loss - scenario_avg_rate
                    epoch_scenario_rate_sum += float(scenario_avg_rate.detach().cpu())
                    epoch_scenario_count += 1
                    batch_scenarios += 1

                if batch_scenarios <= 0:
                    continue

                loss = loss / float(batch_scenarios)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            avg_loss = -float(epoch_scenario_rate_sum / max(epoch_scenario_count, 1))
            avg_rate = float(epoch_n_rate_sum / max(epoch_n_count, 1))
            loss_history.append(avg_loss)
            rate_history.append(avg_rate)
            if verbose:
                print(
                    f"Shared-beam user {k} epoch {epoch + 1}/{int(epochs)}: "
                    f"loss={avg_loss:.6e} | avg_rate={avg_rate:.6f}"
                )

        user_models.append(model.eval())

    if int(epochs) > 0:
        for epoch_idx in range(int(epochs)):
            epoch_losses = [
                training_history["per_user_loss"][k][epoch_idx]
                for k in range(K)
                if len(training_history["per_user_loss"][k]) > epoch_idx
            ]
            epoch_rates = [
                training_history["per_user_rate"][k][epoch_idx]
                for k in range(K)
                if len(training_history["per_user_rate"][k]) > epoch_idx
            ]
            training_history["avg_loss"].append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)
            training_history["avg_user_rate"].append(float(np.mean(epoch_rates)) if epoch_rates else 0.0)

    train_eval_seed = int(train_seeds[0]) if len(train_seeds) > 0 else 0
    initial_baseline = estimate_initial_random_precoder_schedule(system_params, sim_cfg, seed=train_eval_seed)
    train_eval_system = UplinkSystem(system_params, seed=train_eval_seed)
    train_eval_post = evaluate_shared_beam_precoder_net(
        train_eval_system,
        user_models,
        sim_cfg,
        method_name="monte_carlo_shared_beam_train_eval",
    )
    post_training_summary = _build_post_training_summary(
        training_history,
        train_eval_post,
        initial_baseline,
        train_eval_seed=train_eval_seed,
        epochs=int(epochs),
    )

    artifact = {
        **train_eval_post,
        "user_model_specs": export_user_model_specs(
            system_params["NR"],
            system_params["NT"],
            system_params["dk"],
            uses_blocklength_input=False,
            input_mode="channel_noise_epsilon",
        ),
        "user_model_states": export_user_model_states(user_models),
        "training_dataset_sizes": training_dataset_sizes,
        "training_sample_counts_per_user": training_dataset_sizes,
        "training_dataset_summary": dataset_summary,
        "post_training_summary": post_training_summary,
        "precoder_parameterization": "shared_user_channel_noise_epsilon_to_shared_beam_mlp",
        "training_objective": "average_shared_beam_fbl_rate_over_candidate_n_grid",
        "precoder_net_training_history": training_history,
        "precoder_net_training_losses": [list(map(float, row)) for row in training_history["per_user_loss"]],
    }
    return artifact


train_shared_beam_precoder = train_shared_beam_precoder_net
evaluate_shared_beam_precoder = evaluate_shared_beam_precoder_net
