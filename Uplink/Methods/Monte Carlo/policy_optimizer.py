import copy
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

METHOD_DIR = Path(__file__).resolve().parent
LINK_ROOT = METHOD_DIR.parents[1]
PROJECT_ROOT = LINK_ROOT.parent
BASELINE_DIR = METHOD_DIR.parent / "Convergence per sweep"
for path in (METHOD_DIR, LINK_ROOT, PROJECT_ROOT, BASELINE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from UplinkSystem import UplinkSystem
from advanced_methods_common import (
    apply_training_solution,
    clone_nested_arrays,
    collect_uplink_interference_diagnostics,
    ensure_blocks_up_to,
    estimate_initial_random_precoder_schedule,
    estimate_initial_random_precoder_schedule_for_scenario as shared_estimate_initial_random_precoder_schedule_for_scenario,
)
from config_loader import get_config
from experiment_scenarios import (
    FIXED_BLOCK_TARGETS_MODE,
    PAYLOAD_COMPLETION_MODE,
    build_experiment_scenario,
)
from precoder_models import (
    DEVICE,
    build_user_precoder_net_with_blocklength_and_sigma,
    export_user_model_specs,
    export_user_model_states,
    infer_precoder_numpy_with_blocklength_and_sigma,
    infer_precoder_torch_with_blocklength_and_sigma,
)
from terminal_logging import format_log_line
from uplink_rate_model import build_uplink_rate_covariance

CONSTRAINT_LOSS_FORMS = {"plain_lagrangian", "augmented_lagrangian"}


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
    sigma2: float,
    epsilon: float,
    n_kl: int,
    noise_plus_interference_cov: torch.Tensor | None,
) -> torch.Tensor:
    Nr = H.shape[0]
    I = torch.eye(Nr, dtype=torch.complex64, device=H.device)
    if noise_plus_interference_cov is None:
        noise_cov = float(sigma2) * I
    else:
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
    noise_plus_interference_cov: np.ndarray | None,
) -> float:
    from Optimizer_per_block import _compute_R_fbl_np

    return _compute_R_fbl_np(
        H=np.asarray(H, dtype=np.complex64),
        F=np.asarray(Fmat, dtype=np.complex64),
        sigma2=float(sigma2),
        epsilon=float(epsilon),
        n_kl=int(n_kl),
        noise_plus_interference_cov=(
            None
            if noise_plus_interference_cov is None
            else np.asarray(noise_plus_interference_cov, dtype=np.complex128)
        ),
    )


def _resolve_constraint_loss_form(sim_cfg: dict[str, Any]) -> str:
    mode = str(sim_cfg.get("constraint_loss_form", "plain_lagrangian")).strip().lower()
    if mode not in CONSTRAINT_LOSS_FORMS:
        known = ", ".join(sorted(CONSTRAINT_LOSS_FORMS))
        raise ValueError(f"Unknown constraint loss form '{mode}'. Expected one of: {known}")
    return mode


def _constraint_violation_activation(value: torch.Tensor, loss_form: str) -> torch.Tensor:
    if loss_form == "plain_lagrangian":
        return F.leaky_relu(value)
    return torch.relu(value)


def _zero_uplink_precoder(uplinksystem: UplinkSystem, user: int) -> np.ndarray:
    k = int(user)
    return np.zeros((int(uplinksystem.NT[k]), int(uplinksystem.dk[k])), dtype=np.complex64)


def _estimate_initial_random_precoder_schedule_fixed_block_targets(
    system_params: dict[str, Any],
    sim_cfg: dict[str, Any],
    *,
    seed: int,
    scenario: dict[str, Any],
) -> dict[str, Any]:
    baseline_system = UplinkSystem(system_params, seed=int(seed))
    K = int(baseline_system.K)
    n_kl_min = int(sim_cfg["n_kl_min"])
    n_kl_step = int(sim_cfg["n_kl_step"])
    block_targets = np.asarray(scenario["block_bit_targets"], dtype=int)
    num_blocks = int(scenario["num_blocks"])

    initial_n_kl: list[list[int]] = [[] for _ in range(K)]
    initial_B_kl: list[list[int]] = [[] for _ in range(K)]
    initial_R_fbl: list[list[float]] = [[] for _ in range(K)]
    initial_F: list[list[np.ndarray]] = [[] for _ in range(K)]
    skipped_blocks_per_user = [0 for _ in range(K)]

    for block in range(num_blocks):
        ensure_blocks_up_to(baseline_system, block)
        random_snapshot = clone_nested_arrays(baseline_system.F)

        for k in range(K):
            target_bits = int(block_targets[k, block])
            H_kl = np.asarray(baseline_system.H[k][block], dtype=np.complex64)
            F_kl = np.asarray(random_snapshot[k][block], dtype=np.complex64)
            T_ref = int(baseline_system.T[k])
            sigma2 = float(baseline_system.sigma2[k])
            epsilon = float(baseline_system.epsilon[k])
            noise_plus_interference_cov = build_uplink_rate_covariance(
                baseline_system,
                sim_cfg,
                k,
                block,
                F_override=random_snapshot,
            )
            R_T = _compute_r_fbl_np(
                H_kl,
                F_kl,
                sigma2,
                epsilon,
                T_ref,
                noise_plus_interference_cov,
            )
            B_max = max(int(np.floor(float(T_ref) * float(R_T))), 0)
            B_used = int(min(target_bits, B_max))
            best_n = int(T_ref)
            best_R = float(R_T)
            if int(B_used) >= int(target_bits) and int(target_bits) > 0:
                candidate_n = int(T_ref) - int(n_kl_step)
                while candidate_n >= int(n_kl_min):
                    R_candidate = _compute_r_fbl_np(
                        H_kl,
                        F_kl,
                        sigma2,
                        epsilon,
                        candidate_n,
                        noise_plus_interference_cov,
                    )
                    if (float(target_bits) / float(max(candidate_n, 1))) <= R_candidate:
                        best_n = int(candidate_n)
                        best_R = float(R_candidate)
                        candidate_n -= int(n_kl_step)
                    else:
                        break

            initial_n_kl[k].append(int(best_n))
            initial_B_kl[k].append(int(B_used))
            initial_R_fbl[k].append(float(best_R))
            initial_F[k].append(
                np.array(F_kl, copy=True) if int(B_used) > 0 else _zero_uplink_precoder(baseline_system, k)
            )
            if int(B_used) <= 0:
                skipped_blocks_per_user[k] += 1

    initial_n = [int(sum(int(max(v, 0)) for v in user_n)) for user_n in initial_n_kl]
    initial_latency = [
        float(initial_n[k]) / float(max(float(baseline_system.fs[k]), 1e-30))
        for k in range(K)
    ]
    initial_bits_per_symbol_by_block = []
    initial_bits_per_symbol = []
    for k in range(K):
        user_bps = [
            float(bits) / float(max(int(n_kl), 1))
            if int(n_kl) > 0 and int(bits) > 0
            else 0.0
            for bits, n_kl in zip(initial_B_kl[k], initial_n_kl[k])
        ]
        total_n = float(max(initial_n[k], 1))
        initial_bits_per_symbol_by_block.append(user_bps)
        initial_bits_per_symbol.append(float(sum(initial_B_kl[k])) / total_n if initial_n[k] > 0 else 0.0)

    apply_training_solution(baseline_system, initial_n_kl, initial_F)
    _, initial_snr_db = baseline_system.get_SNR()
    _, initial_sinr_db = baseline_system.get_SINR()
    initial_interference_diag = collect_uplink_interference_diagnostics(baseline_system)

    return {
        "initial_n_kl": initial_n_kl,
        "initial_B_kl": initial_B_kl,
        "initial_R_fbl": initial_R_fbl,
        "initial_n": initial_n,
        "initial_latency": [float(v) for v in baseline_system.latency],
        "initial_snr_db": list(map(float, initial_snr_db)),
        "initial_sinr_db": list(map(float, initial_sinr_db)),
        "initial_bits_per_symbol": initial_bits_per_symbol,
        "initial_bits_per_symbol_by_block": initial_bits_per_symbol_by_block,
        "initial_interference_diag": initial_interference_diag,
        "skipped_blocks_per_user": [int(v) for v in skipped_blocks_per_user],
        "scenario_mode": FIXED_BLOCK_TARGETS_MODE,
        "scenario_block_targets": block_targets.tolist(),
    }


def estimate_initial_random_precoder_schedule_for_scenario(
    system_params: dict[str, Any],
    sim_cfg: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    return shared_estimate_initial_random_precoder_schedule_for_scenario(
        system_params,
        sim_cfg,
        seed=int(seed),
    )


def build_training_dataset(
    cfg_name: str,
    train_seeds: Sequence[int],
) -> list[list[dict]]:
    system_params, sim_cfg = get_config(cfg_name)
    K = int(system_params["K"])
    episodes_by_user: list[list[dict]] = [[] for _ in range(K)]
    min_bits_required = max(1, int(sim_cfg.get("monte_carlo_training_fallback_target_bits", 1)))
    blocks_per_seed = max(1, int(sim_cfg.get("monte_carlo_training_blocks_per_seed", 1)))
    scenario_mode = str(sim_cfg.get("experiment_scenario_mode", PAYLOAD_COMPLETION_MODE))

    for seed in train_seeds:
        print(f"\n================ RAW TRAINING DATA seed={seed} ================")
        uplinksystem = UplinkSystem(system_params, seed=int(seed))
        scenario = build_experiment_scenario(system_params, sim_cfg, seed=int(seed))
        block_targets = (
            np.asarray(scenario.get("block_bit_targets", []), dtype=int)
            if str(scenario.get("mode", PAYLOAD_COMPLETION_MODE)) == FIXED_BLOCK_TARGETS_MODE
            else None
        )
        max_blocks_for_seed = int(blocks_per_seed)
        if block_targets is not None and block_targets.ndim == 2:
            max_blocks_for_seed = min(max_blocks_for_seed, int(block_targets.shape[1]))
        ensure_blocks_up_to(uplinksystem, int(max_blocks_for_seed) - 1)

        for k in range(K):
            T_ref = int(uplinksystem.T[k])
            P_user = float(uplinksystem.P[k])
            sigma2 = float(uplinksystem.sigma2[k])
            epsilon = float(uplinksystem.epsilon[k])
            dk_user = int(uplinksystem.dk[k])
            max_blocks = min(int(max_blocks_for_seed), len(uplinksystem.H[k]))
            for block in range(max_blocks):
                H_block = np.asarray(uplinksystem.H[k][int(block)], dtype=np.complex64)
                noise_plus_interference_cov = build_uplink_rate_covariance(
                    uplinksystem,
                    sim_cfg,
                    k,
                    int(block),
                )
                target_bits = (
                    int(block_targets[k, int(block)])
                    if block_targets is not None
                    else int(min_bits_required)
                )
                episodes_by_user[k].append(
                    {
                        "seed": int(seed),
                        "user": int(k),
                        "block": int(block),
                        "H": H_block,
                        "T_ref": int(T_ref),
                        "target_bits": int(target_bits),
                        "P": float(P_user),
                        "dk": int(dk_user),
                        "sigma2": float(sigma2),
                        "epsilon": float(epsilon),
                        "noise_plus_interference_cov": (
                            None
                            if noise_plus_interference_cov is None
                            else np.asarray(noise_plus_interference_cov, dtype=np.complex128)
                        ),
                        "scenario_mode": scenario_mode,
                    }
                )

    return episodes_by_user


def summarize_training_dataset(episodes_by_user: Sequence[Sequence[dict]]) -> dict:
    total_channel_episodes = int(sum(len(episodes) for episodes in episodes_by_user))
    global_episodes_by_target_bits: dict[int, int] = {}
    episodes_by_seed: dict[int, int] = {}
    global_episodes_by_block: dict[int, int] = {}
    scenario_modes: set[str] = set()
    per_user_summary = []

    for user_idx, episodes in enumerate(episodes_by_user):
        user_episodes_by_target_bits: dict[int, int] = {}
        user_episodes_by_seed: dict[int, int] = {}
        user_episodes_by_block: dict[int, int] = {}
        for episode in episodes:
            target_bits = int(episode.get("target_bits", episode.get("min_bits_required", 1)))
            seed = int(episode["seed"])
            block = int(episode.get("block", 0))
            scenario_modes.add(str(episode.get("scenario_mode", PAYLOAD_COMPLETION_MODE)))
            user_episodes_by_target_bits[target_bits] = (
                user_episodes_by_target_bits.get(target_bits, 0) + 1
            )
            user_episodes_by_seed[seed] = user_episodes_by_seed.get(seed, 0) + 1
            user_episodes_by_block[block] = user_episodes_by_block.get(block, 0) + 1
            global_episodes_by_target_bits[target_bits] = (
                global_episodes_by_target_bits.get(target_bits, 0) + 1
            )
            episodes_by_seed[seed] = episodes_by_seed.get(seed, 0) + 1
            global_episodes_by_block[block] = global_episodes_by_block.get(block, 0) + 1

        per_user_summary.append(
            {
                "user": int(user_idx),
                "total_channel_episodes": int(len(episodes)),
                "episodes_by_target_bits": {
                    str(int(k)): int(v) for k, v in sorted(user_episodes_by_target_bits.items())
                },
                "episodes_by_seed": {str(int(k)): int(v) for k, v in sorted(user_episodes_by_seed.items())},
                "episodes_by_block": {str(int(k)): int(v) for k, v in sorted(user_episodes_by_block.items())},
            }
        )

    return {
        "total_channel_episodes": int(total_channel_episodes),
        "num_users": int(len(episodes_by_user)),
        "scenario_modes": sorted(scenario_modes),
        "episodes_by_seed": {str(int(k)): int(v) for k, v in sorted(episodes_by_seed.items())},
        "global_episodes_by_block": {str(int(k)): int(v) for k, v in sorted(global_episodes_by_block.items())},
        "global_episodes_by_target_bits": {
            str(int(k)): int(v) for k, v in sorted(global_episodes_by_target_bits.items())
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


def _serialize_count_dict(counts: dict[int, int]) -> dict[str, int]:
    return {str(int(k)): int(v) for k, v in sorted(counts.items())}


def _evaluate_uplink_rollout_query_numpy(
    model: torch.nn.Module,
    episode: dict[str, Any],
    n_kl: int,
) -> dict[str, Any]:
    H = np.asarray(episode["H"], dtype=np.complex64)
    noise_cov = episode.get("noise_plus_interference_cov")
    if noise_cov is not None:
        noise_cov = np.asarray(noise_cov, dtype=np.complex128)
    F_pred = infer_precoder_numpy_with_blocklength_and_sigma(
        model,
        H,
        int(n_kl),
        float(episode["sigma2"]),
        float(episode["epsilon"]),
        Nt=int(H.shape[1]),
        dk=int(episode.get("dk", H.shape[1] if H.ndim > 1 else 1)),
        P=float(episode["P"]),
        device=DEVICE,
    )
    power = float(np.linalg.norm(F_pred, ord="fro") ** 2)
    rate = float(
        _compute_r_fbl_np(
            H,
            F_pred,
            float(episode["sigma2"]),
            float(episode["epsilon"]),
            int(n_kl),
            noise_cov,
        )
    )
    required_rate = float(episode["target_bits"]) / float(max(int(n_kl), 1))
    rate_margin = float(rate - required_rate)
    power_margin = float(episode["P"]) - float(power)
    return {
        "rate": rate,
        "power": power,
        "required_rate": required_rate,
        "rate_margin": rate_margin,
        "power_margin": power_margin,
        "feasible": bool(rate_margin >= -1e-9 and power_margin >= -1e-9),
    }


def _generate_rollout_queries_for_user(
    model: torch.nn.Module,
    episodes: Sequence[dict[str, Any]],
    sim_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    n_min = int(sim_cfg["n_kl_min"])
    fine_step = max(1, int(sim_cfg["n_kl_step"]))
    coarse_step = max(fine_step, int(sim_cfg.get("monte_carlo_training_n_kl_coarse_step", fine_step)))
    rollout_queries: list[dict[str, Any]] = []

    for episode in episodes:
        T_ref = int(episode["T_ref"])
        visited_n: set[int] = set()
        episode_queries: list[dict[str, Any]] = []

        def record_query(n_kl: int, stage: str) -> dict[str, Any]:
            n_val = int(n_kl)
            if n_val in visited_n:
                for existing in episode_queries:
                    if int(existing["n_kl"]) == n_val:
                        return existing
            metrics = _evaluate_uplink_rollout_query_numpy(model, episode, n_val)
            query = {
                **episode,
                "n_kl": int(n_val),
                "rollout_stage": str(stage),
                "rate": float(metrics["rate"]),
                "power": float(metrics["power"]),
                "required_rate": float(metrics["required_rate"]),
                "rate_margin": float(metrics["rate_margin"]),
                "power_margin": float(metrics["power_margin"]),
                "feasible": bool(metrics["feasible"]),
                "frontier_query": False,
                "query_weight": 1.0,
            }
            visited_n.add(int(n_val))
            episode_queries.append(query)
            return query

        record_query(T_ref, "coarse")
        last_feasible_n = T_ref if bool(episode_queries[-1]["feasible"]) else None
        first_infeasible_n = None

        if last_feasible_n is not None:
            candidate = int(T_ref) - int(coarse_step)
            while candidate >= int(n_min):
                query = record_query(candidate, "coarse")
                if bool(query["feasible"]):
                    last_feasible_n = int(candidate)
                    candidate -= int(coarse_step)
                    continue
                first_infeasible_n = int(candidate)
                break

            if (
                last_feasible_n is not None
                and first_infeasible_n is not None
                and int(fine_step) < int(coarse_step)
            ):
                candidate = int(last_feasible_n) - int(fine_step)
                while candidate > int(first_infeasible_n):
                    query = record_query(candidate, "fine")
                    if bool(query["feasible"]):
                        last_feasible_n = int(candidate)
                        candidate -= int(fine_step)
                        continue
                    first_infeasible_n = int(candidate)
                    break

        feasible_indices = [idx for idx, query in enumerate(episode_queries) if bool(query["feasible"])]
        infeasible_indices = [idx for idx, query in enumerate(episode_queries) if not bool(query["feasible"])]
        if feasible_indices:
            episode_queries[feasible_indices[-1]]["frontier_query"] = True
        if infeasible_indices:
            feasible_cutoff = feasible_indices[-1] if feasible_indices else -1
            frontier_infeasible_idx = next(
                (idx for idx in infeasible_indices if idx > feasible_cutoff),
                infeasible_indices[0],
            )
            episode_queries[frontier_infeasible_idx]["frontier_query"] = True

        for query in episode_queries:
            weight = 1.0
            if bool(query["frontier_query"]):
                weight = 2.0
            elif not bool(query["feasible"]):
                weight = 1.25
            query["query_weight"] = float(weight)
            rollout_queries.append(query)

    return rollout_queries


def _summarize_rollout_queries_by_user(queries_by_user: Sequence[Sequence[dict[str, Any]]]) -> dict[str, Any]:
    global_queries_by_n_kl: dict[int, int] = {}
    global_frontier_queries_by_n_kl: dict[int, int] = {}
    global_queries_by_feasibility = {"feasible": 0, "infeasible": 0, "frontier": 0}
    per_user = []

    for user_idx, queries in enumerate(queries_by_user):
        user_queries_by_n_kl: dict[int, int] = {}
        user_frontier_queries_by_n_kl: dict[int, int] = {}
        feasible_count = 0
        infeasible_count = 0
        frontier_count = 0
        for query in queries:
            n_val = int(query["n_kl"])
            user_queries_by_n_kl[n_val] = user_queries_by_n_kl.get(n_val, 0) + 1
            global_queries_by_n_kl[n_val] = global_queries_by_n_kl.get(n_val, 0) + 1
            if bool(query.get("frontier_query", False)):
                user_frontier_queries_by_n_kl[n_val] = user_frontier_queries_by_n_kl.get(n_val, 0) + 1
                global_frontier_queries_by_n_kl[n_val] = global_frontier_queries_by_n_kl.get(n_val, 0) + 1
                frontier_count += 1
                global_queries_by_feasibility["frontier"] += 1
            if bool(query.get("feasible", False)):
                feasible_count += 1
                global_queries_by_feasibility["feasible"] += 1
            else:
                infeasible_count += 1
                global_queries_by_feasibility["infeasible"] += 1

        per_user.append(
            {
                "user": int(user_idx),
                "total_rollout_queries": int(len(queries)),
                "rollout_queries_by_n_kl": _serialize_count_dict(user_queries_by_n_kl),
                "frontier_rollout_queries_by_n_kl": _serialize_count_dict(user_frontier_queries_by_n_kl),
                "feasible_rollout_queries": int(feasible_count),
                "infeasible_rollout_queries": int(infeasible_count),
                "frontier_rollout_queries": int(frontier_count),
            }
        )

    return {
        "total_rollout_queries": int(sum(len(queries) for queries in queries_by_user)),
        "global_rollout_queries_by_n_kl": _serialize_count_dict(global_queries_by_n_kl),
        "global_frontier_rollout_queries_by_n_kl": _serialize_count_dict(global_frontier_queries_by_n_kl),
        "global_rollout_queries_by_feasibility": {
            key: int(value) for key, value in global_queries_by_feasibility.items()
        },
        "per_user": per_user,
    }


def _build_post_training_summary(
    train_eval_system: UplinkSystem,
    train_eval_post: dict,
    training_history: dict,
    *,
    train_eval_seed: int,
    epochs: int,
    dataset_summary: dict[str, Any],
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
        "train_target_bits_summary": dataset_summary.get("global_episodes_by_target_bits", {}),
        "train_target_bits_per_user": [
            user_summary.get("episodes_by_target_bits", {})
            for user_summary in dataset_summary.get("per_user", [])
        ],
        "train_target_bits_mode": (
            "fixed_block_targets_actual_bits"
            if FIXED_BLOCK_TARGETS_MODE in dataset_summary.get("scenario_modes", [])
            else "minimum_required_bits_fallback"
        ),
        "cumulative_rollout_queries_by_n_kl": training_history.get("cumulative_rollout_queries_by_n_kl", {}),
        "cumulative_frontier_rollout_queries_by_n_kl": training_history.get(
            "cumulative_frontier_rollout_queries_by_n_kl",
            {},
        ),
        "final_epoch_rollout_query_summary": training_history.get("final_epoch_rollout_query_summary", {}),
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
        "train_eval_initial_skipped_blocks_per_user": [
            int(v) for v in (initial_baseline.get("skipped_blocks_per_user", []) if isinstance(initial_baseline, dict) else [])
        ],
        "train_eval_skipped_blocks_per_user": [
            int(v) for v in train_eval_post.get("skipped_blocks_per_user", [0 for _ in range(int(train_eval_system.K))])
        ],
        "train_eval_selected_n_kl_summary": selected_n_summary,
    }


def train_blocklength_aware_precoder_net(
    cfg_name: str,
    train_seeds: Sequence[int],
    *,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> dict:
    system_params, sim_cfg = get_config(cfg_name)
    K = int(system_params["K"])
    scenario_mode = str(sim_cfg.get("experiment_scenario_mode", PAYLOAD_COMPLETION_MODE))
    episodes_by_user = build_training_dataset(cfg_name, train_seeds)
    dataset_summary = summarize_training_dataset(episodes_by_user)
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
        "rollout_query_summaries_per_user": [[] for _ in range(K)],
        "training_objective": (
            "rollout_lagrangian_user_finite_blocklength_rate_with_fixed_target_bits_objective"
            if scenario_mode == FIXED_BLOCK_TARGETS_MODE
            else "rollout_lagrangian_user_finite_blocklength_rate_with_fixed_min_bits_objective"
        ),
    }

    user_models = []
    cumulative_rollout_query_global_counts: dict[int, int] = {}
    cumulative_rollout_query_per_user_counts: list[dict[int, int]] = [{} for _ in range(K)]
    cumulative_frontier_query_global_counts: dict[int, int] = {}
    cumulative_frontier_query_per_user_counts: list[dict[int, int]] = [{} for _ in range(K)]
    last_epoch_queries_by_user: list[list[dict[str, Any]]] = [[] for _ in range(K)]

    for k in range(K):
        Nr = int(system_params["NR"][k])
        Nt = int(system_params["NT"][k])
        dk = int(system_params["dk"][k])

        model = build_user_precoder_net_with_blocklength_and_sigma(Nr=Nr, Nt=Nt, dk=dk, device=DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
        episodes = episodes_by_user[k]
        user_lagrangian_history = training_history["per_user_lagrangian"][k]
        user_rate_history = training_history["per_user_rate"][k]
        user_rate_violation_history = training_history["avg_rate_violation"][k]
        user_power_violation_history = training_history["avg_power_violation"][k]
        constraint_loss_form = _resolve_constraint_loss_form(sim_cfg)
        augmented_lagrangian_rho_rate = float(sim_cfg.get("augmented_lagrangian_rho_rate", 0.0))
        augmented_lagrangian_rho_power = float(sim_cfg.get("augmented_lagrangian_rho_power", 0.0))
        lambda_rate = float(sim_cfg.get("initial_lambda_rate_constraint", 0.1))
        lambda_power = float(sim_cfg.get("initial_lambda_power_constraint", 0.01))
        lr_rate = float(sim_cfg.get("lr_rate_constraint", 1e-2))
        lr_power = float(sim_cfg.get("lr_power_constraint", 1e-3))

        if len(episodes) == 0:
            user_models.append(model.eval())
            continue

        rng = np.random.default_rng(int(train_seeds[0]) + 17 * (k + 1))

        print(
            f"\n================ PRECODER NET TRAIN USER {k} ================\n"
            f"Channel episodes: {len(episodes)} | epochs: {epochs} | batch_size: {batch_size}"
        )

        for epoch in range(int(epochs)):
            rollout_queries = _generate_rollout_queries_for_user(model, episodes, sim_cfg)
            last_epoch_queries_by_user[k] = [dict(query) for query in rollout_queries]
            rollout_summary = _summarize_rollout_queries_by_user([rollout_queries])
            epoch_rollout_summary = (
                dict(rollout_summary["per_user"][0])
                if rollout_summary.get("per_user")
                else {
                    "user": int(k),
                    "total_rollout_queries": 0,
                    "rollout_queries_by_n_kl": {},
                    "frontier_rollout_queries_by_n_kl": {},
                    "feasible_rollout_queries": 0,
                    "infeasible_rollout_queries": 0,
                    "frontier_rollout_queries": 0,
                }
            )
            epoch_rollout_summary["epoch"] = int(epoch + 1)
            training_history["rollout_query_summaries_per_user"][k].append(epoch_rollout_summary)

            for n_key, count in epoch_rollout_summary.get("rollout_queries_by_n_kl", {}).items():
                n_val = int(n_key)
                cumulative_rollout_query_global_counts[n_val] = (
                    cumulative_rollout_query_global_counts.get(n_val, 0) + int(count)
                )
                cumulative_rollout_query_per_user_counts[k][n_val] = (
                    cumulative_rollout_query_per_user_counts[k].get(n_val, 0) + int(count)
                )
            for n_key, count in epoch_rollout_summary.get("frontier_rollout_queries_by_n_kl", {}).items():
                n_val = int(n_key)
                cumulative_frontier_query_global_counts[n_val] = (
                    cumulative_frontier_query_global_counts.get(n_val, 0) + int(count)
                )
                cumulative_frontier_query_per_user_counts[k][n_val] = (
                    cumulative_frontier_query_per_user_counts[k].get(n_val, 0) + int(count)
                )

            epoch_term_sum = 0.0
            epoch_rate_sum = 0.0
            epoch_rate_violation_sum = 0.0
            epoch_power_violation_sum = 0.0
            epoch_query_weight_sum = 0.0

            if len(rollout_queries) == 0:
                user_lagrangian_history.append(0.0)
                user_rate_history.append(0.0)
                user_rate_violation_history.append(0.0)
                user_power_violation_history.append(0.0)
                print(
                    format_log_line(
                        "[UL Monte Carlo Train]",
                        user=int(k),
                        epoch=f"{epoch + 1}/{int(epochs)}",
                        rollout_queries=0,
                        avg_rate=0.0,
                        avg_rate_violation=0.0,
                        avg_power_violation=0.0,
                        avg_lagrangian=0.0,
                    )
                )
                continue

            indices = np.arange(len(rollout_queries))
            rng.shuffle(indices)
            for start in range(0, len(indices), max(int(batch_size), 1)):
                batch_idx = indices[start:start + max(int(batch_size), 1)]
                optimizer.zero_grad()
                loss = torch.zeros((), dtype=torch.float32, device=DEVICE)
                batch_rate_violation = 0.0
                batch_power_violation = 0.0
                batch_query_weight_sum = 0.0

                for idx in batch_idx:
                    query = rollout_queries[int(idx)]
                    query_weight = float(query.get("query_weight", 1.0))
                    H_t = torch.tensor(query["H"], dtype=torch.complex64, device=DEVICE)
                    noise_cov_t = (
                        None
                        if query.get("noise_plus_interference_cov") is None
                        else torch.tensor(
                            query["noise_plus_interference_cov"],
                            dtype=torch.complex64,
                            device=DEVICE,
                        )
                    )

                    pred_t = infer_precoder_torch_with_blocklength_and_sigma(
                        model,
                        H_t,
                        int(query["n_kl"]),
                        float(query["sigma2"]),
                        float(query["epsilon"]),
                        Nt=Nt,
                        dk=dk,
                        P=query["P"],
                    )
                    rate = _compute_r_fbl_torch(
                        H_t,
                        pred_t,
                        sigma2=float(query["sigma2"]),
                        epsilon=float(query["epsilon"]),
                        n_kl=int(query["n_kl"]),
                        noise_plus_interference_cov=noise_cov_t,
                    )
                    power = (torch.linalg.norm(pred_t, ord="fro") ** 2).real
                    required_rate = float(query.get("target_bits", query.get("min_bits_required", 1))) / float(
                        max(int(query["n_kl"]), 1)
                    )
                    rate_violation = torch.tensor(required_rate, dtype=torch.float32, device=DEVICE) - rate
                    power_violation = power - float(query["P"])
                    rate_violation_pos = _constraint_violation_activation(rate_violation, constraint_loss_form)
                    power_violation_pos = _constraint_violation_activation(power_violation, constraint_loss_form)
                    term = (
                        -rate
                        + float(lambda_rate) * rate_violation_pos
                        + float(lambda_power) * power_violation_pos
                    )
                    if constraint_loss_form == "augmented_lagrangian":
                        term = (
                            term
                            + 0.5 * augmented_lagrangian_rho_rate * rate_violation_pos.pow(2)
                            + 0.5 * augmented_lagrangian_rho_power * power_violation_pos.pow(2)
                        )
                    loss = loss + (float(query_weight) * term)
                    batch_rate_violation += float(query_weight) * float(rate_violation_pos.detach().cpu())
                    batch_power_violation += float(query_weight) * float(power_violation_pos.detach().cpu())
                    batch_query_weight_sum += float(query_weight)
                    epoch_term_sum += float(query_weight) * float(term.detach().cpu())
                    epoch_rate_sum += float(query_weight) * float(rate.detach().cpu())
                    epoch_rate_violation_sum += float(query_weight) * float(rate_violation_pos.detach().cpu())
                    epoch_power_violation_sum += float(query_weight) * float(power_violation_pos.detach().cpu())
                    epoch_query_weight_sum += float(query_weight)

                if batch_query_weight_sum <= 0.0:
                    continue
                loss = loss / float(batch_query_weight_sum)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                lambda_rate = max(0.0, lambda_rate + lr_rate * (batch_rate_violation / float(batch_query_weight_sum)))
                lambda_power = max(0.0, lambda_power + lr_power * (batch_power_violation / float(batch_query_weight_sum)))

            avg_lagrangian = float(epoch_term_sum / max(epoch_query_weight_sum, 1.0))
            avg_rate = float(epoch_rate_sum / max(epoch_query_weight_sum, 1.0))
            avg_rate_violation = float(epoch_rate_violation_sum / max(epoch_query_weight_sum, 1.0))
            avg_power_violation = float(epoch_power_violation_sum / max(epoch_query_weight_sum, 1.0))
            user_lagrangian_history.append(avg_lagrangian)
            user_rate_history.append(avg_rate)
            user_rate_violation_history.append(avg_rate_violation)
            user_power_violation_history.append(avg_power_violation)
            print(
                format_log_line(
                    "[UL Monte Carlo Train]",
                    user=int(k),
                    epoch=f"{epoch + 1}/{int(epochs)}",
                    rollout_queries=int(len(rollout_queries)),
                    avg_rate=avg_rate,
                    avg_rate_violation=avg_rate_violation,
                    avg_power_violation=avg_power_violation,
                    avg_lagrangian=avg_lagrangian,
                )
            )

        user_models.append(model.eval())

    training_history["avg_lagrangian"] = _aggregate_epoch_means(training_history["per_user_lagrangian"])
    training_history["avg_user_rate"] = _aggregate_epoch_means(training_history["per_user_rate"])
    training_history["avg_rate_violation_over_users"] = _aggregate_epoch_means(training_history["avg_rate_violation"])
    training_history["avg_power_violation_over_users"] = _aggregate_epoch_means(training_history["avg_power_violation"])
    training_history["cumulative_rollout_queries_by_n_kl"] = {
        "global_rollout_queries_by_n_kl_over_all_epochs": _serialize_count_dict(cumulative_rollout_query_global_counts),
        "per_user_rollout_queries_by_n_kl_over_all_epochs": [
            _serialize_count_dict(user_counts) for user_counts in cumulative_rollout_query_per_user_counts
        ],
    }
    training_history["cumulative_frontier_rollout_queries_by_n_kl"] = {
        "global_frontier_rollout_queries_by_n_kl_over_all_epochs": _serialize_count_dict(
            cumulative_frontier_query_global_counts
        ),
        "per_user_frontier_rollout_queries_by_n_kl_over_all_epochs": [
            _serialize_count_dict(user_counts) for user_counts in cumulative_frontier_query_per_user_counts
        ],
    }
    training_history["final_epoch_rollout_query_summary"] = _summarize_rollout_queries_by_user(last_epoch_queries_by_user)

    train_eval_seed = int(train_seeds[0]) if len(train_seeds) > 0 else 0
    train_eval_initial_baseline = estimate_initial_random_precoder_schedule_for_scenario(
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
        dataset_summary=dataset_summary,
        initial_baseline=train_eval_initial_baseline,
    )

    train_eval_post.update(
        {
            "train_seeds": [int(s) for s in train_seeds],
            "training_dataset_sizes": [len(v) for v in episodes_by_user],
            "training_channel_episode_counts_per_user": [len(v) for v in episodes_by_user],
            "training_sample_counts_per_user": [len(v) for v in episodes_by_user],
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
                input_mode="channel_sigma_epsilon_n",
            ),
            "user_model_states": export_user_model_states(user_models),
            "precoder_parameterization": "shared_user_channel_n_sigma_epsilon_to_precoder_mlp",
            "training_objective": training_history["training_objective"],
            "initial_skipped_blocks_per_user": [
                int(v) for v in train_eval_initial_baseline.get("skipped_blocks_per_user", [0 for _ in range(K)])
            ],
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
            user_blocks.append(
                infer_precoder_numpy_with_blocklength_and_sigma(
                    user_models[k],
                    np.asarray(uplinksystem.H[k][l], dtype=np.complex64),
                    n_kl=int(uplinksystem.T[k]),
                    sigma2=float(uplinksystem.sigma2[k]),
                    epsilon=float(uplinksystem.epsilon[k]),
                    Nt=int(uplinksystem.NT[k]),
                    dk=int(uplinksystem.dk[k]),
                    P=float(uplinksystem.P[k]),
                    device=DEVICE,
                )
            )
        snapshot.append(user_blocks)

    return snapshot


def _build_precoder_net_snapshot_for_active_mask(
    uplinksystem: UplinkSystem,
    user_models: Sequence[torch.nn.Module],
    block_idx: int,
    active_mask: Sequence[int | float],
) -> list[list[np.ndarray]]:
    snapshot = _build_precoder_net_snapshot(uplinksystem, user_models, block_idx)
    for k in range(int(uplinksystem.K)):
        if float(active_mask[int(k)]) > 0.5:
            continue
        snapshot[k][int(block_idx)] = _zero_uplink_precoder(uplinksystem, k)
    return snapshot


def _evaluate_blocklength_precoder_net_fixed_block_targets(
    uplinksystem: UplinkSystem,
    user_models: Sequence[torch.nn.Module],
    sim_cfg: dict,
    *,
    method_name: str,
) -> dict:
    scenario = build_experiment_scenario(uplinksystem.sc, sim_cfg, seed=int(uplinksystem.seed))
    block_targets = np.asarray(scenario["block_bit_targets"], dtype=int)
    num_blocks = int(scenario["num_blocks"])
    K = int(uplinksystem.K)

    n_star = [[] for _ in range(K)]
    F_star = [[] for _ in range(K)]
    R_star = [[] for _ in range(K)]
    all_user_block_results = [[] for _ in range(K)]
    B_used_star = [[] for _ in range(K)]
    B_kl_star = [[] for _ in range(K)]
    target_bits_star = [[] for _ in range(K)]
    unserved_bits_star = [[] for _ in range(K)]
    skipped_blocks_per_user = [0 for _ in range(K)]

    n_kl_min = int(sim_cfg["n_kl_min"])
    n_kl_step = int(sim_cfg["n_kl_step"])

    for block in range(num_blocks):
        ensure_blocks_up_to(uplinksystem, block)
        active_mask = [1 for _ in range(K)]
        snapshot_full = _build_precoder_net_snapshot_for_active_mask(
            uplinksystem,
            user_models,
            block,
            active_mask,
        )

        for k in range(K):
            target_bits = int(block_targets[k, block])
            H_kl = np.asarray(uplinksystem.H[k][block], dtype=np.complex64)
            T_ref = int(uplinksystem.T[k])
            P = float(uplinksystem.P[k])
            sigma2 = float(uplinksystem.sigma2[k])
            epsilon = float(uplinksystem.epsilon[k])
            zero_precoder = _zero_uplink_precoder(uplinksystem, k)
            S_block = []

            F_T = infer_precoder_numpy_with_blocklength_and_sigma(
                user_models[k],
                H_kl,
                n_kl=T_ref,
                sigma2=sigma2,
                epsilon=epsilon,
                Nt=int(uplinksystem.NT[k]),
                dk=int(uplinksystem.dk[k]),
                P=P,
                device=DEVICE,
            )
            snapshot_candidate = copy.deepcopy(snapshot_full)
            snapshot_candidate[k][block] = F_T
            cov_T = build_uplink_rate_covariance(
                uplinksystem,
                sim_cfg,
                k,
                block,
                F_override=snapshot_candidate,
            )
            R_T = _compute_r_fbl_np(H_kl, F_T, sigma2, epsilon, T_ref, cov_T)
            B_max = max(int(np.floor(float(T_ref) * float(R_T))), 0)
            B_used = int(min(target_bits, B_max))
            target_bits_star[k].append(int(target_bits))

            if int(B_used) < int(target_bits):
                S_block.append(
                    {
                        "n_kl": int(T_ref),
                        "n": int(T_ref),
                        "B_l": int(B_used),
                        "Bits per sub-block length B/n_kl": (
                            float(B_used) / float(max(int(T_ref), 1)) if int(B_used) > 0 else 0.0
                        ),
                        "required_R_fbl": float(target_bits) / float(max(int(T_ref), 1)),
                        "achieved_R_fbl": float(R_T),
                        "F": torch.tensor(F_T if int(B_used) > 0 else zero_precoder, dtype=torch.complex64),
                        "R_fbl": float(R_T),
                        "F_power": float(np.linalg.norm(F_T, "fro") ** 2) if int(B_used) > 0 else 0.0,
                        "lambda_rate": 0.0,
                        "lambda_power": 0.0,
                        "loss_curve": [],
                        "method": method_name,
                        "skipped": bool(int(B_used) <= 0),
                        "target_bits": int(target_bits),
                        "unserved_bits": int(max(int(target_bits) - int(B_used), 0)),
                    }
                )
                all_user_block_results[k].append(S_block)
                n_star[k].append(int(T_ref))
                F_star[k].append(np.array(F_T if int(B_used) > 0 else zero_precoder, copy=True))
                R_star[k].append(float(R_T))
                B_used_star[k].append(int(B_used))
                B_kl_star[k].append(int(B_used))
                unserved_bits_star[k].append(int(max(int(target_bits) - int(B_used), 0)))
                if int(B_used) <= 0:
                    skipped_blocks_per_user[k] += 1
                continue

            best_n = int(T_ref)
            best_R = float(R_T)
            best_F = np.array(F_T, copy=True)
            S_block.append(
                {
                    "n_kl": int(T_ref),
                    "n": int(T_ref),
                    "B_l": int(target_bits),
                    "Bits per sub-block length B/n_kl": float(target_bits) / float(max(int(T_ref), 1)),
                    "required_R_fbl": float(target_bits) / float(max(int(T_ref), 1)),
                    "achieved_R_fbl": float(R_T),
                    "F": torch.tensor(F_T, dtype=torch.complex64),
                    "R_fbl": float(R_T),
                    "F_power": float(np.linalg.norm(F_T, "fro") ** 2),
                    "lambda_rate": 0.0,
                    "lambda_power": 0.0,
                    "loss_curve": [],
                    "method": method_name,
                    "skipped": False,
                    "target_bits": int(target_bits),
                    "unserved_bits": 0,
                }
            )

            n_kl = int(T_ref) - int(n_kl_step)
            while n_kl >= int(n_kl_min):
                F_n = infer_precoder_numpy_with_blocklength_and_sigma(
                    user_models[k],
                    H_kl,
                    n_kl=n_kl,
                    sigma2=sigma2,
                    epsilon=epsilon,
                    Nt=int(uplinksystem.NT[k]),
                    dk=int(uplinksystem.dk[k]),
                    P=P,
                    device=DEVICE,
                )
                snapshot_candidate = copy.deepcopy(snapshot_full)
                snapshot_candidate[k][block] = F_n
                cov_n = build_uplink_rate_covariance(
                    uplinksystem,
                    sim_cfg,
                    k,
                    block,
                    F_override=snapshot_candidate,
                )
                R_n = _compute_r_fbl_np(H_kl, F_n, sigma2, epsilon, n_kl, cov_n)
                rate_violation = (target_bits / float(max(int(n_kl), 1))) - R_n
                if rate_violation > 0.0:
                    break

                best_n = int(n_kl)
                best_R = float(R_n)
                best_F = np.array(F_n, copy=True)
                S_block.append(
                    {
                        "n_kl": int(n_kl),
                        "n": int(n_kl),
                        "B_l": int(target_bits),
                        "Bits per sub-block length B/n_kl": float(target_bits) / float(max(int(n_kl), 1)),
                        "required_R_fbl": float(target_bits) / float(max(int(n_kl), 1)),
                        "achieved_R_fbl": float(R_n),
                        "F": torch.tensor(F_n, dtype=torch.complex64),
                        "R_fbl": float(R_n),
                        "F_power": float(np.linalg.norm(F_n, "fro") ** 2),
                        "lambda_rate": 0.0,
                        "lambda_power": 0.0,
                        "loss_curve": [],
                        "method": method_name,
                        "skipped": False,
                        "target_bits": int(target_bits),
                        "unserved_bits": 0,
                    }
                )
                n_kl -= int(n_kl_step)

            all_user_block_results[k].append(S_block)
            n_star[k].append(int(best_n))
            F_star[k].append(np.array(best_F, copy=True))
            R_star[k].append(float(best_R))
            B_used_star[k].append(int(target_bits))
            B_kl_star[k].append(int(target_bits))
            unserved_bits_star[k].append(0)

    apply_training_solution(uplinksystem, n_star, F_star)

    return {
        "L_out": [int(len(v)) for v in n_star],
        "n_star": n_star,
        "F_star": F_star,
        "R_star": R_star,
        "all_user_block_results_train": all_user_block_results,
        "B_used_star": B_used_star,
        "B_kl_star": B_kl_star,
        "target_bits_star": target_bits_star,
        "unserved_bits_star": unserved_bits_star,
        "norm_stats": [(0.0 + 0.0j, 1.0) for _ in range(K)],
        "method_name": method_name,
        "skipped_blocks_per_user": [int(v) for v in skipped_blocks_per_user],
        "scenario_mode": FIXED_BLOCK_TARGETS_MODE,
        "scenario_block_targets": block_targets.tolist(),
    }


def evaluate_blocklength_precoder_net(
    uplinksystem: UplinkSystem,
    user_models: Sequence[torch.nn.Module],
    sim_cfg: dict,
    *,
    method_name: str,
) -> dict:
    scenario = build_experiment_scenario(uplinksystem.sc, sim_cfg, seed=int(uplinksystem.seed))
    if str(scenario["mode"]) == FIXED_BLOCK_TARGETS_MODE:
        return _evaluate_blocklength_precoder_net_fixed_block_targets(
            uplinksystem,
            user_models,
            sim_cfg,
            method_name=method_name,
        )
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

            print(
                format_log_line(
                    "[UL Monte Carlo Eval]",
                    user=int(k),
                    block=int(ell),
                    remaining_bits=int(B_rem),
                )
            )

            S_block = []

            F_T = infer_precoder_numpy_with_blocklength_and_sigma(
                user_models[k],
                H_kl,
                n_kl=T_ref,
                sigma2=sigma2,
                epsilon=epsilon,
                Nt=int(uplinksystem.NT[k]),
                dk=int(uplinksystem.dk[k]),
                P=P,
                device=DEVICE,
            )
            snapshot_candidate = copy.deepcopy(snapshot_full)
            snapshot_candidate[k][ell] = F_T
            cov_T = build_uplink_rate_covariance(
                uplinksystem,
                sim_cfg,
                k,
                ell,
                F_override=snapshot_candidate,
            )
            R_T = _compute_r_fbl_np(H_kl, F_T, sigma2, epsilon, T_ref, cov_T)
            B_max = max(int(np.floor(float(T_ref) * float(R_T))), 0)
            B_used = int(min(B_rem, B_max))

            print(
                format_log_line(
                    "[UL Monte Carlo Eval]",
                    user=int(k),
                    block=int(ell),
                    n_kl=int(T_ref),
                    requested_bits=int(B_rem),
                    feasible_bits=int(B_max),
                    served_bits=int(B_used),
                    achieved_rate=float(R_T),
                )
            )

            if B_used <= 0:
                print(
                    format_log_line(
                        "[UL Monte Carlo Eval]",
                        user=int(k),
                        block=int(ell),
                        status="stop_no_feasible_T_point",
                    )
                )
                break

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

            best_n = int(T_ref)
            best_R = float(S_block[-1]["R_fbl"])
            best_F = S_block[-1]["F"]

            n_kl = int(T_ref) - int(n_kl_step)
            while n_kl >= int(n_kl_min):
                if int(B_used) < int(B_rem):
                    print(
                        format_log_line(
                            "[UL Monte Carlo Eval]",
                            user=int(k),
                            block=int(ell),
                            n_kl=int(n_kl),
                            status="stop_partial_payload",
                        )
                    )
                    break
                F_n = infer_precoder_numpy_with_blocklength_and_sigma(
                    user_models[k],
                    H_kl,
                    n_kl=n_kl,
                    sigma2=sigma2,
                    epsilon=epsilon,
                    Nt=int(uplinksystem.NT[k]),
                    dk=int(uplinksystem.dk[k]),
                    P=P,
                    device=DEVICE,
                )
                snapshot_candidate = copy.deepcopy(snapshot_full)
                snapshot_candidate[k][ell] = F_n
                cov_n = build_uplink_rate_covariance(
                    uplinksystem,
                    sim_cfg,
                    k,
                    ell,
                    F_override=snapshot_candidate,
                )
                R_n = _compute_r_fbl_np(H_kl, F_n, sigma2, epsilon, n_kl, cov_n)
                rate_violation = (B_used / float(n_kl)) - R_n

                print(
                    format_log_line(
                        "[UL Monte Carlo Eval]",
                        user=int(k),
                        block=int(ell),
                        n_kl=int(n_kl),
                        achieved_rate=float(R_n),
                        rate_violation=float(rate_violation),
                    )
                )

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
            print(
                format_log_line(
                    "[UL Monte Carlo Allocation]",
                    user=int(k),
                    block=int(ell),
                    chosen_n_kl=int(best_n),
                    served_bits=int(B_used),
                    committed_bits=int(B_kl),
                    remaining_bits=int(B_rem),
                )
            )

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
        "skipped_blocks_per_user": [0 for _ in range(K)],
        "scenario_mode": PAYLOAD_COMPLETION_MODE,
    }


train_blocklength_aware_precoder = train_blocklength_aware_precoder_net
train_blocklength_aware_policy = train_blocklength_aware_precoder_net
_build_precoder_snapshot = _build_precoder_net_snapshot
_build_policy_snapshot = _build_precoder_net_snapshot
evaluate_blocklength_precoder = evaluate_blocklength_precoder_net
evaluate_blocklength_policy = evaluate_blocklength_precoder_net
