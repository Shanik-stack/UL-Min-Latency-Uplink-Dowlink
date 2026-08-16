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
from config_loader import (
    ASYNCHRONALITY_WEIGHTED_SUM_RATE_OBJECTIVE,
    END_TO_END_LATENCY_BEAM_REWARD_MODE,
    INVERSE_CNR_WEIGHTED_SUM_RATE_OBJECTIVE,
    RATE_BEAM_REWARD_MODE,
    UNWEIGHTED_SUM_RATE_OBJECTIVE,
    get_config,
    resolve_uplink_beam_reward_mode,
    resolve_uplink_objective_mode,
)
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
from blocklength_search import build_fixed_step_n_candidates, build_n_search_config, run_n_frontier_search
from terminal_logging import format_log_line, format_progress_log_line
from uplink_rate_model import build_uplink_rate_covariance, uses_uplink_interference
from Optimizer_per_block import _uplink_supported_bits_proxy

CONSTRAINT_LOSS_FORMS = {"plain_lagrangian", "augmented_lagrangian"}
MONTE_CARLO_ROLLOUT_WEIGHTING_MODES = {"phase_balanced", "uniform_per_query"}
ROLLOUT_QUERY_LAGRANGIAN_TRAINING_STYLE = "rollout_query_lagrangian"
EXACT_ROLLOUT_LATENCY_ALIGNED_TRAINING_STYLE = "exact_rollout_latency_aligned"
UPLINK_MONTE_CARLO_TRAINING_STYLE_ALIASES = {
    "legacy": ROLLOUT_QUERY_LAGRANGIAN_TRAINING_STYLE,
    "existing": ROLLOUT_QUERY_LAGRANGIAN_TRAINING_STYLE,
    "rollout_queries": ROLLOUT_QUERY_LAGRANGIAN_TRAINING_STYLE,
    "rollout_query_lagrangian": ROLLOUT_QUERY_LAGRANGIAN_TRAINING_STYLE,
    "lagrangian_rollout_queries": ROLLOUT_QUERY_LAGRANGIAN_TRAINING_STYLE,
    "latency": EXACT_ROLLOUT_LATENCY_ALIGNED_TRAINING_STYLE,
    "latency_aligned": EXACT_ROLLOUT_LATENCY_ALIGNED_TRAINING_STYLE,
    "episode_latency": EXACT_ROLLOUT_LATENCY_ALIGNED_TRAINING_STYLE,
    "chosen_rollout_latency": EXACT_ROLLOUT_LATENCY_ALIGNED_TRAINING_STYLE,
    "exact_rollout_latency_aligned": EXACT_ROLLOUT_LATENCY_ALIGNED_TRAINING_STYLE,
}
_Q_INV_CACHE: dict[tuple[str, int | None, float], torch.Tensor] = {}


def _build_monte_carlo_training_search_cfg(
    sim_cfg: dict[str, Any],
    *,
    n_min: int,
    n_max: int,
) -> dict[str, int | str]:
    return build_n_search_config(
        n_min=int(n_min),
        n_max=int(n_max),
        fine_step=int(sim_cfg["n_kl_step"]),
        direction=sim_cfg.get("n_search_direction", "descending"),
        strategy=sim_cfg.get("n_search_strategy", "fixed_step"),
        coarse_step=sim_cfg.get("n_search_coarse_step", int(sim_cfg["n_kl_step"])),
        exponential_factor=sim_cfg.get("n_search_exponential_factor", 2),
        allow_only_fixed_step=True,
    )


def _build_monte_carlo_test_search_cfg(
    sim_cfg: dict[str, Any],
    *,
    n_min: int,
    n_max: int,
) -> dict[str, int | str]:
    return build_n_search_config(
        n_min=int(n_min),
        n_max=int(n_max),
        fine_step=int(sim_cfg["n_kl_step"]),
        direction=sim_cfg.get("monte_carlo_test_n_search_direction", sim_cfg.get("n_search_direction", "descending")),
        strategy=sim_cfg.get("monte_carlo_test_n_search_strategy", sim_cfg.get("n_search_strategy", "fixed_step")),
        coarse_step=sim_cfg.get(
            "monte_carlo_test_n_search_coarse_step",
            sim_cfg.get("n_search_coarse_step", int(sim_cfg["n_kl_step"])),
        ),
        exponential_factor=sim_cfg.get(
            "monte_carlo_test_n_search_exponential_factor",
            sim_cfg.get("n_search_exponential_factor", 2),
        ),
        allow_only_fixed_step=False,
    )


def _to_complex_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.astype(np.complex64, copy=False)
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(np.complex64, copy=False)
    return np.asarray(x, dtype=np.complex64)


def _q_inv_torch(epsilon: float, device: torch.device = DEVICE) -> torch.Tensor:
    cache_key = (str(device.type), device.index, round(float(epsilon), 12))
    cached = _Q_INV_CACHE.get(cache_key)
    if cached is not None:
        return cached
    normal = torch.distributions.Normal(
        torch.tensor(0.0, device=device, dtype=torch.float64),
        torch.tensor(1.0, device=device, dtype=torch.float64),
    )
    p = torch.tensor(1.0 - float(epsilon), device=device, dtype=torch.float64)
    p = torch.clamp(p, 1e-12, 1.0 - 1e-12)
    value = normal.icdf(p).to(dtype=torch.float32)
    _Q_INV_CACHE[cache_key] = value
    return value


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


def _resolve_uplink_monte_carlo_training_style(sim_cfg: dict[str, Any]) -> str:
    raw_mode = str(
        sim_cfg.get("monte_carlo_training_style", ROLLOUT_QUERY_LAGRANGIAN_TRAINING_STYLE)
    ).strip().lower()
    resolved = UPLINK_MONTE_CARLO_TRAINING_STYLE_ALIASES.get(raw_mode)
    if resolved is None:
        known = ", ".join(
            sorted(
                {
                    ROLLOUT_QUERY_LAGRANGIAN_TRAINING_STYLE,
                    EXACT_ROLLOUT_LATENCY_ALIGNED_TRAINING_STYLE,
                }
            )
        )
        raise ValueError(
            f"Unknown uplink Monte Carlo training style '{raw_mode}'. Expected one of: {known}"
        )
    return str(resolved)


def _uplink_training_beam_reward_torch(
    rate: torch.Tensor,
    *,
    n_kl: int,
    requested_bits: int,
    beam_reward_mode: str,
    committed_symbols_before_block: float = 0.0,
    fs: float = 1.0,
) -> torch.Tensor:
    resolved_mode = resolve_uplink_beam_reward_mode(beam_reward_mode)
    if resolved_mode == RATE_BEAM_REWARD_MODE:
        return rate
    supported_bits = rate.clamp_min(0.0) * float(max(int(n_kl), 1))
    requested_bits_t = rate.new_tensor(max(int(requested_bits), 0))
    if resolved_mode != END_TO_END_LATENCY_BEAM_REWARD_MODE:
        return torch.minimum(requested_bits_t, supported_bits)
    served_bits = torch.minimum(requested_bits_t, supported_bits)
    remaining_after = torch.clamp(requested_bits_t - served_bits, min=0.0)
    safe_rate = rate.clamp_min(1e-9)
    safe_fs = max(float(fs), 1e-30)
    projected_latency_seconds = (
        rate.new_tensor(max(float(committed_symbols_before_block), 0.0) / safe_fs)
        + rate.new_tensor(float(max(int(n_kl), 1)) / safe_fs)
        + (remaining_after / safe_rate) / safe_fs
    )
    return -1000.0 * projected_latency_seconds


def _constraint_violation_activation(value: torch.Tensor, loss_form: str) -> torch.Tensor:
    return torch.relu(value)


def _uplink_soft_served_bits_torch(
    rate: torch.Tensor,
    *,
    n_kl: int,
    remaining_bits_before_block: int,
) -> torch.Tensor:
    supported_bits = rate.clamp_min(0.0) * float(max(int(n_kl), 1))
    remaining_bits_t = rate.new_tensor(max(int(remaining_bits_before_block), 0))
    return torch.minimum(remaining_bits_t, supported_bits)


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
    allow_n_reduction: bool = True,
) -> dict[str, Any]:
    return shared_estimate_initial_random_precoder_schedule_for_scenario(
        system_params,
        sim_cfg,
        seed=int(seed),
        allow_n_reduction=allow_n_reduction,
    )


def build_training_dataset(
    cfg_name: str,
    train_seeds: Sequence[int],
) -> list[dict[str, Any]]:
    system_params, sim_cfg = get_config(cfg_name)
    episodes: list[dict[str, Any]] = []

    for seed in train_seeds:
        print(
            format_log_line(
                "[UL Monte Carlo Dataset]",
                phase="collect",
                seed=int(seed),
                base_dataset="channel_episode_only",
            )
        )
        scenario = build_experiment_scenario(system_params, sim_cfg, seed=int(seed))
        episodes.append(
            {
                "seed": int(seed),
                "num_users": int(system_params["K"]),
                "scenario_mode": str(scenario.get("mode", PAYLOAD_COMPLETION_MODE)),
                "scenario": scenario,
            }
        )

    return episodes


def summarize_training_dataset(training_episodes: Sequence[dict[str, Any]]) -> dict:
    if len(training_episodes) == 0:
        return {
            "total_channel_episodes": 0,
            "num_users": 0,
            "base_dataset_kind": "channel_episodes_only",
            "scenario_modes": [],
            "channel_episodes_by_seed": {},
            "channel_episodes_per_user": [],
        }
    num_users = int(training_episodes[0].get("num_users", 0))
    return {
        "total_channel_episodes": int(len(training_episodes)),
        "num_users": int(num_users),
        "base_dataset_kind": "channel_episodes_only",
        "scenario_modes": sorted({str(episode.get("scenario_mode", PAYLOAD_COMPLETION_MODE)) for episode in training_episodes}),
        "channel_episodes_by_seed": {str(int(episode["seed"])): 1 for episode in training_episodes},
        "channel_episodes_per_user": [int(len(training_episodes)) for _ in range(num_users)],
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


def _clone_model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _relative_model_state_change(
    model: torch.nn.Module,
    previous_state: dict[str, torch.Tensor] | None,
) -> float:
    if previous_state is None:
        return float("inf")
    current_state = model.state_dict()
    delta_norm_sq = 0.0
    reference_norm_sq = 0.0
    for key, current_value in current_state.items():
        current_cpu = current_value.detach().cpu()
        previous_cpu = previous_state[key]
        delta_norm_sq += float(torch.sum((current_cpu - previous_cpu).pow(2)).item())
        reference_norm_sq += float(torch.sum(previous_cpu.pow(2)).item())
    delta_norm = float(np.sqrt(max(delta_norm_sq, 0.0)))
    reference_norm = float(np.sqrt(max(reference_norm_sq, 0.0)))
    return float(delta_norm / max(reference_norm, 1e-12))


def _resolve_rollout_anchor_bits(rate: float, n_kl: int) -> int:
    achievable_bits = int(np.floor(max(float(rate), 0.0) * float(max(int(n_kl), 1))))
    return max(1, achievable_bits)


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
    power_margin = float(episode["P"]) - float(power)
    return {
        "rate": rate,
        "power": power,
        "power_margin": power_margin,
    }


def _resolve_uplink_rollout_phase_weights(sim_cfg: dict[str, Any]) -> dict[str, float]:
    return {
        "full_block": float(sim_cfg.get("monte_carlo_training_full_block_weight", 1.0)),
        "tail_feasible": float(sim_cfg.get("monte_carlo_training_tail_feasible_weight", 1.0)),
        "tail_frontier": float(sim_cfg.get("monte_carlo_training_tail_frontier_weight", 1.5)),
    }


def _uplink_block_objective_weights(
    system: UplinkSystem,
    sim_cfg: dict[str, Any],
    *,
    block: int,
    active_mask: Sequence[int | float],
    remaining_bits_by_user: Sequence[int] | None = None,
    block_index_by_user: Sequence[int] | None = None,
) -> list[float]:
    objective_mode = resolve_uplink_objective_mode(
        sim_cfg.get("uplink_objective_mode", UNWEIGHTED_SUM_RATE_OBJECTIVE)
    )
    weights = [1.0 for _ in range(int(system.K))]
    if objective_mode == ASYNCHRONALITY_WEIGHTED_SUM_RATE_OBJECTIVE:
        if remaining_bits_by_user is None:
            return weights
        remaining_bits = np.asarray(remaining_bits_by_user, dtype=int)
        if block_index_by_user is None:
            next_block_indices = np.full(int(system.K), int(block), dtype=int)
        else:
            next_block_indices = np.asarray(block_index_by_user, dtype=int)
        active_users = [int(k) for k, flag in enumerate(active_mask) if float(flag) > 0.5 and int(remaining_bits[int(k)]) > 0]
        if len(active_users) <= 0:
            return weights
        ensure_blocks_up_to(system, int(np.max(next_block_indices[active_users])))
        min_weight = float(sim_cfg.get("minimum_user_weight", 0.25))
        exponent = float(sim_cfg.get("remaining_bits_weight_power", 1.0))
        projected_latencies: dict[int, float] = {}
        for k in active_users:
            next_block = int(next_block_indices[int(k)])
            cutoff = max(min(int(next_block), len(system.n_kl[int(k)])), 0)
            committed_symbols = float(sum(int(v) for v in system.n_kl[int(k)][:cutoff]))
            committed_latency = committed_symbols / max(float(system.fs[int(k)]), 1e-30)
            supported_bits = _uplink_supported_bits_proxy(system, sim_cfg, int(k), next_block)
            block_duration = float(system.T[int(k)]) / max(float(system.fs[int(k)]), 1e-30)
            projected_latencies[int(k)] = committed_latency + (
                float(max(int(remaining_bits[int(k)]), 0)) / max(float(supported_bits), 1.0)
            ) * block_duration
        min_latency = min(projected_latencies.values()) if projected_latencies else 1.0
        denom_latency = max(float(min_latency), 1e-30)
        raw_scores = {
            int(k): float(projected_latencies[int(k)] / denom_latency)
            for k in active_users
        }
        max_score = max(raw_scores.values()) if raw_scores else 1.0
        denom_score = max(float(max_score), 1e-30)
        for k in active_users:
            normalized = (float(raw_scores[int(k)]) / denom_score) ** exponent
            weights[int(k)] = float(min_weight + (1.0 - min_weight) * normalized)
        return weights

    if objective_mode != INVERSE_CNR_WEIGHTED_SUM_RATE_OBJECTIVE:
        return weights

    active_users = [int(k) for k, flag in enumerate(active_mask) if float(flag) > 0.5]
    if len(active_users) <= 0:
        return weights

    inverse_cnr_values: dict[int, float] = {}
    for k in active_users:
        H_kl = np.asarray(system.H[int(k)][int(block)], dtype=np.complex128)
        cnr_linear = float(np.linalg.norm(H_kl, ord="fro") ** 2) / max(float(system.sigma2[int(k)]), 1e-30)
        inverse_cnr_values[int(k)] = 1.0 / max(cnr_linear, 1e-30)

    mean_inverse_cnr = float(np.mean(list(inverse_cnr_values.values())))
    for k in active_users:
        weights[int(k)] = float(inverse_cnr_values[int(k)] / max(mean_inverse_cnr, 1e-30))
    return weights


def _resolve_uplink_rollout_query_weighting_mode(sim_cfg: dict[str, Any]) -> str:
    aliases = {
        "phase_balanced": "phase_balanced",
        "phase": "phase_balanced",
        "enabled": "phase_balanced",
        "on": "phase_balanced",
        "uniform_per_query": "uniform_per_query",
        "uniform": "uniform_per_query",
        "disabled": "uniform_per_query",
        "off": "uniform_per_query",
        "none": "uniform_per_query",
    }
    raw_mode = str(sim_cfg.get("monte_carlo_rollout_query_weighting_mode", "phase_balanced")).strip().lower()
    resolved = aliases.get(raw_mode)
    if resolved not in MONTE_CARLO_ROLLOUT_WEIGHTING_MODES:
        known = ", ".join(sorted(MONTE_CARLO_ROLLOUT_WEIGHTING_MODES))
        raise ValueError(
            f"Unknown Monte Carlo rollout query weighting mode '{raw_mode}'. Expected one of: {known}"
        )
    return resolved


def _normalize_uplink_episode_query_weights(
    episode_queries: Sequence[dict[str, Any]],
    phase_weights: dict[str, float],
    *,
    weighting_mode: str,
) -> list[dict[str, Any]]:
    if len(episode_queries) == 0:
        return []
    if str(weighting_mode) == "uniform_per_query":
        normalized: list[dict[str, Any]] = []
        for query in episode_queries:
            updated = dict(query)
            updated["query_weight"] = 1.0
            normalized.append(updated)
        return normalized
    phase_counts: dict[str, int] = {}
    for query in episode_queries:
        phase = str(query.get("rollout_phase", "full_block"))
        phase_counts[phase] = phase_counts.get(phase, 0) + 1

    normalized: list[dict[str, Any]] = []
    for query in episode_queries:
        phase = str(query.get("rollout_phase", "full_block"))
        weight = float(phase_weights.get(phase, 1.0)) / float(max(phase_counts.get(phase, 1), 1))
        updated = dict(query)
        updated["query_weight"] = float(weight)
        normalized.append(updated)
    return normalized


def _replace_snapshot_block(
    snapshot: Sequence[Sequence[np.ndarray]],
    user: int,
    block: int,
    precoder: np.ndarray,
) -> list[list[np.ndarray]]:
    replaced = [list(user_blocks) for user_blocks in snapshot]
    replaced[int(user)][int(block)] = precoder
    return replaced


def _count_uplink_forward_call(
    evaluation_cost_counters: dict[str, Any] | None,
    user: int,
) -> None:
    if evaluation_cost_counters is None:
        return
    evaluation_cost_counters["total_forward_calls"] = int(
        evaluation_cost_counters.get("total_forward_calls", 0)
    ) + 1
    per_user = evaluation_cost_counters.get("per_user_forward_calls")
    if isinstance(per_user, list) and 0 <= int(user) < len(per_user):
        per_user[int(user)] = int(per_user[int(user)]) + 1


def _ensure_precoder_net_snapshot_block(
    uplinksystem: UplinkSystem,
    user_models: Sequence[torch.nn.Module],
    snapshot_cache: list[list[np.ndarray]],
    block_idx: int,
    *,
    evaluation_cost_counters: dict[str, Any] | None = None,
) -> list[list[np.ndarray]]:
    ensure_blocks_up_to(uplinksystem, int(block_idx))
    for k in range(int(uplinksystem.K)):
        while len(snapshot_cache[int(k)]) <= int(block_idx):
            l = len(snapshot_cache[int(k)])
            H_kl = np.asarray(uplinksystem.H[int(k)][int(l)], dtype=np.complex64)
            _count_uplink_forward_call(evaluation_cost_counters, int(k))
            snapshot_cache[int(k)].append(
                infer_precoder_numpy_with_blocklength_and_sigma(
                    user_models[int(k)],
                    H_kl,
                    n_kl=int(uplinksystem.T[int(k)]),
                    sigma2=float(uplinksystem.sigma2[int(k)]),
                    epsilon=float(uplinksystem.epsilon[int(k)]),
                    Nt=int(uplinksystem.NT[int(k)]),
                    dk=int(uplinksystem.dk[int(k)]),
                    P=float(uplinksystem.P[int(k)]),
                    device=DEVICE,
                )
            )
    return snapshot_cache


def _build_uplink_rollout_query(
    *,
    seed: int,
    user: int,
    block: int,
    H: np.ndarray,
    T_ref: int,
    P: float,
    dk: int,
    sigma2: float,
    epsilon: float,
    noise_cov: np.ndarray | None,
    n_kl: int,
    required_bits: int,
    remaining_bits_before_block: int,
    committed_symbols_before_block: float,
    fs: float,
    metrics: dict[str, Any],
    scenario_mode: str,
    rollout_phase: str,
    rollout_stage: str,
    frontier_query: bool,
    objective_weight: float,
) -> dict[str, Any]:
    required_rate = (
        float(required_bits) / float(max(int(n_kl), 1))
        if int(required_bits) > 0
        else 0.0
    )
    rate_margin = float(metrics["rate"] - required_rate)
    return {
        "seed": int(seed),
        "user": int(user),
        "block": int(block),
        "H": np.asarray(H, dtype=np.complex64),
        "T_ref": int(T_ref),
        "P": float(P),
        "dk": int(dk),
        "sigma2": float(sigma2),
        "epsilon": float(epsilon),
        "noise_plus_interference_cov": (
            None if noise_cov is None else np.asarray(noise_cov, dtype=np.complex128)
        ),
        "scenario_mode": str(scenario_mode),
        "n_kl": int(n_kl),
        "rollout_anchor_bits": int(required_bits),
        "remaining_bits_before_block": int(remaining_bits_before_block),
        "committed_symbols_before_block": float(committed_symbols_before_block),
        "fs": float(fs),
        "required_rate": float(required_rate),
        "rate": float(metrics["rate"]),
        "power": float(metrics["power"]),
        "rate_margin": float(rate_margin),
        "power_margin": float(metrics["power_margin"]),
        "feasible": bool(rate_margin >= 0.0 and float(metrics["power_margin"]) >= 0.0),
        "frontier_query": bool(frontier_query),
        "rollout_phase": str(rollout_phase),
        "rollout_stage": str(rollout_stage),
        "query_weight": 1.0,
        "objective_weight": float(objective_weight),
    }


def _collect_uplink_payload_rollout_queries_for_episode(
    system_params: dict[str, Any],
    sim_cfg: dict[str, Any],
    training_episode: dict[str, Any],
    user_models: Sequence[torch.nn.Module],
) -> list[list[dict[str, Any]]]:
    seed = int(training_episode["seed"])
    scenario = dict(training_episode["scenario"])
    K = int(system_params["K"])
    system = UplinkSystem(system_params, seed=int(seed))
    phase_weights = _resolve_uplink_rollout_phase_weights(sim_cfg)
    weighting_mode = _resolve_uplink_rollout_query_weighting_mode(sim_cfg)
    remaining = np.asarray(scenario["payload_bits_per_user"], dtype=int).copy()
    max_blocks = int(sim_cfg.get("max_total_blocks", 256))
    queries_by_user: list[list[dict[str, Any]]] = [[] for _ in range(K)]
    elapsed_symbols = np.zeros(K, dtype=float)
    elapsed_symbols = np.zeros(K, dtype=float)
    block = 0

    while np.any(remaining > 0):
        if block >= max_blocks:
            raise RuntimeError(
                f"Uplink Monte Carlo training rollout hit max_total_blocks={max_blocks} for seed={seed} with remaining bits {remaining.tolist()}."
            )
        ensure_blocks_up_to(system, int(block))
        active_mask = [1 if int(remaining[int(k)]) > 0 else 0 for k in range(K)]
        objective_weights = _uplink_block_objective_weights(
            system,
            sim_cfg,
            block=int(block),
            active_mask=active_mask,
            remaining_bits_by_user=remaining.tolist(),
            block_index_by_user=[int(block) for _ in range(K)],
        )
        snapshot_full = _build_precoder_net_snapshot_for_active_mask(system, user_models, int(block), active_mask)

        for k in range(K):
            if int(active_mask[int(k)]) <= 0:
                continue
            remaining_before_block = int(remaining[int(k)])
            committed_symbols_before_block = float(elapsed_symbols[int(k)])
            H_kl = np.asarray(system.H[int(k)][int(block)], dtype=np.complex64)
            T_ref = int(system.T[int(k)])
            P = float(system.P[int(k)])
            sigma2 = float(system.sigma2[int(k)])
            epsilon = float(system.epsilon[int(k)])
            fs = float(system.fs[int(k)])
            dk = int(system.dk[int(k)])
            F_T = infer_precoder_numpy_with_blocklength_and_sigma(
                user_models[int(k)],
                H_kl,
                n_kl=T_ref,
                sigma2=sigma2,
                epsilon=epsilon,
                Nt=int(system.NT[int(k)]),
                dk=dk,
                P=P,
                device=DEVICE,
            )
            snapshot_candidate = _replace_snapshot_block(snapshot_full, int(k), int(block), F_T)
            cov_T = build_uplink_rate_covariance(
                system,
                sim_cfg,
                int(k),
                int(block),
                F_override=snapshot_candidate,
            )
            base_episode = {
                "H": H_kl,
                "sigma2": sigma2,
                "epsilon": epsilon,
                "P": P,
                "dk": dk,
                "noise_plus_interference_cov": cov_T,
            }
            full_metrics = _evaluate_uplink_rollout_query_numpy(user_models[int(k)], base_episode, T_ref)
            queries_by_user[int(k)].append(
                _build_uplink_rollout_query(
                    seed=seed,
                    user=int(k),
                    block=int(block),
                    H=H_kl,
                    T_ref=T_ref,
                    P=P,
                    dk=dk,
                    sigma2=sigma2,
                    epsilon=epsilon,
                    noise_cov=cov_T,
                    n_kl=T_ref,
                    required_bits=0,
                    remaining_bits_before_block=remaining_before_block,
                    committed_symbols_before_block=committed_symbols_before_block,
                    fs=fs,
                    metrics=full_metrics,
                    scenario_mode=PAYLOAD_COMPLETION_MODE,
                    rollout_phase="full_block",
                    rollout_stage="full_block",
                    frontier_query=False,
                    objective_weight=float(objective_weights[int(k)]),
                )
            )

            supported_bits = max(int(np.floor(float(full_metrics["rate"]) * float(max(T_ref, 1)))), 0)
            committed_bits = min(int(remaining_before_block), int(supported_bits))
            if int(committed_bits) <= 0:
                elapsed_symbols[int(k)] += float(T_ref)
                continue
            if int(committed_bits) < int(remaining_before_block):
                remaining[int(k)] = max(int(remaining[int(k)]) - int(committed_bits), 0)
                elapsed_symbols[int(k)] += float(T_ref)
                continue

            queries_by_user[int(k)].append(
                _build_uplink_rollout_query(
                    seed=seed,
                    user=int(k),
                    block=int(block),
                    H=H_kl,
                    T_ref=T_ref,
                    P=P,
                    dk=dk,
                    sigma2=sigma2,
                    epsilon=epsilon,
                    noise_cov=cov_T,
                    n_kl=T_ref,
                    required_bits=int(committed_bits),
                    remaining_bits_before_block=remaining_before_block,
                    committed_symbols_before_block=committed_symbols_before_block,
                    fs=fs,
                    metrics=full_metrics,
                    scenario_mode=PAYLOAD_COMPLETION_MODE,
                    rollout_phase="tail_feasible",
                    rollout_stage="full_block_commit",
                    frontier_query=False,
                    objective_weight=float(objective_weights[int(k)]),
                )
            )

            search_cfg = _build_monte_carlo_training_search_cfg(
                sim_cfg,
                n_min=int(sim_cfg["n_kl_min"]),
                n_max=int(T_ref),
            )
            def _evaluate_payload_rollout_candidate(candidate: int, stage_name: str) -> dict[str, Any]:
                metrics = _evaluate_uplink_rollout_query_numpy(
                    user_models[int(k)],
                    base_episode,
                    int(candidate),
                )
                query = _build_uplink_rollout_query(
                    seed=seed,
                    user=int(k),
                    block=int(block),
                    H=H_kl,
                    T_ref=T_ref,
                    P=P,
                    dk=dk,
                    sigma2=sigma2,
                    epsilon=epsilon,
                    noise_cov=cov_T,
                    n_kl=int(candidate),
                    required_bits=int(committed_bits),
                    remaining_bits_before_block=remaining_before_block,
                    committed_symbols_before_block=committed_symbols_before_block,
                    fs=fs,
                    metrics=metrics,
                    scenario_mode=PAYLOAD_COMPLETION_MODE,
                    rollout_phase="tail_feasible",
                    rollout_stage=str(stage_name),
                    frontier_query=False,
                    objective_weight=float(objective_weights[int(k)]),
                )
                return {
                    "query": query,
                    "feasible": bool(query["feasible"]),
                }

            search_result = run_n_frontier_search(search_cfg, _evaluate_payload_rollout_candidate)
            for visited in search_result["visited"]:
                query = dict(visited["result"]["query"])
                query["feasible"] = bool(query["feasible"])
                if bool(visited["feasible"]):
                    query["rollout_phase"] = "tail_feasible"
                    query["frontier_query"] = False
                else:
                    query["rollout_phase"] = "tail_frontier"
                    query["frontier_query"] = True
                queries_by_user[int(k)].append(query)
            elapsed_symbols[int(k)] += float(search_result.get("best_n", T_ref))
            remaining[int(k)] = max(int(remaining[int(k)]) - int(committed_bits), 0)
        block += 1

    return [
        _normalize_uplink_episode_query_weights(
            user_queries,
            phase_weights,
            weighting_mode=weighting_mode,
        )
        for user_queries in queries_by_user
    ]


def _collect_uplink_fixed_target_rollout_queries_for_episode(
    system_params: dict[str, Any],
    sim_cfg: dict[str, Any],
    training_episode: dict[str, Any],
    user_models: Sequence[torch.nn.Module],
) -> list[list[dict[str, Any]]]:
    seed = int(training_episode["seed"])
    scenario = dict(training_episode["scenario"])
    block_targets = np.asarray(scenario["block_bit_targets"], dtype=int)
    num_blocks = int(scenario["num_blocks"])
    K = int(system_params["K"])
    system = UplinkSystem(system_params, seed=int(seed))
    phase_weights = _resolve_uplink_rollout_phase_weights(sim_cfg)
    weighting_mode = _resolve_uplink_rollout_query_weighting_mode(sim_cfg)
    queries_by_user: list[list[dict[str, Any]]] = [[] for _ in range(K)]

    for block in range(num_blocks):
        ensure_blocks_up_to(system, int(block))
        active_mask = [1 if int(block_targets[int(k), int(block)]) > 0 else 0 for k in range(K)]
        objective_weights = _uplink_block_objective_weights(
            system,
            sim_cfg,
            block=int(block),
            active_mask=active_mask,
            remaining_bits_by_user=block_targets[:, int(block)].astype(int).tolist(),
            block_index_by_user=[int(block) for _ in range(K)],
        )
        snapshot_full = _build_precoder_net_snapshot_for_active_mask(system, user_models, int(block), active_mask)
        for k in range(K):
            target_bits = int(block_targets[int(k), int(block)])
            if target_bits <= 0:
                continue
            committed_symbols_before_block = float(elapsed_symbols[int(k)])
            H_kl = np.asarray(system.H[int(k)][int(block)], dtype=np.complex64)
            T_ref = int(system.T[int(k)])
            P = float(system.P[int(k)])
            sigma2 = float(system.sigma2[int(k)])
            epsilon = float(system.epsilon[int(k)])
            fs = float(system.fs[int(k)])
            dk = int(system.dk[int(k)])
            F_T = infer_precoder_numpy_with_blocklength_and_sigma(
                user_models[int(k)],
                H_kl,
                n_kl=T_ref,
                sigma2=sigma2,
                epsilon=epsilon,
                Nt=int(system.NT[int(k)]),
                dk=dk,
                P=P,
                device=DEVICE,
            )
            snapshot_candidate = _replace_snapshot_block(snapshot_full, int(k), int(block), F_T)
            cov_T = build_uplink_rate_covariance(
                system,
                sim_cfg,
                int(k),
                int(block),
                F_override=snapshot_candidate,
            )
            base_episode = {
                "H": H_kl,
                "sigma2": sigma2,
                "epsilon": epsilon,
                "P": P,
                "dk": dk,
                "noise_plus_interference_cov": cov_T,
            }
            full_metrics = _evaluate_uplink_rollout_query_numpy(user_models[int(k)], base_episode, T_ref)
            queries_by_user[int(k)].append(
                _build_uplink_rollout_query(
                    seed=seed,
                    user=int(k),
                    block=int(block),
                    H=H_kl,
                    T_ref=T_ref,
                    P=P,
                    dk=dk,
                    sigma2=sigma2,
                    epsilon=epsilon,
                    noise_cov=cov_T,
                    n_kl=T_ref,
                    required_bits=0,
                    remaining_bits_before_block=int(target_bits),
                    committed_symbols_before_block=committed_symbols_before_block,
                    fs=fs,
                    metrics=full_metrics,
                    scenario_mode=FIXED_BLOCK_TARGETS_MODE,
                    rollout_phase="full_block",
                    rollout_stage="full_block",
                    frontier_query=False,
                    objective_weight=float(objective_weights[int(k)]),
                )
            )

            supported_bits = max(int(np.floor(float(full_metrics["rate"]) * float(max(T_ref, 1)))), 0)
            if int(supported_bits) < int(target_bits):
                elapsed_symbols[int(k)] += float(T_ref)
                continue

            queries_by_user[int(k)].append(
                _build_uplink_rollout_query(
                    seed=seed,
                    user=int(k),
                    block=int(block),
                    H=H_kl,
                    T_ref=T_ref,
                    P=P,
                    dk=dk,
                    sigma2=sigma2,
                    epsilon=epsilon,
                    noise_cov=cov_T,
                    n_kl=T_ref,
                    required_bits=int(target_bits),
                    remaining_bits_before_block=int(target_bits),
                    committed_symbols_before_block=committed_symbols_before_block,
                    fs=fs,
                    metrics=full_metrics,
                    scenario_mode=FIXED_BLOCK_TARGETS_MODE,
                    rollout_phase="tail_feasible",
                    rollout_stage="full_block_commit",
                    frontier_query=False,
                    objective_weight=float(objective_weights[int(k)]),
                )
            )

            n_min = int(sim_cfg["n_kl_min"])
            fine_step = max(1, int(sim_cfg["n_kl_step"]))
            coarse_step = max(fine_step, int(sim_cfg.get("monte_carlo_training_n_kl_coarse_step", fine_step)))
            last_feasible_n = int(T_ref)
            first_infeasible_query: dict[str, Any] | None = None

            candidate = int(T_ref) - int(coarse_step)
            while candidate >= int(n_min):
                metrics = _evaluate_uplink_rollout_query_numpy(user_models[int(k)], base_episode, int(candidate))
                query = _build_uplink_rollout_query(
                    seed=seed,
                    user=int(k),
                    block=int(block),
                    H=H_kl,
                    T_ref=T_ref,
                    P=P,
                    dk=dk,
                    sigma2=sigma2,
                    epsilon=epsilon,
                    noise_cov=cov_T,
                    n_kl=int(candidate),
                    required_bits=int(target_bits),
                    remaining_bits_before_block=int(target_bits),
                    committed_symbols_before_block=committed_symbols_before_block,
                    fs=fs,
                    metrics=metrics,
                    scenario_mode=FIXED_BLOCK_TARGETS_MODE,
                    rollout_phase="tail_feasible",
                    rollout_stage="coarse",
                    frontier_query=False,
                    objective_weight=float(objective_weights[int(k)]),
                )
                if bool(query["feasible"]):
                    queries_by_user[int(k)].append(query)
                    last_feasible_n = int(candidate)
                    candidate -= int(coarse_step)
                    continue
                query["rollout_phase"] = "tail_frontier"
                query["frontier_query"] = True
                first_infeasible_query = query
                break

            if (
                first_infeasible_query is not None
                and int(fine_step) < int(coarse_step)
                and int(last_feasible_n) - int(fine_step) > int(first_infeasible_query["n_kl"])
            ):
                candidate = int(last_feasible_n) - int(fine_step)
                first_infeasible_n = int(first_infeasible_query["n_kl"])
                while candidate > int(first_infeasible_n):
                    metrics = _evaluate_uplink_rollout_query_numpy(user_models[int(k)], base_episode, int(candidate))
                    query = _build_uplink_rollout_query(
                        seed=seed,
                        user=int(k),
                        block=int(block),
                        H=H_kl,
                        T_ref=T_ref,
                        P=P,
                        dk=dk,
                        sigma2=sigma2,
                        epsilon=epsilon,
                        noise_cov=cov_T,
                        n_kl=int(candidate),
                        required_bits=int(target_bits),
                        remaining_bits_before_block=int(target_bits),
                        committed_symbols_before_block=committed_symbols_before_block,
                        fs=fs,
                        metrics=metrics,
                        scenario_mode=FIXED_BLOCK_TARGETS_MODE,
                        rollout_phase="tail_feasible",
                        rollout_stage="fine",
                        frontier_query=False,
                        objective_weight=float(objective_weights[int(k)]),
                    )
                    if bool(query["feasible"]):
                        queries_by_user[int(k)].append(query)
                        last_feasible_n = int(candidate)
                        candidate -= int(fine_step)
                        continue
                    query["rollout_phase"] = "tail_frontier"
                    query["frontier_query"] = True
                    first_infeasible_query = query
                    break

            if first_infeasible_query is not None:
                queries_by_user[int(k)].append(first_infeasible_query)
            elapsed_symbols[int(k)] += float(last_feasible_n)

    return [
        _normalize_uplink_episode_query_weights(
            user_queries,
            phase_weights,
            weighting_mode=weighting_mode,
        )
        for user_queries in queries_by_user
    ]


def _annotate_uplink_latency_aligned_queries(
    queries_by_user: Sequence[Sequence[dict[str, Any]]],
) -> list[list[dict[str, Any]]]:
    annotated: list[list[dict[str, Any]]] = []
    for user_queries in queries_by_user:
        future_latency_ms = 0.0
        updated_user_queries: list[dict[str, Any]] = []
        for query in reversed([dict(q) for q in user_queries]):
            safe_fs = max(float(query.get("fs", 1.0)), 1e-30)
            future_latency_ms += 1000.0 * float(max(int(query.get("n_kl", 0)), 0)) / safe_fs
            remaining_before = int(query.get("remaining_bits_before_block", 0))
            query["committed_bits_target"] = int(query.get("rollout_anchor_bits", 0))
            query["latency_weight_ms_per_bit"] = (
                float(future_latency_ms) / float(max(remaining_before, 1))
                if remaining_before > 0
                else 0.0
            )
            query["rollout_phase"] = "chosen_rollout_state"
            query["rollout_stage"] = str(query.get("rollout_stage", "chosen_rollout_state"))
            query["frontier_query"] = False
            updated_user_queries.append(query)
        annotated.append(list(reversed(updated_user_queries)))
    return annotated


def _collect_uplink_payload_latency_aligned_queries_for_episode(
    system_params: dict[str, Any],
    sim_cfg: dict[str, Any],
    training_episode: dict[str, Any],
    user_models: Sequence[torch.nn.Module],
) -> list[list[dict[str, Any]]]:
    seed = int(training_episode["seed"])
    scenario = dict(training_episode["scenario"])
    K = int(system_params["K"])
    system = UplinkSystem(system_params, seed=int(seed))
    weighting_mode = _resolve_uplink_rollout_query_weighting_mode(sim_cfg)
    remaining = np.asarray(scenario["payload_bits_per_user"], dtype=int).copy()
    max_blocks = int(sim_cfg.get("max_total_blocks", 256))
    queries_by_user: list[list[dict[str, Any]]] = [[] for _ in range(K)]
    elapsed_symbols = np.zeros(K, dtype=float)
    block = 0

    while np.any(remaining > 0):
        if block >= max_blocks:
            raise RuntimeError(
                f"Uplink Monte Carlo latency-aligned training rollout hit max_total_blocks={max_blocks} for seed={seed} with remaining bits {remaining.tolist()}."
            )
        ensure_blocks_up_to(system, int(block))
        active_mask = [1 if int(remaining[int(k)]) > 0 else 0 for k in range(K)]
        snapshot_full = _build_precoder_net_snapshot_for_active_mask(system, user_models, int(block), active_mask)

        for k in range(K):
            if int(active_mask[int(k)]) <= 0:
                continue
            remaining_before_block = int(remaining[int(k)])
            committed_symbols_before_block = float(elapsed_symbols[int(k)])
            H_kl = np.asarray(system.H[int(k)][int(block)], dtype=np.complex64)
            T_ref = int(system.T[int(k)])
            P = float(system.P[int(k)])
            sigma2 = float(system.sigma2[int(k)])
            epsilon = float(system.epsilon[int(k)])
            fs = float(system.fs[int(k)])
            dk = int(system.dk[int(k)])
            F_T = infer_precoder_numpy_with_blocklength_and_sigma(
                user_models[int(k)],
                H_kl,
                n_kl=T_ref,
                sigma2=sigma2,
                epsilon=epsilon,
                Nt=int(system.NT[int(k)]),
                dk=dk,
                P=P,
                device=DEVICE,
            )
            snapshot_candidate = _replace_snapshot_block(snapshot_full, int(k), int(block), F_T)
            cov_T = build_uplink_rate_covariance(
                system,
                sim_cfg,
                int(k),
                int(block),
                F_override=snapshot_candidate,
            )
            base_episode = {
                "H": H_kl,
                "sigma2": sigma2,
                "epsilon": epsilon,
                "P": P,
                "dk": dk,
                "noise_plus_interference_cov": cov_T,
            }
            full_metrics = _evaluate_uplink_rollout_query_numpy(user_models[int(k)], base_episode, T_ref)
            supported_bits = max(int(np.floor(float(full_metrics["rate"]) * float(max(T_ref, 1)))), 0)
            committed_bits = min(int(remaining_before_block), int(supported_bits))
            chosen_n = int(T_ref)
            chosen_metrics = dict(full_metrics)

            if int(committed_bits) >= int(remaining_before_block) and int(remaining_before_block) > 0:
                search_cfg = _build_monte_carlo_test_search_cfg(
                    sim_cfg,
                    n_min=int(sim_cfg["n_kl_min"]),
                    n_max=int(T_ref),
                )

                def _evaluate_latency_candidate(candidate: int, stage_name: str) -> dict[str, Any]:
                    metrics = _evaluate_uplink_rollout_query_numpy(
                        user_models[int(k)],
                        base_episode,
                        int(candidate),
                    )
                    return {
                        "feasible": bool(
                            float(metrics["rate"])
                            >= (float(remaining_before_block) / float(max(int(candidate), 1)))
                        ),
                        "metrics": metrics,
                        "stage": str(stage_name),
                    }

                search_result = run_n_frontier_search(search_cfg, _evaluate_latency_candidate)
                chosen_n = int(search_result.get("best_n", T_ref))
                if int(chosen_n) != int(T_ref):
                    chosen_event = None
                    for visited in search_result.get("visited", []):
                        if int(visited.get("n_kl", -1)) == int(chosen_n):
                            chosen_event = visited
                            break
                    if chosen_event is not None:
                        chosen_metrics = dict(chosen_event["result"]["metrics"])

            chosen_query = _build_uplink_rollout_query(
                seed=seed,
                user=int(k),
                block=int(block),
                H=H_kl,
                T_ref=T_ref,
                P=P,
                dk=dk,
                sigma2=sigma2,
                epsilon=epsilon,
                noise_cov=cov_T,
                n_kl=int(chosen_n),
                required_bits=int(committed_bits),
                remaining_bits_before_block=remaining_before_block,
                committed_symbols_before_block=committed_symbols_before_block,
                fs=fs,
                metrics=chosen_metrics,
                scenario_mode=PAYLOAD_COMPLETION_MODE,
                rollout_phase="chosen_rollout_state",
                rollout_stage="chosen_rollout_state",
                frontier_query=False,
                objective_weight=1.0,
            )
            queries_by_user[int(k)].append(chosen_query)
            remaining[int(k)] = max(int(remaining[int(k)]) - int(committed_bits), 0)
            elapsed_symbols[int(k)] += float(chosen_n)
        block += 1

    return [
        _normalize_uplink_episode_query_weights(
            user_queries,
            {},
            weighting_mode=weighting_mode,
        )
        for user_queries in _annotate_uplink_latency_aligned_queries(queries_by_user)
    ]


def _generate_rollout_queries_for_training_episodes(
    system_params: dict[str, Any],
    sim_cfg: dict[str, Any],
    training_episodes: Sequence[dict[str, Any]],
    user_models: Sequence[torch.nn.Module],
    *,
    training_style: str,
) -> list[list[dict[str, Any]]]:
    K = int(system_params["K"])
    queries_by_user: list[list[dict[str, Any]]] = [[] for _ in range(K)]
    for training_episode in training_episodes:
        scenario_mode = str(training_episode.get("scenario_mode", PAYLOAD_COMPLETION_MODE))
        if str(training_style) == EXACT_ROLLOUT_LATENCY_ALIGNED_TRAINING_STYLE:
            if scenario_mode != PAYLOAD_COMPLETION_MODE:
                raise ValueError(
                    "The exact_rollout_latency_aligned Monte Carlo training style currently supports payload_completion only."
                )
            episode_queries = _collect_uplink_payload_latency_aligned_queries_for_episode(
                system_params,
                sim_cfg,
                training_episode,
                user_models,
            )
        elif scenario_mode == FIXED_BLOCK_TARGETS_MODE:
            episode_queries = _collect_uplink_fixed_target_rollout_queries_for_episode(
                system_params,
                sim_cfg,
                training_episode,
                user_models,
            )
        else:
            episode_queries = _collect_uplink_payload_rollout_queries_for_episode(
                system_params,
                sim_cfg,
                training_episode,
                user_models,
            )
        for k in range(K):
            queries_by_user[int(k)].extend(episode_queries[int(k)])
    return queries_by_user


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
        "configured_max_epochs": int(training_history.get("configured_max_epochs", epochs)),
        "per_user_epochs_completed": [
            int(v) for v in training_history.get("per_user_epochs_completed", [len(history) for history in per_user_lagrangian])
        ],
        "per_user_training_solve_status": [
            str(v) for v in training_history.get("per_user_training_solve_status", ["unknown" for _ in per_user_lagrangian])
        ],
        "per_user_restored_solution_source": [
            str(v) for v in training_history.get("per_user_restored_solution_source", ["unknown" for _ in per_user_lagrangian])
        ],
        "base_dataset_kind": dataset_summary.get("base_dataset_kind", "channel_episodes_only"),
        "total_training_channel_episodes": int(dataset_summary.get("total_channel_episodes", 0)),
        "rollout_anchor_bits_mode": "derived_online_from_current_episode_served_bits",
        "rollout_query_weighting_mode": str(
            training_history.get("rollout_query_weighting_mode", "phase_balanced")
        ),
        "monte_carlo_training_style": str(
            training_history.get("monte_carlo_training_style", ROLLOUT_QUERY_LAGRANGIAN_TRAINING_STYLE)
        ),
        "uplink_objective_mode": str(
            training_history.get("uplink_objective_mode", UNWEIGHTED_SUM_RATE_OBJECTIVE)
        ),
        "beam_reward_mode": str(
            training_history.get("beam_reward_mode", RATE_BEAM_REWARD_MODE)
        ),
        "rollout_phase_weights": dict(training_history.get("rollout_phase_weights", {})),
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
        "per_user_last_epoch_avg_rate_over_rollout_queries": [
            float(history[-1]) if len(history) > 0 else 0.0 for history in per_user_rate
        ],
        "per_user_last_epoch_avg_lagrangian_over_rollout_queries": [
            float(history[-1]) if len(history) > 0 else 0.0 for history in per_user_lagrangian
        ],
        "per_user_final_rate_violation": [
            float(history[-1]) if len(history) > 0 else 0.0 for history in per_user_rate_violation
        ],
        "per_user_final_power_violation": [
            float(history[-1]) if len(history) > 0 else 0.0 for history in per_user_power_violation
        ],
        "per_user_final_kkt_primal_residual": [
            float(history[-1]) if len(history) > 0 else 0.0
            for history in training_history.get("per_user_kkt_primal_residual", [])
        ],
        "per_user_final_kkt_complementarity_residual": [
            float(history[-1]) if len(history) > 0 else 0.0
            for history in training_history.get("per_user_kkt_complementarity_residual", [])
        ],
        "per_user_final_kkt_stationarity_residual": [
            float(history[-1]) if len(history) > 0 else 0.0
            for history in training_history.get("per_user_kkt_stationarity_residual", [])
        ],
        "final_avg_lagrangian": float(avg_lagrangian[-1]) if len(avg_lagrangian) > 0 else 0.0,
        "best_avg_lagrangian": float(min(avg_lagrangian)) if len(avg_lagrangian) > 0 else 0.0,
        "last_epoch_mean_user_rollout_lagrangian": float(avg_lagrangian[-1]) if len(avg_lagrangian) > 0 else 0.0,
        "best_epoch_mean_user_rollout_lagrangian": float(min(avg_lagrangian)) if len(avg_lagrangian) > 0 else 0.0,
        "final_avg_user_rate": float(avg_user_rate[-1]) if len(avg_user_rate) > 0 else 0.0,
        "best_avg_user_rate": float(max(avg_user_rate)) if len(avg_user_rate) > 0 else 0.0,
        "last_epoch_mean_user_rollout_rate": float(avg_user_rate[-1]) if len(avg_user_rate) > 0 else 0.0,
        "best_epoch_mean_user_rollout_rate": float(max(avg_user_rate)) if len(avg_user_rate) > 0 else 0.0,
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
        "last_epoch_total_rollout_queries": int(
            training_history.get("final_epoch_rollout_query_summary", {}).get("total_rollout_queries", 0)
        ),
        "last_epoch_feasible_rollout_queries": int(
            training_history.get("final_epoch_rollout_query_summary", {})
            .get("global_rollout_queries_by_feasibility", {})
            .get("feasible", 0)
        ),
        "last_epoch_infeasible_rollout_queries": int(
            training_history.get("final_epoch_rollout_query_summary", {})
            .get("global_rollout_queries_by_feasibility", {})
            .get("infeasible", 0)
        ),
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
    epochs: int | None = None,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> dict:
    system_params, sim_cfg = get_config(cfg_name)
    K = int(system_params["K"])
    objective_mode = resolve_uplink_objective_mode(
        sim_cfg.get("uplink_objective_mode", UNWEIGHTED_SUM_RATE_OBJECTIVE)
    )
    max_epochs = max(
        1,
        int(epochs if epochs is not None else sim_cfg.get("monte_carlo_training_max_epochs", sim_cfg.get("max_epochs", 20))),
    )
    print_every_epoch = max(1, int(sim_cfg.get("print_every_epoch", 1)))
    kkt_primal_tol = float(sim_cfg.get("kkt_primal_tol", sim_cfg.get("convergence_feasibility_tol", 1e-5)))
    kkt_complementarity_tol = float(
        sim_cfg.get("kkt_complementarity_tol", sim_cfg.get("convergence_feasibility_tol", 1e-5))
    )
    kkt_stationarity_tol = float(
        sim_cfg.get("kkt_stationarity_tol", sim_cfg.get("convergence_precoder_tol", 1e-4))
    )
    training_episodes = build_training_dataset(cfg_name, train_seeds)
    dataset_summary = summarize_training_dataset(training_episodes)
    rollout_phase_weights = _resolve_uplink_rollout_phase_weights(sim_cfg)
    rollout_query_weighting_mode = _resolve_uplink_rollout_query_weighting_mode(sim_cfg)
    training_style = _resolve_uplink_monte_carlo_training_style(sim_cfg)
    beam_reward_mode = resolve_uplink_beam_reward_mode(
        sim_cfg.get("beam_reward_mode", RATE_BEAM_REWARD_MODE)
    )
    training_history = {
        "per_user_lagrangian": [[] for _ in range(K)],
        "per_user_rate": [[] for _ in range(K)],
        "avg_rate_violation": [[] for _ in range(K)],
        "avg_power_violation": [[] for _ in range(K)],
        "per_user_kkt_primal_residual": [[] for _ in range(K)],
        "per_user_kkt_complementarity_residual": [[] for _ in range(K)],
        "per_user_kkt_stationarity_residual": [[] for _ in range(K)],
        "per_user_training_epoch_status": [[] for _ in range(K)],
        "avg_lagrangian": [],
        "avg_user_rate": [],
        "avg_rate_violation_over_users": [],
        "avg_power_violation_over_users": [],
        "dataset_summary": dataset_summary,
        "rollout_query_summaries_per_user": [[] for _ in range(K)],
        "rollout_phase_weights": dict(rollout_phase_weights),
        "rollout_query_weighting_mode": str(rollout_query_weighting_mode),
        "monte_carlo_training_style": str(training_style),
        "uplink_objective_mode": str(objective_mode),
        "beam_reward_mode": str(beam_reward_mode),
        "configured_max_epochs": int(max_epochs),
        "training_objective": (
            "chosen_rollout_latency_aligned_served_bits_payload"
            if str(training_style) == EXACT_ROLLOUT_LATENCY_ALIGNED_TRAINING_STYLE
            else (
                "scenario_driven_episode_rollout_lagrangian_"
                f"{objective_mode}_user_{beam_reward_mode}"
            )
        ),
    }
    user_models = [
        build_user_precoder_net_with_blocklength_and_sigma(
            Nr=int(system_params["NR"][k]),
            Nt=int(system_params["NT"][k]),
            dk=int(system_params["dk"][k]),
            device=DEVICE,
        )
        for k in range(K)
    ]
    optimizers = [
        torch.optim.Adam(user_models[k].parameters(), lr=float(lr))
        for k in range(K)
    ]
    rngs = [
        np.random.default_rng(int(train_seeds[0]) + 17 * (k + 1) if len(train_seeds) > 0 else (17 * (k + 1)))
        for k in range(K)
    ]
    constraint_loss_form = _resolve_constraint_loss_form(sim_cfg)
    augmented_lagrangian_rho_rate = float(sim_cfg.get("augmented_lagrangian_rho_rate", 0.0))
    augmented_lagrangian_rho_power = float(sim_cfg.get("augmented_lagrangian_rho_power", 0.0))
    lambda_rate = np.full(K, float(sim_cfg.get("initial_lambda_rate_constraint", 0.1)), dtype=float)
    lambda_power = np.full(K, float(sim_cfg.get("initial_lambda_power_constraint", 0.01)), dtype=float)
    lr_rate = float(sim_cfg.get("lr_rate_constraint", 1e-2))
    lr_power = float(sim_cfg.get("lr_power_constraint", 1e-3))
    cumulative_rollout_query_global_counts: dict[int, int] = {}
    cumulative_rollout_query_per_user_counts: list[dict[int, int]] = [{} for _ in range(K)]
    cumulative_frontier_query_global_counts: dict[int, int] = {}
    cumulative_frontier_query_per_user_counts: list[dict[int, int]] = [{} for _ in range(K)]
    last_epoch_queries_by_user: list[list[dict[str, Any]]] = [[] for _ in range(K)]
    previous_epoch_model_states: list[dict[str, torch.Tensor] | None] = [None for _ in range(K)]
    best_primal_residual = [float("inf") for _ in range(K)]
    best_feasible_rate = [-float("inf") for _ in range(K)]
    best_primal_model_states = [_clone_model_state(model) for model in user_models]
    best_primal_optimizer_states = [copy.deepcopy(optimizer.state_dict()) for optimizer in optimizers]
    best_primal_lambda_rate = np.array(lambda_rate, copy=True)
    best_primal_lambda_power = np.array(lambda_power, copy=True)
    best_feasible_model_states: list[dict[str, torch.Tensor] | None] = [None for _ in range(K)]
    best_feasible_optimizer_states: list[dict[str, Any] | None] = [None for _ in range(K)]
    best_feasible_lambda_rate: list[float | None] = [None for _ in range(K)]
    best_feasible_lambda_power: list[float | None] = [None for _ in range(K)]
    per_user_solve_status = ["max_epochs_reached" for _ in range(K)]
    epochs_completed = 0

    print(
        format_log_line(
            "[UL Monte Carlo Train]",
            phase="start",
            channel_episodes=int(len(training_episodes)),
            epochs=int(max_epochs),
            batch_size=int(batch_size),
        )
    )

    for epoch in range(int(max_epochs)):
        epochs_completed = int(epoch + 1)
        rollout_queries_by_user = _generate_rollout_queries_for_training_episodes(
            system_params,
            sim_cfg,
            training_episodes,
            user_models,
            training_style=training_style,
        )
        last_epoch_queries_by_user = [
            [dict(query) for query in user_queries]
            for user_queries in rollout_queries_by_user
        ]
        rollout_summary = _summarize_rollout_queries_by_user(rollout_queries_by_user)

        for n_key, count in rollout_summary.get("global_rollout_queries_by_n_kl", {}).items():
            n_val = int(n_key)
            cumulative_rollout_query_global_counts[n_val] = (
                cumulative_rollout_query_global_counts.get(n_val, 0) + int(count)
            )
        for user_idx, user_summary in enumerate(rollout_summary.get("per_user", [])):
            epoch_rollout_summary = dict(user_summary)
            epoch_rollout_summary["epoch"] = int(epoch + 1)
            training_history["rollout_query_summaries_per_user"][int(user_idx)].append(epoch_rollout_summary)
            for n_key, count in epoch_rollout_summary.get("rollout_queries_by_n_kl", {}).items():
                n_val = int(n_key)
                cumulative_rollout_query_per_user_counts[int(user_idx)][n_val] = (
                    cumulative_rollout_query_per_user_counts[int(user_idx)].get(n_val, 0) + int(count)
                )
            for n_key, count in epoch_rollout_summary.get("frontier_rollout_queries_by_n_kl", {}).items():
                n_val = int(n_key)
                cumulative_frontier_query_per_user_counts[int(user_idx)][n_val] = (
                    cumulative_frontier_query_per_user_counts[int(user_idx)].get(n_val, 0) + int(count)
                )
        for n_key, count in rollout_summary.get("global_frontier_rollout_queries_by_n_kl", {}).items():
            n_val = int(n_key)
            cumulative_frontier_query_global_counts[n_val] = (
                cumulative_frontier_query_global_counts.get(n_val, 0) + int(count)
            )

        epoch_lagrangians: list[float] = []
        epoch_rates: list[float] = []
        epoch_rate_violations: list[float] = []
        epoch_power_violations: list[float] = []
        epoch_statuses: list[str] = []

        for k in range(K):
            model = user_models[int(k)]
            optimizer = optimizers[int(k)]
            rollout_queries = rollout_queries_by_user[int(k)]
            Nt = int(system_params["NT"][k])
            dk = int(system_params["dk"][k])
            if len(rollout_queries) == 0:
                training_history["per_user_lagrangian"][k].append(0.0)
                training_history["per_user_rate"][k].append(0.0)
                training_history["avg_rate_violation"][k].append(0.0)
                training_history["avg_power_violation"][k].append(0.0)
                training_history["per_user_kkt_primal_residual"][k].append(0.0)
                training_history["per_user_kkt_complementarity_residual"][k].append(0.0)
                training_history["per_user_kkt_stationarity_residual"][k].append(0.0)
                training_history["per_user_training_epoch_status"][k].append("no_rollout_queries")
                epoch_lagrangians.append(0.0)
                epoch_rates.append(0.0)
                epoch_rate_violations.append(0.0)
                epoch_power_violations.append(0.0)
                epoch_statuses.append("no_rollout_queries")
                continue

            model.train()
            indices = np.arange(len(rollout_queries))
            rngs[int(k)].shuffle(indices)
            epoch_term_sum = 0.0
            epoch_rate_sum = 0.0
            epoch_rate_violation_sum = 0.0
            epoch_power_violation_sum = 0.0
            epoch_query_weight_sum = 0.0

            for start in range(0, len(indices), max(int(batch_size), 1)):
                batch_idx = indices[start : start + max(int(batch_size), 1)]
                optimizer.zero_grad()
                loss = torch.zeros((), dtype=torch.float32, device=DEVICE)
                batch_rate_violation = 0.0
                batch_power_violation = 0.0
                batch_query_weight_sum = 0.0

                for idx in batch_idx:
                    query = rollout_queries[int(idx)]
                    query_weight = float(query.get("query_weight", 1.0))
                    H_t = torch.as_tensor(query["H"], dtype=torch.complex64, device=DEVICE)
                    noise_cov_t = (
                        None
                        if query.get("noise_plus_interference_cov") is None
                        else torch.as_tensor(
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
                    required_rate = float(query.get("required_rate", 0.0))
                    objective_weight = float(query.get("objective_weight", 1.0))
                    rate_violation = rate.new_tensor(required_rate) - rate
                    power_violation = power - float(query["P"])
                    rate_violation_pos = _constraint_violation_activation(rate_violation, constraint_loss_form)
                    power_violation_pos = _constraint_violation_activation(power_violation, constraint_loss_form)
                    if str(training_style) == EXACT_ROLLOUT_LATENCY_ALIGNED_TRAINING_STYLE:
                        reward = (
                            rate.new_tensor(float(query.get("latency_weight_ms_per_bit", 0.0)))
                            * _uplink_soft_served_bits_torch(
                                rate,
                                n_kl=int(query["n_kl"]),
                                remaining_bits_before_block=int(
                                    query.get("remaining_bits_before_block", 0)
                                ),
                            )
                        )
                    else:
                        reward = _uplink_training_beam_reward_torch(
                            rate,
                            n_kl=int(query["n_kl"]),
                            requested_bits=int(
                                query.get(
                                    "remaining_bits_before_block",
                                    query.get("rollout_anchor_bits", 0),
                                )
                            ),
                            beam_reward_mode=beam_reward_mode,
                            committed_symbols_before_block=float(
                                query.get("committed_symbols_before_block", 0.0)
                            ),
                            fs=float(query.get("fs", 1.0)),
                        )
                    term = (
                        -(objective_weight * reward)
                        + float(lambda_rate[int(k)]) * rate_violation_pos
                        + float(lambda_power[int(k)]) * power_violation_pos
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
                lambda_rate[int(k)] = max(
                    0.0,
                    float(lambda_rate[int(k)]) + lr_rate * (batch_rate_violation / float(batch_query_weight_sum)),
                )
                lambda_power[int(k)] = max(
                    0.0,
                    float(lambda_power[int(k)]) + lr_power * (batch_power_violation / float(batch_query_weight_sum)),
                )

            avg_lagrangian = float(epoch_term_sum / max(epoch_query_weight_sum, 1.0))
            avg_rate = float(epoch_rate_sum / max(epoch_query_weight_sum, 1.0))
            avg_rate_violation = float(epoch_rate_violation_sum / max(epoch_query_weight_sum, 1.0))
            avg_power_violation = float(epoch_power_violation_sum / max(epoch_query_weight_sum, 1.0))
            epoch_r_p = float(max(max(avg_rate_violation, 0.0), max(avg_power_violation, 0.0)))
            epoch_r_c = float(
                max(
                    abs(float(lambda_rate[int(k)]) * max(avg_rate_violation, 0.0)),
                    abs(float(lambda_power[int(k)]) * max(avg_power_violation, 0.0)),
                )
            )
            epoch_r_s = _relative_model_state_change(model, previous_epoch_model_states[int(k)])
            previous_epoch_model_states[int(k)] = _clone_model_state(model)
            exact_feasible = float(avg_rate_violation) <= 0.0 and float(avg_power_violation) <= 0.0
            if epoch_r_p < best_primal_residual[int(k)]:
                best_primal_residual[int(k)] = float(epoch_r_p)
                best_primal_model_states[int(k)] = _clone_model_state(model)
                best_primal_optimizer_states[int(k)] = copy.deepcopy(optimizer.state_dict())
                best_primal_lambda_rate[int(k)] = float(lambda_rate[int(k)])
                best_primal_lambda_power[int(k)] = float(lambda_power[int(k)])
            if exact_feasible and avg_rate >= best_feasible_rate[int(k)]:
                best_feasible_rate[int(k)] = float(avg_rate)
                best_feasible_model_states[int(k)] = _clone_model_state(model)
                best_feasible_optimizer_states[int(k)] = copy.deepcopy(optimizer.state_dict())
                best_feasible_lambda_rate[int(k)] = float(lambda_rate[int(k)])
                best_feasible_lambda_power[int(k)] = float(lambda_power[int(k)])

            epoch_status = "running"
            if (
                epoch_r_p <= kkt_primal_tol
                and epoch_r_c <= kkt_complementarity_tol
                and epoch_r_s <= kkt_stationarity_tol
            ):
                epoch_status = "kkt_converged"
                per_user_solve_status[int(k)] = "kkt_converged"
            elif epoch > 0 and epoch_r_s <= kkt_stationarity_tol and epoch_r_p > kkt_primal_tol:
                epoch_status = "stationary_infeasible"
                per_user_solve_status[int(k)] = "stationary_infeasible"

            training_history["per_user_lagrangian"][k].append(avg_lagrangian)
            training_history["per_user_rate"][k].append(avg_rate)
            training_history["avg_rate_violation"][k].append(avg_rate_violation)
            training_history["avg_power_violation"][k].append(avg_power_violation)
            training_history["per_user_kkt_primal_residual"][k].append(float(epoch_r_p))
            training_history["per_user_kkt_complementarity_residual"][k].append(float(epoch_r_c))
            training_history["per_user_kkt_stationarity_residual"][k].append(float(epoch_r_s))
            training_history["per_user_training_epoch_status"][k].append(str(epoch_status))
            epoch_lagrangians.append(avg_lagrangian)
            epoch_rates.append(avg_rate)
            epoch_rate_violations.append(avg_rate_violation)
            epoch_power_violations.append(avg_power_violation)
            epoch_statuses.append(epoch_status)

            if (
                ((epoch + 1) % print_every_epoch) == 0
                or epoch == 0
                or epoch_status in {"kkt_converged", "stationary_infeasible"}
            ):
                print(
                    format_progress_log_line(
                        "[UL Monte Carlo]",
                        phase="train",
                        method="monte_carlo",
                        user=int(k),
                        epoch=f"{epoch + 1}/{int(max_epochs)}",
                        objective=avg_lagrangian,
                        rate=avg_rate,
                        r_p=epoch_r_p,
                        r_c=epoch_r_c,
                        r_s=epoch_r_s,
                        status=epoch_status,
                    )
                )
            model.eval()

        training_history["avg_lagrangian"].append(float(np.mean(epoch_lagrangians)) if epoch_lagrangians else 0.0)
        training_history["avg_user_rate"].append(float(np.mean(epoch_rates)) if epoch_rates else 0.0)
        training_history["avg_rate_violation_over_users"].append(
            float(np.mean(epoch_rate_violations)) if epoch_rate_violations else 0.0
        )
        training_history["avg_power_violation_over_users"].append(
            float(np.mean(epoch_power_violations)) if epoch_power_violations else 0.0
        )

        terminal_statuses = {"kkt_converged", "stationary_infeasible", "no_rollout_queries"}
        if all(status == "kkt_converged" for status in epoch_statuses if status != "no_rollout_queries"):
            break
        if epoch_statuses and all(status in terminal_statuses for status in epoch_statuses):
            break

    training_history.setdefault("per_user_epochs_completed", [0 for _ in range(K)])
    training_history.setdefault("per_user_training_solve_status", ["not_started" for _ in range(K)])
    training_history.setdefault("per_user_restored_solution_source", ["not_started" for _ in range(K)])

    for k in range(K):
        restored_solution_source = "best_primal"
        if best_feasible_model_states[int(k)] is not None and best_feasible_optimizer_states[int(k)] is not None:
            user_models[int(k)].load_state_dict(best_feasible_model_states[int(k)])
            optimizers[int(k)].load_state_dict(best_feasible_optimizer_states[int(k)])
            if best_feasible_lambda_rate[int(k)] is not None:
                lambda_rate[int(k)] = float(best_feasible_lambda_rate[int(k)])
            if best_feasible_lambda_power[int(k)] is not None:
                lambda_power[int(k)] = float(best_feasible_lambda_power[int(k)])
            restored_solution_source = "best_feasible"
            if per_user_solve_status[int(k)] == "max_epochs_reached":
                per_user_solve_status[int(k)] = "max_epochs_feasible_best"
        else:
            user_models[int(k)].load_state_dict(best_primal_model_states[int(k)])
            optimizers[int(k)].load_state_dict(best_primal_optimizer_states[int(k)])
            lambda_rate[int(k)] = float(best_primal_lambda_rate[int(k)])
            lambda_power[int(k)] = float(best_primal_lambda_power[int(k)])
            if per_user_solve_status[int(k)] == "max_epochs_reached":
                per_user_solve_status[int(k)] = "max_epochs_best_primal"
        if training_history["per_user_training_epoch_status"][int(k)]:
            training_history["per_user_training_epoch_status"][int(k)][-1] = str(per_user_solve_status[int(k)])
        training_history["per_user_epochs_completed"][int(k)] = int(epochs_completed)
        training_history["per_user_training_solve_status"][int(k)] = str(per_user_solve_status[int(k)])
        training_history["per_user_restored_solution_source"][int(k)] = str(restored_solution_source)
        user_models[int(k)].eval()

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
        epochs=int(max_epochs),
        dataset_summary=dataset_summary,
        initial_baseline=train_eval_initial_baseline,
    )

    train_eval_post.update(
        {
            "train_seeds": [int(s) for s in train_seeds],
            "training_dataset_sizes": [int(len(training_episodes)) for _ in range(K)],
            "training_channel_episode_counts_per_user": [int(len(training_episodes)) for _ in range(K)],
            "training_sample_counts_per_user": [int(len(training_episodes)) for _ in range(K)],
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
            "monte_carlo_training_style": str(
                training_history.get(
                    "monte_carlo_training_style",
                    ROLLOUT_QUERY_LAGRANGIAN_TRAINING_STYLE,
                )
            ),
            "uplink_objective_mode": str(training_history.get("uplink_objective_mode", objective_mode)),
            "beam_reward_mode": str(training_history.get("beam_reward_mode", beam_reward_mode)),
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
            snapshot_candidate = _replace_snapshot_block(snapshot_full, int(k), int(block), F_T)
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
                snapshot_candidate = _replace_snapshot_block(snapshot_full, int(k), int(block), F_n)
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
    evaluation_cost_counters = {
        "per_user_forward_calls": [0 for _ in range(K)],
        "total_forward_calls": 0,
    }

    n_kl_min = int(sim_cfg["n_kl_min"])
    n_kl_step = int(sim_cfg["n_kl_step"])
    use_interference = uses_uplink_interference(sim_cfg)
    snapshot_cache: list[list[np.ndarray]] | None = ([[] for _ in range(K)] if use_interference else None)

    for k in range(K):
        print(
            format_log_line(
                "[UL Monte Carlo Eval]",
                phase="start",
                user=int(k),
            )
        )
        B_rem = int(uplinksystem.B[k])
        ell = 0

        while B_rem > 0:
            ensure_blocks_up_to(uplinksystem, ell)

            H_kl = np.asarray(uplinksystem.H[k][ell], dtype=np.complex64)
            T_ref = int(uplinksystem.T[k])
            P = float(uplinksystem.P[k])
            sigma2 = float(uplinksystem.sigma2[k])
            epsilon = float(uplinksystem.epsilon[k])

            snapshot_full = None
            if snapshot_cache is not None:
                snapshot_full = _ensure_precoder_net_snapshot_block(
                    uplinksystem,
                    user_models,
                    snapshot_cache,
                    int(ell),
                    evaluation_cost_counters=evaluation_cost_counters,
                )

            print(
                format_log_line(
                    "[UL Monte Carlo Eval]",
                    user=int(k),
                    block=int(ell),
                    remaining_bits=int(B_rem),
                )
            )

            S_block = []

            _count_uplink_forward_call(evaluation_cost_counters, int(k))
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
            snapshot_candidate = (
                _replace_snapshot_block(snapshot_full, int(k), int(ell), F_T)
                if snapshot_full is not None
                else None
            )
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
            if int(B_used) < int(B_rem):
                print(
                    format_log_line(
                        "[UL Monte Carlo Eval]",
                        user=int(k),
                        block=int(ell),
                        status="stop_partial_payload",
                    )
                )
            else:
                search_cfg = _build_monte_carlo_test_search_cfg(
                    sim_cfg,
                    n_min=int(n_kl_min),
                    n_max=int(T_ref),
                )

                def _evaluate_payload_eval_candidate(candidate_n: int, stage_name: str) -> dict[str, Any]:
                    _count_uplink_forward_call(evaluation_cost_counters, int(k))
                    F_n = infer_precoder_numpy_with_blocklength_and_sigma(
                        user_models[k],
                        H_kl,
                        n_kl=int(candidate_n),
                        sigma2=sigma2,
                        epsilon=epsilon,
                        Nt=int(uplinksystem.NT[k]),
                        dk=int(uplinksystem.dk[k]),
                        P=P,
                        device=DEVICE,
                    )
                    snapshot_candidate = (
                        _replace_snapshot_block(snapshot_full, int(k), int(ell), F_n)
                        if snapshot_full is not None
                        else None
                    )
                    cov_n = build_uplink_rate_covariance(
                        uplinksystem,
                        sim_cfg,
                        k,
                        ell,
                        F_override=snapshot_candidate,
                    )
                    R_n = _compute_r_fbl_np(H_kl, F_n, sigma2, epsilon, int(candidate_n), cov_n)
                    rate_violation = (float(B_used) / float(max(int(candidate_n), 1))) - float(R_n)
                    print(
                        format_log_line(
                            "[UL Monte Carlo Eval]",
                            user=int(k),
                            block=int(ell),
                            n_kl=int(candidate_n),
                            search_stage=str(stage_name),
                            achieved_rate=float(R_n),
                            rate_violation=float(rate_violation),
                        )
                    )
                    return {
                        "feasible": bool(rate_violation <= 0.0),
                        "R_fbl": float(R_n),
                        "F": np.array(F_n, copy=True),
                    }

                reduction_search = run_n_frontier_search(search_cfg, _evaluate_payload_eval_candidate)
                for accepted in reduction_search["accepted"]:
                    best_n = int(accepted["n_kl"])
                    best_R = float(accepted["result"]["R_fbl"])
                    best_F = torch.tensor(accepted["result"]["F"], dtype=torch.complex64)
                    S_block.append(
                        {
                            "n_kl": int(best_n),
                            "n": int(best_n),
                            "B_l": int(B_used),
                            "Bits per sub-block length B/n_kl": float(B_used) / float(best_n),
                            "F": torch.tensor(accepted["result"]["F"], dtype=torch.complex64),
                            "R_fbl": float(best_R),
                            "F_power": float(np.linalg.norm(accepted["result"]["F"], "fro") ** 2),
                            "lambda_rate": 0.0,
                            "lambda_power": 0.0,
                            "loss_curve": [],
                            "method": method_name,
                        }
                    )

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
        "evaluation_cost_counters": evaluation_cost_counters,
        "scenario_mode": PAYLOAD_COMPLETION_MODE,
    }


train_blocklength_aware_precoder = train_blocklength_aware_precoder_net
train_blocklength_aware_policy = train_blocklength_aware_precoder_net
_build_precoder_snapshot = _build_precoder_net_snapshot
_build_policy_snapshot = _build_precoder_net_snapshot
evaluate_blocklength_precoder = evaluate_blocklength_precoder_net
evaluate_blocklength_policy = evaluate_blocklength_precoder_net
