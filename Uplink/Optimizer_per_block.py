import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm

from experiment_scenarios import FIXED_BLOCK_TARGETS_MODE, build_experiment_scenario
from precoder_models import (
    DEVICE,
    build_user_precoder_net_with_sigma_context,
    export_user_model_specs,
    export_user_model_states,
    infer_precoder_numpy_with_sigma_context,
    infer_precoder_torch_with_sigma_context,
    load_user_precoder_models,
    net_output_to_precoder,
    project_precoder_power,
)
from uplink_rate_model import build_uplink_rate_covariance
LOG2E_SQ = (np.log2(np.e)) ** 2
CONSTRAINT_LOSS_FORMS = {"plain_lagrangian", "augmented_lagrangian"}
POWER_PROJECTION_SAFETY_MARGIN = 1e-6

# ============================================================
# Utils
# ============================================================
def Q_inv(eps: float, device=DEVICE, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(norm.ppf(1.0 - float(eps)), device=device, dtype=dtype)

def update_lambdas(lambda_rate: float, lambda_power: float,
                   rate_violation_pos: torch.Tensor, power_violation_pos: torch.Tensor,
                   lr_rate: float, lr_power: float):
    r = float(rate_violation_pos.detach().item())
    p = float(power_violation_pos.detach().item())
    lambda_rate = max(0.0, float(lambda_rate) + float(lr_rate) * r)
    lambda_power = max(0.0, float(lambda_power) + float(lr_power) * p)
    return lambda_rate, lambda_power


def resolve_constraint_loss_form(raw_mode: str) -> str:
    mode = str(raw_mode).strip().lower()
    if mode not in CONSTRAINT_LOSS_FORMS:
        known = ", ".join(sorted(CONSTRAINT_LOSS_FORMS))
        raise ValueError(f"Unknown constraint loss form '{raw_mode}'. Expected one of: {known}")
    return mode


def _constraint_violation_activation(value: torch.Tensor, loss_form: str) -> torch.Tensor:
    if loss_form == "plain_lagrangian":
        return F.leaky_relu(value)
    return torch.relu(value)


def project_power(Fmat: torch.Tensor, P: float, eps: float = 1e-12, delta: float = 1e-9) -> torch.Tensor:
    """
    Enforce ||F||_F^2 < P by scaling with a small safety margin.
    delta: fractional margin so final power ≈ (1 - delta)^2 * P
    """
    power = (torch.linalg.norm(Fmat, ord="fro") ** 2).real

    if float(power.detach().cpu()) <= float(P):
        return Fmat

    # Slightly under-scale to guarantee strict inequality
    scale = torch.sqrt(
        torch.tensor(float(P), device=Fmat.device, dtype=torch.float32)
        / (power + eps)
    )

    scale = scale * (1.0 - float(POWER_PROJECTION_SAFETY_MARGIN))

    return Fmat * scale.to(Fmat.dtype)

def print_result(result_list, key: str):
    print(f"\nResult {key}: ", end="")
    for d in result_list[::-1]:
        v = d.get(key, None)
        if torch.is_tensor(v):
            v = float(v.detach().cpu())
        print(v, end=", ")
    print()

# ============================================================
# Precoder MLP
# ============================================================
class UplinkMLP_PrecoderNet(nn.Module):
    def __init__(self, Nr: int, Nt: int, dk: int):
        super().__init__()
        in_dim = 2 * Nr * Nt
        out_dim = 2 * Nt * dk

        h1 = max(512, 8 * out_dim)
        h2 = max(256, 4 * out_dim)
        h3 = max(128, 2 * out_dim)

        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, h3), nn.ReLU(),
            nn.Linear(h3, out_dim),
        )

    def forward(self, H_kl: torch.Tensor) -> torch.Tensor:
        # H_kl: (Nr, Nt) complex64
        H_flat = H_kl.reshape(1, -1)
        x = torch.cat([H_flat.real, H_flat.imag], dim=1)  # (1, 2*Nr*Nt)
        return self.net(x)  # (1, 2*Nt*dk)

# ============================================================
# Lagrangian Loss (finite blocklength rate)
# ============================================================
class LagrangianLoss(nn.Module):
    def __init__(
        self,
        H_kl: torch.Tensor,
        sigma2: float,
        epsilon: float,
        B: float,
        P: float,
        n_kl: int,
        noise_plus_interference_cov: torch.Tensor | None = None,
        constraint_loss_form: str = "plain_lagrangian",
        augmented_lagrangian_rho_rate: float = 0.0,
        augmented_lagrangian_rho_power: float = 0.0,
    ):
        super().__init__()
        self.H_kl = H_kl
        self.sigma2 = float(sigma2)
        self.epsilon = float(epsilon)
        self.B = float(B)
        self.P = float(P)
        self.n_kl = int(n_kl)
        self.noise_plus_interference_cov = noise_plus_interference_cov
        self.constraint_loss_form = resolve_constraint_loss_form(constraint_loss_form)
        self.augmented_lagrangian_rho_rate = float(augmented_lagrangian_rho_rate)
        self.augmented_lagrangian_rho_power = float(augmented_lagrangian_rho_power)

    def set_blocklength(self, n_kl: int):
        self.n_kl = int(n_kl)

    def set_payload(self, B: float):
        self.B = float(B)

    def R_fbl(self, Fmat: torch.Tensor) -> torch.Tensor:
        """
        Fmat: (Nt, dk) complex64
        """
        
        H = self.H_kl
        Nr = H.shape[0] 
        I = torch.eye(Nr, dtype=torch.complex64, device=H.device)

        HF = H @ Fmat  # (Nr, dk)

        if self.noise_plus_interference_cov is None:
            noise_cov = self.sigma2 * I
        else:
            noise_cov = self.noise_plus_interference_cov.to(device=H.device, dtype=torch.complex64)

        chol = torch.linalg.cholesky(noise_cov)
        G = torch.linalg.solve(chol, HF)
        A = G @ G.conj().transpose(1, 0)  # whitened effective signal covariance

        sign, logdet = torch.linalg.slogdet(I + A)
        C = logdet / np.log(2.0)  # log2(det(.))
        if not torch.isfinite(A).all():
            bad = (~torch.isfinite(A)).nonzero(as_tuple=False)[:10]
            raise RuntimeError(f"A has NaN/Inf. First bad indices: {bad}")

        # Force exact Hermitian (complex) / symmetric (real)
        A = 0.5 * (A + A.conj().transpose(1, 0))

        eigvals = torch.linalg.eigvalsh(A)  # (Nr,) real
        V = torch.sum(eigvals * (eigvals + 2.0) / (eigvals + 1.0) ** 2) * LOG2E_SQ

        R = C - torch.sqrt(V / float(self.n_kl)) * Q_inv(self.epsilon, device=Fmat.device)
        return R.real

    def forward(self, Fmat: torch.Tensor, lambda_rate: float, lambda_power: float):
        R = self.R_fbl(Fmat)
        F_power = (torch.linalg.norm(Fmat, ord="fro") ** 2).real
        
        

        # constraints (positive => violation)
        rate_violation = (self.B / float(self.n_kl)) - R
        power_violation = F_power - self.P

        rate_violation_pos = torch.relu(rate_violation)
        power_violation_pos = torch.relu(power_violation)
        rate_constraint_term = _constraint_violation_activation(rate_violation, self.constraint_loss_form)
        power_constraint_term = _constraint_violation_activation(power_violation, self.constraint_loss_form)

        loss = (-R
                + float(lambda_rate) * rate_constraint_term
                + float(lambda_power) * power_constraint_term)
        if self.constraint_loss_form == "augmented_lagrangian":
            loss = (
                loss
                + 0.5 * float(self.augmented_lagrangian_rho_rate) * rate_violation_pos.pow(2)
                + 0.5 * float(self.augmented_lagrangian_rho_power) * power_violation_pos.pow(2)
            )

        return (
            loss,
            R,
            F_power,
            rate_violation,
            power_violation,
            rate_violation_pos,
            power_violation_pos,
        )

# ============================================================
# Optimize precoder for a fixed n_kl
# ============================================================
def optimize_precoder_for_nl(
    precoder_net: nn.Module,
    loss_fn: LagrangianLoss,
    Nt: int,
    dk: int,
    max_epochs: int,
    lambda_rate: float,
    lambda_power: float,
    lr_rate: float,
    lr_power: float,
    optimizer: torch.optim.Optimizer,
    kkt_primal_tol: float = 1e-5,
    kkt_complementarity_tol: float = 1e-5,
    kkt_stationarity_tol: float = 1e-5,
):
    losses = []
    max_epochs = max(1, int(max_epochs))

    kkt_history: list[dict[str, float]] = []
    best_primal_residual = float("inf")
    best_feasible_rate = -float("inf")
    solve_status = "max_epochs_reached"

    best_primal_model_state = {
        key: value.detach().cpu().clone()
        for key, value in precoder_net.state_dict().items()
    }
    best_primal_optimizer_state = copy.deepcopy(optimizer.state_dict())
    best_primal_lambda_rate = float(lambda_rate)
    best_primal_lambda_power = float(lambda_power)
    previous_precoder_eval: torch.Tensor | None = None

    best_feasible_model_state: dict[str, torch.Tensor] | None = None
    best_feasible_optimizer_state: dict | None = None
    best_feasible_lambda_rate: float | None = None
    best_feasible_lambda_power: float | None = None

    for epoch_idx in range(max_epochs):
        Fmat = infer_precoder_torch_with_sigma_context(
            precoder_net,
            loss_fn.H_kl,
            loss_fn.sigma2,
            loss_fn.epsilon,
            Nt,
            dk,
            loss_fn.P,
        )

        optimizer.zero_grad()
        loss, R, F_power, rv_raw, pv_raw, rv_pos, pv_pos = loss_fn(Fmat, lambda_rate, lambda_power)
        loss.backward()

        r_p = max(float(rv_pos.detach().cpu()), float(pv_pos.detach().cpu()))
        r_c = max(
            abs(float(lambda_rate) * float(rv_pos.detach().cpu())),
            abs(float(lambda_power) * float(pv_pos.detach().cpu())),
        )
        if previous_precoder_eval is None:
            r_s = float("inf")
        else:
            delta_num = float(
                torch.linalg.norm(Fmat.detach() - previous_precoder_eval, ord="fro").detach().cpu()
            )
            delta_den = max(
                float(torch.linalg.norm(previous_precoder_eval, ord="fro").detach().cpu()),
                1e-12,
            )
            r_s = float(delta_num / delta_den)
        exact_feasible = (
            float(rv_raw.detach().cpu()) <= 0.0
            and float(pv_raw.detach().cpu()) <= 0.0
        )
        losses.append(float(loss.detach().cpu()))
        kkt_history.append(
            {
                "epoch": float(epoch_idx + 1),
                "primal_residual": float(r_p),
                "complementarity_residual": float(r_c),
                "stationarity_residual": float(r_s),
                "rate_gap": float(rv_raw.detach().cpu()),
                "power_gap": float(pv_raw.detach().cpu()),
                "rate_violation": float(rv_pos.detach().cpu()),
                "power_violation": float(pv_pos.detach().cpu()),
                "rate": float(R.detach().cpu()),
                "power": float(F_power.detach().cpu()),
                "lambda_rate": float(lambda_rate),
                "lambda_power": float(lambda_power),
            }
        )
        previous_precoder_eval = Fmat.detach().clone()

        lambda_rate, lambda_power = update_lambdas(
            lambda_rate, lambda_power, rv_pos, pv_pos, lr_rate, lr_power
        )

        if r_p < best_primal_residual:
            best_primal_residual = float(r_p)
            best_primal_model_state = {
                key: value.detach().cpu().clone()
                for key, value in precoder_net.state_dict().items()
            }
            best_primal_optimizer_state = copy.deepcopy(optimizer.state_dict())
            best_primal_lambda_rate = float(lambda_rate)
            best_primal_lambda_power = float(lambda_power)

        if exact_feasible and float(R.detach().cpu()) >= best_feasible_rate:
            best_feasible_rate = float(R.detach().cpu())
            best_feasible_model_state = {
                key: value.detach().cpu().clone()
                for key, value in precoder_net.state_dict().items()
            }
            best_feasible_optimizer_state = copy.deepcopy(optimizer.state_dict())
            best_feasible_lambda_rate = float(lambda_rate)
            best_feasible_lambda_power = float(lambda_power)

        if (
            r_p <= float(kkt_primal_tol)
            and r_c <= float(kkt_complementarity_tol)
            and r_s <= float(kkt_stationarity_tol)
        ):
            solve_status = "kkt_converged"
            break

        if previous_precoder_eval is not None and r_s <= float(kkt_stationarity_tol) and r_p > float(kkt_primal_tol):
            solve_status = "stationary_infeasible"
            break

        if (epoch_idx + 1) < max_epochs:
            optimizer.step()

    if best_feasible_model_state is not None:
        precoder_net.load_state_dict(best_feasible_model_state)
        if best_feasible_optimizer_state is not None:
            optimizer.load_state_dict(best_feasible_optimizer_state)
        lambda_rate = float(best_feasible_lambda_rate)
        lambda_power = float(best_feasible_lambda_power)
        if solve_status == "max_epochs_reached":
            solve_status = "max_epochs_feasible_best"
    else:
        precoder_net.load_state_dict(best_primal_model_state)
        optimizer.load_state_dict(best_primal_optimizer_state)
        lambda_rate = float(best_primal_lambda_rate)
        lambda_power = float(best_primal_lambda_power)
        if solve_status == "max_epochs_reached":
            solve_status = "max_epochs_best_primal"

    with torch.no_grad():
        F_final = infer_precoder_torch_with_sigma_context(
            precoder_net,
            loss_fn.H_kl,
            loss_fn.sigma2,
            loss_fn.epsilon,
            Nt,
            dk,
            loss_fn.P,
        )
        loss, R, F_power, rv_raw, pv_raw, rv_pos, pv_pos = loss_fn(F_final, lambda_rate, lambda_power)

    return {
        "F": F_final.detach(),
        "lambda_rate": float(lambda_rate),
        "lambda_power": float(lambda_power),
        "R_fbl": float(R.detach().item()),
        "F_power": float(F_power.detach().item()),
        "rate_gap": float(rv_raw.detach().item()),
        "power_gap": float(pv_raw.detach().item()),
        "rate_violation": float(rv_pos.detach().item()),
        "power_violation": float(pv_pos.detach().item()),
        "loss_curve": losses,
        "solve_status": solve_status,
        "kkt_history": kkt_history,
        "final_primal_residual": float(max(float(rv_pos.detach().cpu()), float(pv_pos.detach().cpu()))),
        "final_complementarity_residual": float(
            max(
                abs(float(lambda_rate) * float(rv_pos.detach().cpu())),
                abs(float(lambda_power) * float(pv_pos.detach().cpu())),
            )
        ),
    }

# ============================================================
# OptimizeSubBlocklength-Precoder (returns + logs like your older pipeline)
# ============================================================
def optimize_subblocklength_precoder(
    uplinksystem,
    user: int,
    block: int,
    B_rem: int,
    lambda_rate_0: float,
    lambda_power_0: float,
    sim_cfg: dict,
    precoder_net: nn.Module,
    optimizer: torch.optim.Optimizer,
    interference_F_snapshot=None,
    bit_budget_role: str = "payload_completion",
):
    is_fixed_target_block = str(bit_budget_role).strip().lower() == "fixed_block_targets"
    bit_budget_label = "target_bits" if is_fixed_target_block else "B_rem"

    print(f"\n========== OptimizeSubBlocklength-Precoder ==========")
    print(f"User: {user}, Block: {block}")
    print(f"Initial {bit_budget_label}: {B_rem}")
    print("-----------------------------------------------------")

    P = float(uplinksystem.P[user])
    Nr = int(uplinksystem.NR[user])
    Nt = int(uplinksystem.NT[user])
    dk = int(uplinksystem.dk[user])
    sigma2 = float(uplinksystem.sigma2[user])
    epsilon = float(uplinksystem.epsilon[user])

    H_kl = torch.tensor(uplinksystem.H[user][block], dtype=torch.complex64, device=DEVICE)
    noise_plus_interference_cov_np = build_uplink_rate_covariance(
        uplinksystem,
        sim_cfg,
        user,
        block,
        F_override=interference_F_snapshot,
    )
    noise_plus_interference_cov = (
        None
        if noise_plus_interference_cov_np is None
        else torch.tensor(
            noise_plus_interference_cov_np,
            dtype=torch.complex64,
            device=DEVICE,
        )
    )

    T = int(uplinksystem.T[user])
    n_kl_max = int(uplinksystem.T[user])
    n_kl_min = int(sim_cfg["n_kl_min"])
    n_kl_step = int(sim_cfg["n_kl_step"])
    max_epochs = max(1, int(sim_cfg["max_epochs"]))
    reduced_n_kl_log_interval = max(1, int(sim_cfg.get("reduced_n_kl_log_interval", 1)))
    
    # psuedo_n_kl_max = int(1000*np.sin(1/real_n_kl_max))
    # psuedo_n_kl_max = int(B_rem*1.2)
    # n_kl_max = psuedo_n_kl_max

    lr_net = float(sim_cfg["lr_net"])
    lr_rate = float(sim_cfg["lr_rate_constraint"])
    lr_power = float(sim_cfg["lr_power_constraint"])
    kkt_primal_tol = float(sim_cfg.get("kkt_primal_tol", sim_cfg.get("convergence_feasibility_tol", 1e-5)))
    kkt_complementarity_tol = float(
        sim_cfg.get("kkt_complementarity_tol", sim_cfg.get("convergence_feasibility_tol", 1e-5))
    )
    kkt_stationarity_tol = float(
        sim_cfg.get("kkt_stationarity_tol", sim_cfg.get("convergence_precoder_tol", 1e-4))
    )
    constraint_loss_form = resolve_constraint_loss_form(
        sim_cfg.get("constraint_loss_form", "plain_lagrangian")
    )
    augmented_lagrangian_rho_rate = float(sim_cfg.get("augmented_lagrangian_rho_rate", 0.0))
    augmented_lagrangian_rho_power = float(sim_cfg.get("augmented_lagrangian_rho_power", 0.0))

    lambda_rate = float(lambda_rate_0)
    lambda_power = float(lambda_power_0)

    loss_fn = LagrangianLoss(
        H_kl=H_kl, sigma2=sigma2, epsilon=epsilon,
        B=float(B_rem), P=float(P), n_kl=n_kl_max,
        noise_plus_interference_cov=noise_plus_interference_cov,
        constraint_loss_form=constraint_loss_form,
        augmented_lagrangian_rho_rate=augmented_lagrangian_rho_rate,
        augmented_lagrangian_rho_power=augmented_lagrangian_rho_power,
    ).to(DEVICE)

    results = []

    # ------------------------------------------------------------
    # STEP A: Optimize once at n = T, then clip the served bits to the
    # maximum payload that the optimized beam can support.
    # ------------------------------------------------------------
    print(f"\n--- Step A: Optimize at n = T = {n_kl_max} ---")
    loss_fn.set_blocklength(n_kl_max)
    loss_fn.set_payload(float(B_rem))
    B_initial = int(B_rem)
    out = optimize_precoder_for_nl(
        precoder_net=precoder_net,
        loss_fn=loss_fn,
        Nt=Nt, dk=dk,
        max_epochs=max_epochs,
        lambda_rate=lambda_rate, lambda_power=lambda_power,
        lr_rate=lr_rate, lr_power=lr_power,
        optimizer=optimizer,
        kkt_primal_tol=kkt_primal_tol,
        kkt_complementarity_tol=kkt_complementarity_tol,
        kkt_stationarity_tol=kkt_stationarity_tol,
    )

    step_a_diagnostics = {
        "achieved_R_fbl": float(out["R_fbl"]),
        "F": out["F"],
        "F_power": float(out["F_power"]),
        "lambda_rate": float(out["lambda_rate"]),
        "lambda_power": float(out["lambda_power"]),
        "rate_gap": float(out["rate_gap"]),
        "power_gap": float(out["power_gap"]),
        "loss_curve": out["loss_curve"],
        "kkt_history": copy.deepcopy(out.get("kkt_history", [])),
        "solve_status": out.get("solve_status", "unknown"),
        "final_primal_residual": float(out.get("final_primal_residual", 0.0)),
        "final_complementarity_residual": float(out.get("final_complementarity_residual", 0.0)),
    }

    lambda_rate = out["lambda_rate"]
    lambda_power = out["lambda_power"]

    print(f"R_fbl: {out['R_fbl']}")
    print(f"F_power: {out['F_power']}")
    print(f"Rate gap: {out['rate_gap']}")
    print(f"Power gap: {out['power_gap']}")

    if out["power_gap"] > 0.0:
        print(">>> Power constraint violated at n = T. STOP.")
        return [], 0, step_a_diagnostics

    B_max_T = max(int(np.floor(float(n_kl_max) * float(out["R_fbl"]))), 0)
    B_used = int(min(B_initial, B_max_T))
    fully_feasible_at_T = out["rate_gap"] <= 0.0

    if B_used <= 0:
        if is_fixed_target_block:
            print(">>> No feasible service at n = T for this block target.")
        else:
            print(">>> No feasible service at n = T for this payload.")
        return [], 0, step_a_diagnostics

    if fully_feasible_at_T:
        print(">>> Requested bits are feasible at n = T.")
    else:
        print(
            f">>> Requested bits are not feasible at n = T. "
            f"Serving the feasible payload B_used={B_used} without retry."
        )

    results.append({
        "n_kl": int(n_kl_max),
        "n": int(n_kl_max),
        "B_l": int(B_used),
        "Bits per sub-block length B/n_kl": float(B_used) / float(n_kl_max),
        "required_R_fbl": float(B_initial) / float(max(int(n_kl_max), 1)),
        "achieved_R_fbl": float(out["R_fbl"]),
        "F": out["F"],
        "R_fbl": float(out["R_fbl"]),
        "F_power": float(out["F_power"]),
        "lambda_rate": float(lambda_rate),
        "lambda_power": float(lambda_power),
        "loss_curve": out["loss_curve"],
        "kkt_history": copy.deepcopy(out.get("kkt_history", [])),
        "solve_status": out.get("solve_status", "unknown"),
        "final_primal_residual": float(out.get("final_primal_residual", 0.0)),
        "final_complementarity_residual": float(out.get("final_complementarity_residual", 0.0)),
    })

    bits_left_after_current_request = int(B_initial - B_used)
    if is_fixed_target_block:
        print(
            f"\n>>> Block target processed at n=T: B_used={B_used}, "
            f"unserved_bits={bits_left_after_current_request}"
        )
    else:
        print(
            f"\n>>> Payload accepted at n=T: B_used={B_used}, "
            f"remaining after this block would be B_rem={bits_left_after_current_request}"
        )

    # ------------------------------------------------------------
    # STEP B: Reduce n while keeping B fixed. Each new n triggers a fresh
    # inner optimization warm-started from the last feasible solution.
    # ------------------------------------------------------------
    print(f"\n--- Step B: Reduce n while keeping served bits B = {B_used} fixed ---")
    n_kl = n_kl_max - n_kl_step

    while n_kl >= n_kl_min:
        if bits_left_after_current_request != 0:
            if is_fixed_target_block:
                print(f"\n Not reducing n_kl because the block target was only partially served (n_kl = {n_kl})")
            else:
                print(f"\n Not reducing n_kl since not last sub-block (n_kl = {n_kl})")
            break
        should_print_candidate = (
            n_kl == n_kl_max - n_kl_step
            or n_kl == n_kl_min
            or ((n_kl_max - n_kl) % max(int(n_kl_step) * int(reduced_n_kl_log_interval), 1) == 0)
        )
        if should_print_candidate:
            print(f"\nRe-optimizing at n = {n_kl}")
        model_checkpoint = {
            key: value.detach().cpu().clone()
            for key, value in precoder_net.state_dict().items()
        }
        optimizer_checkpoint = copy.deepcopy(optimizer.state_dict())
        lambda_rate_checkpoint = float(lambda_rate)
        lambda_power_checkpoint = float(lambda_power)

        loss_fn.set_blocklength(int(n_kl))
        loss_fn.set_payload(float(B_used))
        out = optimize_precoder_for_nl(
            precoder_net=precoder_net,
            loss_fn=loss_fn,
            Nt=Nt,
            dk=dk,
            max_epochs=max_epochs,
            lambda_rate=lambda_rate,
            lambda_power=lambda_power,
            lr_rate=lr_rate,
            lr_power=lr_power,
            optimizer=optimizer,
            kkt_primal_tol=kkt_primal_tol,
            kkt_complementarity_tol=kkt_complementarity_tol,
            kkt_stationarity_tol=kkt_stationarity_tol,
        )
        lambda_rate = out["lambda_rate"]
        lambda_power = out["lambda_power"]

        if should_print_candidate:
            print(f"R_fbl: {out['R_fbl']}")
            print(f"Rate gap: {out['rate_gap']}")
            print(f"Power gap: {out['power_gap']}")

        feasible = (out["rate_gap"] <= 0.0) and (out["power_gap"] <= 0.0)
        if not feasible:
            precoder_net.load_state_dict(model_checkpoint)
            optimizer.load_state_dict(optimizer_checkpoint)
            lambda_rate = float(lambda_rate_checkpoint)
            lambda_power = float(lambda_power_checkpoint)
            print(">>> Not feasible after re-optimization. Stop decreasing n.")
            break

        if should_print_candidate:
            print(">>> Feasible. Storing result.")
        results.append({
            "n_kl": int(n_kl),
            "n": int(n_kl),
            "B_l": int(B_used),
            "Bits per sub-block length B/n_kl": float(B_used) / float(n_kl),
            "required_R_fbl": float(B_used) / float(max(int(n_kl), 1)),
            "achieved_R_fbl": float(out["R_fbl"]),
            "F": out["F"],
            "R_fbl": float(out["R_fbl"]),
            "F_power": float(out["F_power"]),
            "lambda_rate": float(lambda_rate),
            "lambda_power": float(lambda_power),
            "loss_curve": out["loss_curve"],
            "kkt_history": copy.deepcopy(out.get("kkt_history", [])),
            "solve_status": out.get("solve_status", "unknown"),
            "final_primal_residual": float(out.get("final_primal_residual", 0.0)),
            "final_complementarity_residual": float(out.get("final_complementarity_residual", 0.0)),
        })

        n_kl -= n_kl_step

    print("\n>>> Finished OptimizeSubBlocklength-Precoder")
    print_result(results, "n_kl")
    print_result(results, "R_fbl")
    print_result(results, "F_power")
    print_result(results, "lambda_rate")
    print_result(results, "lambda_power")

    return results, int(B_used), step_a_diagnostics

# ============================================================
# Training loop (per user, per block) - same outputs as before
# ============================================================

import numpy as np

def dynamic_subblocklength_precoder_training(
    uplinksystem,
    sim_cfg: dict,
    channel_norm: bool = True,
    interference_F_snapshot=None,
    commit_live_precoders: bool = True,
):
    """
    Training version that is consistent with the TEST logic style:
      - For each user, iteratively allocate blocks until B_rem == 0
      - For each block:
          Step A: optimize once at n=T and clip the served bits to the
                  feasible payload supported by that beam
          Step B: if the whole request was served, decrease n_kl and
                  re-optimize the precoder at each new n_kl
      - Saves per-block trajectories S for plotting.

    Returns:
      post_training_data_dict with:
        - L_out, n_star, F_star, R_star, norm_stats
        - all_user_block_results: list[user][block] = S (trajectory over n_kl)
        - B_used_star: list[user][block] = B_used at n=T (after reduction)
        - B_kl_star:   list[user][block] = bits actually transmitted in that block
    """
    K = int(uplinksystem.K)

    L_out = [1] * K
    n_star = [[] for _ in range(K)]
    F_star = [[] for _ in range(K)]
    R_star = [[] for _ in range(K)]

    # extra bookkeeping (useful for plots/repro)
    B_used_star = [[] for _ in range(K)]
    B_kl_star = [[] for _ in range(K)]

    norm_stats = []
    all_user_block_results = [[] for _ in range(K)]

    lambda_rate_0 = float(sim_cfg["initial_lambda_rate_constraint"])
    lambda_power_0 = float(sim_cfg["initial_lambda_power_constraint"])
    user_precoder_models: list[nn.Module] = []

    for k in range(K):
        print(f"\n================ TRAIN USER {k} ================")

        # ---- normalize user channel across all currently available blocks ----
        H_user = np.array(uplinksystem.H[k], dtype=np.complex64)
        mean = np.mean(H_user)
        var = np.mean(np.abs(H_user - mean) ** 2) + 1e-12
        norm_stats.append((mean, var))

        if channel_norm:
            uplinksystem.H[k] = list((H_user - mean) / np.sqrt(var))

        # ---- remaining payload ----
        B_rem = int(uplinksystem.B[k])
        ell = 0

        # ensure at least one block exists
        while len(uplinksystem.H[k]) < 1:
            uplinksystem.add_block(k)

        user_model = build_user_precoder_net_with_sigma_context(
            Nr=int(uplinksystem.NR[k]),
            Nt=int(uplinksystem.NT[k]),
            dk=int(uplinksystem.dk[k]),
            device=DEVICE,
        )
        user_optimizer = torch.optim.Adam(user_model.parameters(), lr=float(sim_cfg["lr_net"]))
        user_precoder_models.append(user_model)

        # reset lambdas per user (or keep global if you prefer)
        lambda_rate_user = float(lambda_rate_0)
        lambda_power_user = float(lambda_power_0)

        while B_rem > 0:
            # ensure block exists
            if ell >= len(uplinksystem.H[k]):
                uplinksystem.add_block(k)

            print(f"\n--- TRAIN User {k}, Block {ell}, B_rem={B_rem} ---")

            # This function already performs:
            #  Step A: optimize once at n=T and clip to feasible served bits
            #  Step B: if this is the tail block, decrease n_kl and re-optimize
            S, B_used, _ = optimize_subblocklength_precoder(
                uplinksystem=uplinksystem,
                user=k,
                block=ell,
                B_rem=B_rem,
                lambda_rate_0=lambda_rate_user,
                lambda_power_0=lambda_power_user,
                sim_cfg=sim_cfg,
                precoder_net=user_model,
                optimizer=user_optimizer,
                interference_F_snapshot=interference_F_snapshot,
            )

            if len(S) == 0 or B_used <= 0:
                print(f">>> STOP training user {k} at block {ell} (no feasible).")
                break

            # Save trajectory for plotting (your plot_optimization_result_train expects this)
            all_user_block_results[k].append(S)

            # Pick smallest feasible n_kl in this block
            best = S[-1]
            n_opt = int(best["n_kl"])
            F_opt = best["F"]
            R_opt = float(best["R_fbl"])

            # Save chosen decisions
            n_star[k].append(n_opt)
            F_star[k].append(F_opt)
            R_star[k].append(R_opt)
            if commit_live_precoders:
                uplinksystem.F[k][ell] = F_opt.detach().cpu().numpy()


            # Save payload bookkeeping
            B_used_star[k].append(int(B_used))
            B_kl = min(B_rem, int(B_used))
            B_kl_star[k].append(int(B_kl))

            B_rem -= B_kl
            print(f">>> Chosen: n_kl={n_opt}, B_used={B_used}, B_kl={B_kl}, remaining B_rem={B_rem}")

            # advance block if bits remain
            if B_rem > 0:
                ell += 1
                L_out[k] = ell + 1

        # write back to system bookkeeping (optional)
        uplinksystem.L[k] = int(L_out[k])
        if len(n_star[k]) > 0:
            uplinksystem.n_kl[k] = list(n_star[k])
    post_training_data_dict = {
        "L_out": L_out,
        "n_star": n_star,
        "F_star": F_star,
        "R_star": R_star,
        "norm_stats": norm_stats,
        "all_user_block_results_train": all_user_block_results,
        "B_used_star": B_used_star,
        "B_kl_star": B_kl_star,
        "user_model_specs": export_user_model_specs(
            uplinksystem.NR,
            uplinksystem.NT,
            uplinksystem.dk,
            input_mode="channel_sigma_epsilon",
        ),
        "user_model_states": export_user_model_states(user_precoder_models),
        "precoder_parameterization": "shared_user_channel_sigma_epsilon_to_precoder_mlp",
    }
    return post_training_data_dict


def dynamic_fixed_target_precoder_training(
    uplinksystem,
    sim_cfg: dict,
    channel_norm: bool = True,
    interference_F_snapshot=None,
    commit_live_precoders: bool = True,
):
    scenario = build_experiment_scenario(uplinksystem.sc, sim_cfg, seed=int(uplinksystem.seed))
    if str(scenario["mode"]) != FIXED_BLOCK_TARGETS_MODE:
        raise ValueError("dynamic_fixed_target_precoder_training requires fixed_block_targets scenario mode.")

    block_targets = np.asarray(scenario["block_bit_targets"], dtype=int)
    num_blocks = int(scenario["num_blocks"])
    K = int(uplinksystem.K)

    L_out = [int(num_blocks)] * K
    n_star = [[] for _ in range(K)]
    F_star = [[] for _ in range(K)]
    R_star = [[] for _ in range(K)]
    B_used_star = [[] for _ in range(K)]
    B_kl_star = [[] for _ in range(K)]
    target_bits_star = [[] for _ in range(K)]
    unserved_bits_star = [[] for _ in range(K)]
    skipped_blocks_per_user = [0 for _ in range(K)]
    norm_stats = []
    all_user_block_results = [[] for _ in range(K)]

    lambda_rate_0 = float(sim_cfg["initial_lambda_rate_constraint"])
    lambda_power_0 = float(sim_cfg["initial_lambda_power_constraint"])
    user_precoder_models: list[nn.Module] = []

    for k in range(K):
        print(f"\n================ FIXED-TARGET TRAIN USER {k} ================")

        while len(uplinksystem.H[k]) < num_blocks:
            uplinksystem.add_block(k)

        H_user = np.array(uplinksystem.H[k], dtype=np.complex64)
        mean = np.mean(H_user)
        var = np.mean(np.abs(H_user - mean) ** 2) + 1e-12
        norm_stats.append((mean, var))
        if channel_norm:
            uplinksystem.H[k] = list((H_user - mean) / np.sqrt(var))

        user_model = build_user_precoder_net_with_sigma_context(
            Nr=int(uplinksystem.NR[k]),
            Nt=int(uplinksystem.NT[k]),
            dk=int(uplinksystem.dk[k]),
            device=DEVICE,
        )
        user_optimizer = torch.optim.Adam(user_model.parameters(), lr=float(sim_cfg["lr_net"]))
        user_precoder_models.append(user_model)

        lambda_rate_user = float(lambda_rate_0)
        lambda_power_user = float(lambda_power_0)

        for ell in range(num_blocks):
            while len(uplinksystem.H[k]) <= ell:
                uplinksystem.add_block(k)

            target_bits = int(block_targets[k, ell])
            print(f"\n--- FIXED-TARGET User {k}, Block {ell}, target_bits={target_bits} ---")
            S, B_used, step_a_diagnostics = optimize_subblocklength_precoder(
                uplinksystem=uplinksystem,
                user=k,
                block=ell,
                B_rem=int(target_bits),
                lambda_rate_0=lambda_rate_user,
                lambda_power_0=lambda_power_user,
                sim_cfg=sim_cfg,
                precoder_net=user_model,
                optimizer=user_optimizer,
                interference_F_snapshot=interference_F_snapshot,
                bit_budget_role="fixed_block_targets",
            )

            target_bits_star[k].append(int(target_bits))

            if len(S) == 0:
                H_kl = np.asarray(uplinksystem.H[k][ell], dtype=np.complex64)
                F_zero = np.zeros((int(uplinksystem.NT[k]), int(uplinksystem.dk[k])), dtype=np.complex64)
                zero_result = {
                    "n_kl": int(uplinksystem.T[k]),
                    "n": int(uplinksystem.T[k]),
                    "B_l": 0,
                    "Bits per sub-block length B/n_kl": 0.0,
                    "required_R_fbl": float(target_bits) / float(max(int(uplinksystem.T[k]), 1)),
                    "achieved_R_fbl": float(step_a_diagnostics.get("achieved_R_fbl", 0.0)),
                    "F": torch.tensor(F_zero, dtype=torch.complex64),
                    "R_fbl": float(step_a_diagnostics.get("achieved_R_fbl", 0.0)),
                    "F_power": 0.0,
                    "lambda_rate": float(step_a_diagnostics.get("lambda_rate", lambda_rate_user)),
                    "lambda_power": float(step_a_diagnostics.get("lambda_power", lambda_power_user)),
                    "loss_curve": list(step_a_diagnostics.get("loss_curve", [])),
                    "kkt_history": list(step_a_diagnostics.get("kkt_history", [])),
                    "solve_status": step_a_diagnostics.get("solve_status", "unknown"),
                    "final_primal_residual": float(step_a_diagnostics.get("final_primal_residual", 0.0)),
                    "final_complementarity_residual": float(
                        step_a_diagnostics.get("final_complementarity_residual", 0.0)
                    ),
                    "target_bits": int(target_bits),
                    "unserved_bits": int(target_bits),
                    "skipped": True,
                }
                S = [zero_result]
                B_used = 0
                skipped_blocks_per_user[k] += 1

            all_user_block_results[k].append(S)
            best = S[-1]
            n_opt = int(best["n_kl"])
            F_opt = best["F"]
            R_opt = float(best["R_fbl"])
            unserved_bits = max(0, int(target_bits) - int(B_used))

            n_star[k].append(int(n_opt))
            F_star[k].append(F_opt)
            R_star[k].append(float(R_opt))
            B_used_star[k].append(int(B_used))
            B_kl_star[k].append(int(B_used))
            unserved_bits_star[k].append(int(unserved_bits))
            if int(B_used) <= 0 and len(S) > 0 and not bool(S[-1].get("skipped", False)):
                skipped_blocks_per_user[k] += 1

            if commit_live_precoders:
                uplinksystem.F[k][ell] = F_opt.detach().cpu().numpy()

            print(
                f">>> Fixed-target choice: n_kl={n_opt}, served_bits={int(B_used)}, "
                f"unserved_bits={int(unserved_bits)}"
            )

        uplinksystem.L[k] = int(num_blocks)
        uplinksystem.n_kl[k] = list(map(int, n_star[k]))

    post_training_data_dict = {
        "L_out": L_out,
        "n_star": n_star,
        "F_star": F_star,
        "R_star": R_star,
        "norm_stats": norm_stats,
        "all_user_block_results_train": all_user_block_results,
        "B_used_star": B_used_star,
        "B_kl_star": B_kl_star,
        "target_bits_star": target_bits_star,
        "unserved_bits_star": unserved_bits_star,
        "skipped_blocks_per_user": [int(v) for v in skipped_blocks_per_user],
        "scenario_mode": FIXED_BLOCK_TARGETS_MODE,
        "scenario_block_targets": block_targets.tolist(),
        "user_model_specs": export_user_model_specs(
            uplinksystem.NR,
            uplinksystem.NT,
            uplinksystem.dk,
            input_mode="channel_sigma_epsilon",
        ),
        "user_model_states": export_user_model_states(user_precoder_models),
        "precoder_parameterization": "shared_user_channel_sigma_epsilon_to_precoder_mlp",
    }
    return post_training_data_dict

import numpy as np
from scipy.stats import norm

def _Q_inv_np(eps: float) -> float:
    return float(norm.ppf(1.0 - float(eps)))

def _project_power_np(F: np.ndarray, P: float, eps: float = 1e-12) -> np.ndarray:
    p = np.linalg.norm(F, "fro") ** 2
    if p <= float(P):
        return F
    return F * (np.sqrt(P / (p + eps)) * (1.0 - float(POWER_PROJECTION_SAFETY_MARGIN)))


def _build_precoder_snapshot_from_models(
    uplinksystem,
    user_models: list[nn.Module],
) -> list[list[np.ndarray]]:
    snapshot: list[list[np.ndarray]] = []
    for k in range(int(uplinksystem.K)):
        user_blocks: list[np.ndarray] = []
        for l in range(len(uplinksystem.H[k])):
            user_blocks.append(
                infer_precoder_numpy_with_sigma_context(
                    user_models[k],
                    np.asarray(uplinksystem.H[k][l], dtype=np.complex64),
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

def _compute_R_fbl_np(
    H: np.ndarray,
    F: np.ndarray,
    sigma2: float,
    epsilon: float,
    n_kl: int,
    noise_plus_interference_cov: np.ndarray | None = None,
) -> float:
    Nr = H.shape[0]
    HF = H @ F

    if noise_plus_interference_cov is None:
        noise_cov = float(sigma2) * np.eye(Nr, dtype=np.complex128)
    else:
        noise_cov = np.asarray(noise_plus_interference_cov, dtype=np.complex128)

    chol = np.linalg.cholesky(noise_cov)
    G = np.linalg.solve(chol, HF)
    A = G @ G.conj().T
    A = 0.5 * (A + A.conj().T)

    I = np.eye(Nr, dtype=np.complex64)
    sign, logdet = np.linalg.slogdet(I + A)
    C = (logdet / np.log(2.0)).real

    eigvals = np.linalg.eigvalsh(A).real
    V = np.sum(eigvals * (eigvals + 2.0) / (eigvals + 1.0) ** 2) * (np.log2(np.e) ** 2)

    R = C - np.sqrt(V / float(n_kl)) * _Q_inv_np(epsilon)
    return float(np.real(R))

def dynamic_subblocklength_precoder_testing(
    uplinksystem,
    post_training_data_dict: dict,
    sim_cfg: dict,
    channel_norm: bool = True,
):
    """
    Testing version consistent with training_v2 block allocation, but WITHOUT precoder optimization:
      - Uses fixed trained precoders F_star (per user per block; fallback to last if needed)
      - Step A: at n=T, serve the feasible payload supported by the fixed F
      - Step B: if this is the tail block, reduce n_kl while keeping B_used fixed
      - Create additional blocks while B_rem > 0
      - Saves per-block trajectory S_test for plotting (same structure as training)

    Returns:
      test_data_dict with:
        - L_out_test, n_star_test, F_star_test, R_star_test
        - all_user_block_results_test: list[user][block] = S_test (trajectory over n_kl)
        - B_used_star_test, B_kl_star_test
        - norm_stats_used (the normalization stats applied)
    """
    K = int(uplinksystem.K)

    # from training
    F_star_train = post_training_data_dict["F_star"]
    norm_stats_train = post_training_data_dict["norm_stats"]
    user_model_specs = post_training_data_dict.get("user_model_specs")
    user_model_states = post_training_data_dict.get("user_model_states")
    user_models = None
    if user_model_specs is not None and user_model_states is not None:
        user_models = load_user_precoder_models(user_model_specs, user_model_states, device=DEVICE)

    # outputs
    L_out = [1] * K
    n_star = [[] for _ in range(K)]
    F_star = [[] for _ in range(K)]
    R_star = [[] for _ in range(K)]
    all_user_block_results = [[] for _ in range(K)]
    B_used_star = [[] for _ in range(K)]
    B_kl_star = [[] for _ in range(K)]

    # ---- apply same per-user normalization as training ----
    if channel_norm:
        for k in range(K):
            mean, var = norm_stats_train[k]
            H_user = np.array(uplinksystem.H[k], dtype=np.complex64)
            uplinksystem.H[k] = list((H_user - mean) / np.sqrt(var + 1e-12))

    for k in range(K):
        print(f"\n================ TEST USER {k} ================")

        P = float(uplinksystem.P[k])
        sigma2 = float(uplinksystem.sigma2[k])
        epsilon = float(uplinksystem.epsilon[k])
        T = int(uplinksystem.T[k])

        n_kl_min = int(sim_cfg["n_kl_min"])
        n_kl_step = int(sim_cfg["n_kl_step"])

        B_rem = int(uplinksystem.B[k])
        ell = 0

        while len(uplinksystem.H[k]) < 1:
            uplinksystem.add_block(k)

        while B_rem > 0:
            if ell >= len(uplinksystem.H[k]):
                uplinksystem.add_block(k)

            H_kl = np.array(uplinksystem.H[k][ell], dtype=np.complex64)
            if user_models is not None:
                shared_snapshot = _build_precoder_snapshot_from_models(uplinksystem, user_models)
                F_fix = np.asarray(shared_snapshot[k][ell], dtype=np.complex64)
                F_fix_t = torch.tensor(F_fix, dtype=torch.complex64)
                noise_plus_interference_cov = build_uplink_rate_covariance(
                    uplinksystem,
                    sim_cfg,
                    k,
                    ell,
                    F_override=shared_snapshot,
                )
            else:
                # ---- fallback for older saved training artifacts ----
                if k < len(F_star_train) and ell < len(F_star_train[k]):
                    F_fix_t = F_star_train[k][ell]
                elif k < len(F_star_train) and len(F_star_train[k]) > 0:
                    print("Number of sub-blocks has exceeded expected value, no fixed precoder from training found, resorting to precoder from last block")
                    F_fix_t = F_star_train[k][-1]
                elif len(F_star_train[k]) == 0:
                    print(f">>> STOP test user {k}: no trained precoder available.")
                    break

                F_fix = F_fix_t.detach().cpu().numpy().astype(np.complex64)
                F_fix = _project_power_np(F_fix, P)
                noise_plus_interference_cov = build_uplink_rate_covariance(
                    uplinksystem,
                    sim_cfg,
                    k,
                    ell,
                    F_override=F_star_train,
                )

            print(f"\n--- TEST User {k}, Block {ell}, B_rem={B_rem} ---")

            # ==========================================================
            # STEP A: evaluate the fixed beam at n=T and serve the
            # feasible payload directly, without retrying.
            # ==========================================================
            R_T = _compute_R_fbl_np(
                H_kl, F_fix, sigma2, epsilon, n_kl=T,
                noise_plus_interference_cov=noise_plus_interference_cov
            )
            B_max = max(int(np.floor(float(T) * float(R_T))), 0)
            B_used = int(min(B_rem, B_max))

            print(
                f"n=T={T}, requested_bits={B_rem}, feasible_bits={B_max}, "
                f"served_bits={B_used}, R_fbl={R_T}"
            )

            if B_used <= 0:
                print(f">>> STOP test user {k} at block {ell}: cannot make n=T feasible.")
                break

            # Build S trajectory like training: include n=T point + feasible decreasing n_kl points
            S_block = []

            # n=T point
            R_T = _compute_R_fbl_np(
                H_kl, F_fix, sigma2, epsilon, n_kl=T,
                noise_plus_interference_cov=noise_plus_interference_cov
            )
            S_block.append({
                "n_kl": int(T),
                "n": int(T),
                "B_l": int(B_used),
                "Bits per sub-block length B/n_kl": float(B_used) / float(T),
                "F": F_fix_t,  # keep torch tensor for consistency with training plots/SNR plots
                "R_fbl": float(R_T),
                "F_power": float(np.linalg.norm(F_fix, "fro") ** 2),
                "lambda_rate": 0.0,
                "lambda_power": 0.0,
                "loss_curve": [],
            })

            # ==========================================================
            # STEP B: decrease n_kl with B_used fixed (fixed F)
            # ==========================================================
            best_n = int(T)
            best_R = float(R_T)

            n_kl = T - n_kl_step
            while n_kl >= n_kl_min:
                if int(B_used) < int(B_rem):
                    print(f"Not reducing n_kl since this block only served a partial payload (n_kl = {n_kl})")
                    break
                R = _compute_R_fbl_np(
                    H_kl, F_fix, sigma2, epsilon, n_kl=n_kl,
                    noise_plus_interference_cov=noise_plus_interference_cov
                )
                rate_violation = (B_used / float(n_kl)) - R

                print(f"Test n_kl={n_kl}: R_fbl={R}, rate_violation={rate_violation}")

                if rate_violation <= 0.0:
                    best_n = int(n_kl)
                    best_R = float(R)

                    S_block.append({
                        "n_kl": int(n_kl),
                        "n": int(n_kl),
                        "B_l": int(B_used),
                        "Bits per sub-block length B/n_kl": float(B_used) / float(n_kl),
                        "F": F_fix_t,
                        "R_fbl": float(R),
                        "F_power": float(np.linalg.norm(F_fix, "fro") ** 2),
                        "lambda_rate": 0.0,
                        "lambda_power": 0.0,
                        "loss_curve": [],
                    })

                    n_kl -= n_kl_step
                else:
                    break

            print(f">>> Chosen block {ell}: n_kl={best_n}, B_used={B_used}, R_fbl={best_R}")

            # save trajectory for plotting
            all_user_block_results[k].append(S_block)

            # save chosen decisions
            n_star[k].append(best_n)
            F_star[k].append(F_fix_t)
            R_star[k].append(best_R)
            B_used_star[k].append(int(B_used))

            B_kl = min(B_rem, int(B_used))
            B_kl_star[k].append(int(B_kl))
            B_rem -= B_kl
            print(f">>> Transmitted B_kl={B_kl}, remaining B_rem={B_rem}")

            if B_rem > 0:
                ell += 1
                L_out[k] = ell + 1
            else:
                ell += 1
                L_out[k] = ell
                
        
        uplinksystem.L[k] = int(L_out[k])
        if len(n_star[k]) > 0:
            uplinksystem.n_kl[k] = list(n_star[k])

    # push F into uplinksystem (numpy)
    for k in range(K):
        if len(F_star[k]) > 0:
            uplinksystem.F[k] = np.array([F.detach().cpu().numpy() for F in F_star[k]])

    try:
        uplinksystem.update_system()
    except Exception:
        pass

    test_data_dict = {
        "L_out_test": L_out,
        "n_star_test": n_star,
        "F_star_test": F_star,
        "R_star_test": R_star,
        "all_user_block_results_test": all_user_block_results,
        "B_used_star_test": B_used_star,
        "B_kl_star_test": B_kl_star,
        "norm_stats_used": norm_stats_train,
        "user_model_specs": user_model_specs,
        "user_model_states": user_model_states,
        "precoder_parameterization": post_training_data_dict.get(
            "precoder_parameterization",
            "per_block_precoders" if user_models is None else "shared_user_channel_sigma_epsilon_to_precoder_mlp",
        ),
    }
    return test_data_dict

# ============================================================
# Usage
# ============================================================
if __name__ == "__main__":
    from config_loader import get_config
    from UplinkSystem import UplinkSystem

    SYSTEM_TEST_PARAMS, SIMULATION_TEST_PARAMS = get_config()
    print(SYSTEM_TEST_PARAMS)
    print(SIMULATION_TEST_PARAMS)
    uplinksystem = UplinkSystem(SYSTEM_TEST_PARAMS, seed=0)

    # expected keys in SIMULATION_TEST_PARAMS:
    #   initial_lambda_rate_constraint, initial_lambda_power_constraint,
    #   max_epochs, lr_net, lr_rate_constraint, lr_power_constraint,
    #   n_kl_min, n_kl_step
    sim_cfg = dict(SIMULATION_TEST_PARAMS)

    L_out, n_star, F_star, R_star, norm_stats = dynamic_subblocklength_precoder_training(
        uplinksystem=uplinksystem,
        sim_cfg=sim_cfg,
        channel_norm=True
    )

    print("\n================ FINAL ================")
    print("L:", L_out)
    print("n*:", n_star)
    print("R*:", R_star)
