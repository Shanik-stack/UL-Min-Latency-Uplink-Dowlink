from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def net_output_to_precoder(F_out: torch.Tensor, Nt: int, dk: int) -> torch.Tensor:
    if F_out.dim() == 2:
        F_out = F_out.squeeze(0)
    F_reshaped = F_out.view(2, Nt, dk)
    return (F_reshaped[0] + 1j * F_reshaped[1]).to(torch.complex64)


def project_precoder_power(Fmat: torch.Tensor, P: float, eps: float = 1e-12) -> torch.Tensor:
    fro = torch.linalg.norm(Fmat, ord="fro").real
    if float(fro.detach().cpu()) <= float(eps):
        fallback = torch.zeros_like(Fmat)
        diag_dim = min(int(Fmat.shape[0]), int(Fmat.shape[1]))
        fallback[:diag_dim, :diag_dim] = torch.eye(diag_dim, dtype=Fmat.dtype, device=Fmat.device)
        fro = torch.linalg.norm(fallback, ord="fro").real
        return fallback * (
            torch.sqrt(torch.tensor(float(P), device=Fmat.device, dtype=torch.float32)) / (fro + eps)
        ).to(Fmat.dtype)

    scale = torch.sqrt(torch.tensor(float(P), device=Fmat.device, dtype=torch.float32)) / (fro + eps)
    return Fmat * scale.to(Fmat.dtype)


class ChannelToPrecoderNet(nn.Module):
    """
    One user-specific model theta_k maps one channel block H_{k,l} to one precoder F_{k,l}.
    """

    def __init__(self, Nr: int, Nt: int, dk: int):
        super().__init__()
        self.Nr = int(Nr)
        self.Nt = int(Nt)
        self.dk = int(dk)

        in_dim = 2 * self.Nr * self.Nt
        out_dim = 2 * self.Nt * self.dk

        h1 = max(256, 8 * out_dim)
        h2 = max(128, 4 * out_dim)
        h3 = max(64, 2 * out_dim)

        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Linear(h3, out_dim),
        )

    def forward(self, H_kl: torch.Tensor) -> torch.Tensor:
        H_flat = H_kl.reshape(1, -1)
        x = torch.cat([H_flat.real, H_flat.imag], dim=1)
        return self.net(x)


class ChannelAndInterferenceToPrecoderNet(nn.Module):
    """
    One user-specific model theta_k maps one channel block together with one
    interference/noise context to one shared precoder F_{k,l}.
    """

    def __init__(self, Nr: int, Nt: int, dk: int):
        super().__init__()
        self.Nr = int(Nr)
        self.Nt = int(Nt)
        self.dk = int(dk)

        in_dim = 2 * self.Nr * self.Nt + 2 * self.Nr * self.Nr + 1
        out_dim = 2 * self.Nt * self.dk

        h1 = max(256, 8 * out_dim)
        h2 = max(128, 4 * out_dim)
        h3 = max(64, 2 * out_dim)

        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Linear(h3, out_dim),
        )

    def forward(
        self,
        H_kl: torch.Tensor,
        noise_plus_interference_cov: torch.Tensor,
        epsilon: float,
    ) -> torch.Tensor:
        H_flat = H_kl.reshape(1, -1)
        x_h = torch.cat([H_flat.real, H_flat.imag], dim=1)
        noise_flat = noise_plus_interference_cov.reshape(1, -1)
        x_noise = torch.cat([noise_flat.real, noise_flat.imag], dim=1)
        x_meta = torch.tensor(
            [[float(epsilon)]],
            dtype=x_h.dtype,
            device=H_kl.device,
        )
        x = torch.cat([x_h, x_noise, x_meta], dim=1)
        return self.net(x)


class ChannelAndBlocklengthToPrecoderNet(nn.Module):
    """
    One user-specific model theta_k maps one channel block and one candidate
    sub-blocklength to one precoder F_{k,l}(n).
    """

    def __init__(self, Nr: int, Nt: int, dk: int):
        super().__init__()
        self.Nr = int(Nr)
        self.Nt = int(Nt)
        self.dk = int(dk)

        in_dim = 2 * self.Nr * self.Nt + 2 * self.Nr * self.Nr + 2
        out_dim = 2 * self.Nt * self.dk

        h1 = max(256, 8 * out_dim)
        h2 = max(128, 4 * out_dim)
        h3 = max(64, 2 * out_dim)

        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Linear(h3, out_dim),
        )

    def forward(
        self,
        H_kl: torch.Tensor,
        n_kl: int | float,
        noise_plus_interference_cov: torch.Tensor,
        epsilon: float,
    ) -> torch.Tensor:
        H_flat = H_kl.reshape(1, -1)
        x_h = torch.cat([H_flat.real, H_flat.imag], dim=1)
        noise_flat = noise_plus_interference_cov.reshape(1, -1)
        x_noise = torch.cat([noise_flat.real, noise_flat.imag], dim=1)
        n_safe = max(float(n_kl), 1.0)
        x_meta = torch.tensor(
            [[np.log1p(n_safe), float(epsilon)]],
            dtype=x_h.dtype,
            device=H_kl.device,
        )
        x = torch.cat([x_h, x_noise, x_meta], dim=1)
        return self.net(x)


def build_user_precoder_net(Nr: int, Nt: int, dk: int, *, device: torch.device = DEVICE) -> ChannelToPrecoderNet:
    return ChannelToPrecoderNet(Nr=Nr, Nt=Nt, dk=dk).to(device)


def build_user_precoder_net_with_interference_context(
    Nr: int,
    Nt: int,
    dk: int,
    *,
    device: torch.device = DEVICE,
) -> ChannelAndInterferenceToPrecoderNet:
    return ChannelAndInterferenceToPrecoderNet(Nr=Nr, Nt=Nt, dk=dk).to(device)


def build_user_precoder_net_with_blocklength(
    Nr: int,
    Nt: int,
    dk: int,
    *,
    device: torch.device = DEVICE,
) -> ChannelAndBlocklengthToPrecoderNet:
    return ChannelAndBlocklengthToPrecoderNet(Nr=Nr, Nt=Nt, dk=dk).to(device)


def infer_precoder_torch(
    model: nn.Module,
    H_kl: torch.Tensor,
    Nt: int,
    dk: int,
    P: float,
) -> torch.Tensor:
    F_out = model(H_kl)
    Fmat = net_output_to_precoder(F_out, Nt, dk)
    return project_precoder_power(Fmat, P)


def infer_precoder_torch_with_interference_context(
    model: nn.Module,
    H_kl: torch.Tensor,
    noise_plus_interference_cov: torch.Tensor,
    epsilon: float,
    Nt: int,
    dk: int,
    P: float,
) -> torch.Tensor:
    F_out = model(H_kl, noise_plus_interference_cov, float(epsilon))
    Fmat = net_output_to_precoder(F_out, Nt, dk)
    return project_precoder_power(Fmat, P)


def infer_precoder_torch_with_blocklength(
    model: nn.Module,
    H_kl: torch.Tensor,
    n_kl: int,
    noise_plus_interference_cov: torch.Tensor,
    epsilon: float,
    Nt: int,
    dk: int,
    P: float,
) -> torch.Tensor:
    F_out = model(H_kl, int(n_kl), noise_plus_interference_cov, float(epsilon))
    Fmat = net_output_to_precoder(F_out, Nt, dk)
    return project_precoder_power(Fmat, P)


def infer_precoder_numpy(
    model: nn.Module,
    H_kl: np.ndarray,
    Nt: int,
    dk: int,
    P: float,
    *,
    device: torch.device = DEVICE,
) -> np.ndarray:
    with torch.no_grad():
        H_t = torch.tensor(np.asarray(H_kl), dtype=torch.complex64, device=device)
        F_t = infer_precoder_torch(model, H_t, Nt, dk, P)
    return F_t.detach().cpu().numpy().astype(np.complex128)


def infer_precoder_numpy_with_interference_context(
    model: nn.Module,
    H_kl: np.ndarray,
    noise_plus_interference_cov: np.ndarray,
    epsilon: float,
    Nt: int,
    dk: int,
    P: float,
    *,
    device: torch.device = DEVICE,
) -> np.ndarray:
    with torch.no_grad():
        H_t = torch.tensor(np.asarray(H_kl), dtype=torch.complex64, device=device)
        noise_cov_t = torch.tensor(np.asarray(noise_plus_interference_cov), dtype=torch.complex64, device=device)
        F_t = infer_precoder_torch_with_interference_context(
            model,
            H_t,
            noise_cov_t,
            epsilon,
            Nt,
            dk,
            P,
        )
    return F_t.detach().cpu().numpy().astype(np.complex128)


def infer_precoder_numpy_with_blocklength(
    model: nn.Module,
    H_kl: np.ndarray,
    n_kl: int,
    noise_plus_interference_cov: np.ndarray,
    epsilon: float,
    Nt: int,
    dk: int,
    P: float,
    *,
    device: torch.device = DEVICE,
) -> np.ndarray:
    with torch.no_grad():
        H_t = torch.tensor(np.asarray(H_kl), dtype=torch.complex64, device=device)
        noise_cov_t = torch.tensor(np.asarray(noise_plus_interference_cov), dtype=torch.complex64, device=device)
        F_t = infer_precoder_torch_with_blocklength(model, H_t, n_kl, noise_cov_t, epsilon, Nt, dk, P)
    return F_t.detach().cpu().numpy().astype(np.complex128)


def export_user_model_specs(
    NR: Sequence[int],
    NT: Sequence[int],
    dk: Sequence[int],
    *,
    uses_blocklength_input: bool = False,
    input_mode: str | None = None,
) -> list[dict[str, int | bool]]:
    return [
        {
            "Nr": int(NR[k]),
            "Nt": int(NT[k]),
            "dk": int(dk[k]),
            "uses_blocklength_input": bool(uses_blocklength_input),
            "input_mode": str(
                input_mode
                if input_mode is not None
                else ("channel_noise_epsilon_n" if uses_blocklength_input else "channel_only")
            ),
        }
        for k in range(len(NR))
    ]


def export_user_model_states(models: Sequence[nn.Module]) -> list[dict[str, Any]]:
    return [
        {key: value.detach().cpu() for key, value in model.state_dict().items()}
        for model in models
    ]


def load_user_precoder_models(
    model_specs: Sequence[dict[str, int]],
    model_states: Sequence[dict[str, Any]],
    *,
    device: torch.device = DEVICE,
) -> list[ChannelToPrecoderNet]:
    models: list[ChannelToPrecoderNet] = []
    for spec, state in zip(model_specs, model_states):
        input_mode = str(spec.get("input_mode", "")).strip().lower()
        if input_mode == "channel_noise_epsilon":
            model = build_user_precoder_net_with_interference_context(
                spec["Nr"],
                spec["Nt"],
                spec["dk"],
                device=device,
            )
        elif bool(spec.get("uses_blocklength_input", False)) or input_mode == "channel_noise_epsilon_n":
            model = build_user_precoder_net_with_blocklength(spec["Nr"], spec["Nt"], spec["dk"], device=device)
        else:
            model = build_user_precoder_net(spec["Nr"], spec["Nt"], spec["dk"], device=device)
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    return models
