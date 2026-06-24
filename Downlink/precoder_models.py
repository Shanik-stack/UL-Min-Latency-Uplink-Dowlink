from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def net_output_to_precoder(F_out: torch.Tensor, nb: int, dk: int) -> torch.Tensor:
    if F_out.dim() == 2:
        F_out = F_out.squeeze(0)
    F_reshaped = F_out.view(2, nb, dk)
    return (F_reshaped[0] + 1j * F_reshaped[1]).to(torch.complex64)


def project_precoder_power(Fmat: torch.Tensor, power_limit: float, eps: float = 1e-12) -> torch.Tensor:
    fro = torch.linalg.norm(Fmat, ord="fro").real
    if float(fro.detach().cpu()) <= float(eps):
        fallback = torch.zeros_like(Fmat)
        diag_dim = min(int(Fmat.shape[0]), int(Fmat.shape[1]))
        fallback[:diag_dim, :diag_dim] = torch.eye(diag_dim, dtype=Fmat.dtype, device=Fmat.device)
        fro = torch.linalg.norm(fallback, ord="fro").real
        return fallback * (
            torch.sqrt(torch.tensor(float(power_limit), device=Fmat.device, dtype=torch.float32)) / (fro + eps)
        ).to(Fmat.dtype)

    scale = torch.sqrt(torch.tensor(float(power_limit), device=Fmat.device, dtype=torch.float32)) / (fro + eps)
    return Fmat * scale.to(Fmat.dtype)


class ChannelToPrecoderNet(nn.Module):
    def __init__(self, nr: int, nb: int, dk: int):
        super().__init__()
        self.nr = int(nr)
        self.nb = int(nb)
        self.dk = int(dk)

        in_dim = 2 * self.nr * self.nb
        out_dim = 2 * self.nb * self.dk

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


class BlockContextAndInterferenceToPrecoderNet(nn.Module):
    def __init__(
        self,
        nr: int,
        nb: int,
        dk: int,
        *,
        k_count: int = 1,
        max_nr: int | None = None,
        max_nb: int | None = None,
    ):
        super().__init__()
        self.nr = int(nr)
        self.nb = int(nb)
        self.dk = int(dk)
        self.k_count = int(k_count)
        self.max_nr = int(max_nr if max_nr is not None else nr)
        self.max_nb = int(max_nb if max_nb is not None else nb)

        in_dim = (
            2 * self.k_count * self.max_nr * self.max_nb
            + self.k_count
            + 2 * self.max_nr * self.max_nr
            + 1
        )
        out_dim = 2 * self.nb * self.dk

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

    def _pad_complex_matrix(self, matrix: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
        padded = torch.zeros((rows, cols), dtype=torch.complex64, device=matrix.device)
        row_lim = min(int(matrix.shape[0]), int(rows))
        col_lim = min(int(matrix.shape[1]), int(cols))
        padded[:row_lim, :col_lim] = matrix[:row_lim, :col_lim].to(dtype=torch.complex64)
        return padded

    def forward(
        self,
        H_block: Sequence[torch.Tensor] | torch.Tensor,
        active_mask: Sequence[int | float] | torch.Tensor,
        noise_plus_interference_cov: torch.Tensor,
        epsilon: float,
    ) -> torch.Tensor:
        if isinstance(H_block, torch.Tensor):
            if H_block.dim() == 2:
                block_channels = [H_block]
            elif H_block.dim() == 3:
                block_channels = [H_block[idx] for idx in range(int(H_block.shape[0]))]
            else:
                raise ValueError(f"Unexpected H_block tensor rank: {H_block.dim()}")
        else:
            block_channels = [torch.as_tensor(H_kl, dtype=torch.complex64, device=DEVICE) for H_kl in H_block]

        if len(block_channels) == 0:
            raise ValueError("H_block must contain at least one user channel.")

        device = block_channels[0].device
        padded_channels = []
        for idx in range(self.k_count):
            if idx < len(block_channels):
                padded_channels.append(self._pad_complex_matrix(block_channels[idx].to(device=device), self.max_nr, self.max_nb))
            else:
                padded_channels.append(torch.zeros((self.max_nr, self.max_nb), dtype=torch.complex64, device=device))
        H_stack = torch.stack(padded_channels, dim=0)
        x_h = torch.cat([H_stack.real.reshape(1, -1), H_stack.imag.reshape(1, -1)], dim=1)

        if isinstance(active_mask, torch.Tensor):
            mask_vals = active_mask.to(device=device, dtype=torch.float32).reshape(1, -1)
        else:
            mask_vals = torch.tensor([list(active_mask)], dtype=torch.float32, device=device)
        if mask_vals.shape[1] < self.k_count:
            pad_width = self.k_count - int(mask_vals.shape[1])
            mask_vals = torch.cat([mask_vals, torch.zeros((1, pad_width), dtype=mask_vals.dtype, device=device)], dim=1)
        elif mask_vals.shape[1] > self.k_count:
            mask_vals = mask_vals[:, : self.k_count]

        noise_pad = self._pad_complex_matrix(
            noise_plus_interference_cov.to(device=device, dtype=torch.complex64),
            self.max_nr,
            self.max_nr,
        )
        x_noise = torch.cat([noise_pad.real.reshape(1, -1), noise_pad.imag.reshape(1, -1)], dim=1)

        x_meta = torch.tensor(
            [[float(epsilon)]],
            dtype=torch.float32,
            device=device,
        )
        return self.net(torch.cat([x_h, mask_vals, x_noise, x_meta], dim=1))


class ChannelAndBlocklengthToPrecoderNet(nn.Module):
    def __init__(
        self,
        nr: int,
        nb: int,
        dk: int,
        *,
        k_count: int = 1,
        max_nr: int | None = None,
        max_nb: int | None = None,
    ):
        super().__init__()
        self.nr = int(nr)
        self.nb = int(nb)
        self.dk = int(dk)
        self.k_count = int(k_count)
        self.max_nr = int(max_nr if max_nr is not None else nr)
        self.max_nb = int(max_nb if max_nb is not None else nb)

        in_dim = (
            2 * self.k_count * self.max_nr * self.max_nb
            + self.k_count
            + 2 * self.max_nr * self.max_nr
            + 2
        )
        out_dim = 2 * self.nb * self.dk

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

    def _pad_complex_matrix(self, matrix: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
        padded = torch.zeros((rows, cols), dtype=torch.complex64, device=matrix.device)
        row_lim = min(int(matrix.shape[0]), int(rows))
        col_lim = min(int(matrix.shape[1]), int(cols))
        padded[:row_lim, :col_lim] = matrix[:row_lim, :col_lim].to(dtype=torch.complex64)
        return padded

    def forward(
        self,
        H_block: Sequence[torch.Tensor] | torch.Tensor,
        n_kl: int | float,
        active_mask: Sequence[int | float] | torch.Tensor,
        noise_plus_interference_cov: torch.Tensor,
        epsilon: float,
    ) -> torch.Tensor:
        if isinstance(H_block, torch.Tensor):
            if H_block.dim() == 2:
                block_channels = [H_block]
            elif H_block.dim() == 3:
                block_channels = [H_block[idx] for idx in range(int(H_block.shape[0]))]
            else:
                raise ValueError(f"Unexpected H_block tensor rank: {H_block.dim()}")
        else:
            block_channels = [torch.as_tensor(H_kl, dtype=torch.complex64, device=DEVICE) for H_kl in H_block]

        if len(block_channels) == 0:
            raise ValueError("H_block must contain at least one user channel.")

        device = block_channels[0].device
        padded_channels = []
        for idx in range(self.k_count):
            if idx < len(block_channels):
                padded_channels.append(self._pad_complex_matrix(block_channels[idx].to(device=device), self.max_nr, self.max_nb))
            else:
                padded_channels.append(torch.zeros((self.max_nr, self.max_nb), dtype=torch.complex64, device=device))
        H_stack = torch.stack(padded_channels, dim=0)
        x_h = torch.cat([H_stack.real.reshape(1, -1), H_stack.imag.reshape(1, -1)], dim=1)

        if isinstance(active_mask, torch.Tensor):
            mask_vals = active_mask.to(device=device, dtype=torch.float32).reshape(1, -1)
        else:
            mask_vals = torch.tensor([list(active_mask)], dtype=torch.float32, device=device)
        if mask_vals.shape[1] < self.k_count:
            pad_width = self.k_count - int(mask_vals.shape[1])
            mask_vals = torch.cat([mask_vals, torch.zeros((1, pad_width), dtype=mask_vals.dtype, device=device)], dim=1)
        elif mask_vals.shape[1] > self.k_count:
            mask_vals = mask_vals[:, : self.k_count]

        noise_pad = self._pad_complex_matrix(
            noise_plus_interference_cov.to(device=device, dtype=torch.complex64),
            self.max_nr,
            self.max_nr,
        )
        x_noise = torch.cat([noise_pad.real.reshape(1, -1), noise_pad.imag.reshape(1, -1)], dim=1)

        n_safe = max(float(n_kl), 1.0)
        x_meta = torch.tensor(
            [[np.log1p(n_safe), float(epsilon)]],
            dtype=torch.float32,
            device=device,
        )
        return self.net(torch.cat([x_h, mask_vals, x_noise, x_meta], dim=1))


def build_user_precoder_net(nr: int, nb: int, dk: int, *, device: torch.device = DEVICE) -> ChannelToPrecoderNet:
    return ChannelToPrecoderNet(nr=nr, nb=nb, dk=dk).to(device)


def build_user_precoder_net_with_context(
    nr: int,
    nb: int,
    dk: int,
    *,
    k_count: int = 1,
    max_nr: int | None = None,
    max_nb: int | None = None,
    device: torch.device = DEVICE,
) -> BlockContextAndInterferenceToPrecoderNet:
    return BlockContextAndInterferenceToPrecoderNet(
        nr=nr,
        nb=nb,
        dk=dk,
        k_count=k_count,
        max_nr=max_nr,
        max_nb=max_nb,
    ).to(device)


def build_user_precoder_net_with_blocklength(
    nr: int,
    nb: int,
    dk: int,
    *,
    k_count: int = 1,
    max_nr: int | None = None,
    max_nb: int | None = None,
    device: torch.device = DEVICE,
) -> ChannelAndBlocklengthToPrecoderNet:
    return ChannelAndBlocklengthToPrecoderNet(
        nr=nr,
        nb=nb,
        dk=dk,
        k_count=k_count,
        max_nr=max_nr,
        max_nb=max_nb,
    ).to(device)


def infer_precoder_torch(
    model: nn.Module,
    H_kl: torch.Tensor,
    nb: int,
    dk: int,
    power_limit: float,
) -> torch.Tensor:
    F_out = model(H_kl)
    Fmat = net_output_to_precoder(F_out, nb, dk)
    return project_precoder_power(Fmat, power_limit)


def infer_precoder_torch_with_context(
    model: nn.Module,
    H_block: Sequence[torch.Tensor] | torch.Tensor,
    active_mask: Sequence[int | float] | torch.Tensor,
    noise_plus_interference_cov: torch.Tensor,
    epsilon: float,
    nb: int,
    dk: int,
    power_limit: float,
) -> torch.Tensor:
    F_out = model(
        H_block,
        active_mask,
        noise_plus_interference_cov,
        float(epsilon),
    )
    Fmat = net_output_to_precoder(F_out, nb, dk)
    return project_precoder_power(Fmat, power_limit)


def infer_precoder_torch_with_blocklength(
    model: nn.Module,
    H_block: Sequence[torch.Tensor] | torch.Tensor,
    n_kl: int,
    active_mask: Sequence[int | float] | torch.Tensor,
    noise_plus_interference_cov: torch.Tensor,
    epsilon: float,
    nb: int,
    dk: int,
    power_limit: float,
) -> torch.Tensor:
    F_out = model(
        H_block,
        int(n_kl),
        active_mask,
        noise_plus_interference_cov,
        float(epsilon),
    )
    Fmat = net_output_to_precoder(F_out, nb, dk)
    return project_precoder_power(Fmat, power_limit)


def infer_precoder_numpy(
    model: nn.Module,
    H_kl: np.ndarray,
    nb: int,
    dk: int,
    power_limit: float,
    *,
    device: torch.device = DEVICE,
) -> np.ndarray:
    with torch.no_grad():
        H_t = torch.tensor(np.asarray(H_kl), dtype=torch.complex64, device=device)
        F_t = infer_precoder_torch(model, H_t, nb, dk, power_limit)
    return F_t.detach().cpu().numpy().astype(np.complex128)


def infer_precoder_numpy_with_context(
    model: nn.Module,
    H_block: Sequence[np.ndarray] | np.ndarray,
    active_mask: Sequence[int | float] | np.ndarray,
    noise_plus_interference_cov: np.ndarray,
    epsilon: float,
    nb: int,
    dk: int,
    power_limit: float,
    *,
    device: torch.device = DEVICE,
) -> np.ndarray:
    with torch.no_grad():
        if isinstance(H_block, np.ndarray) and H_block.ndim == 2:
            H_t = torch.tensor(np.asarray(H_block), dtype=torch.complex64, device=device)
        else:
            H_t = [
                torch.tensor(np.asarray(H_kl), dtype=torch.complex64, device=device)
                for H_kl in list(H_block)
            ]
        mask_t = torch.tensor(np.asarray(active_mask), dtype=torch.float32, device=device)
        noise_cov_t = torch.tensor(np.asarray(noise_plus_interference_cov), dtype=torch.complex64, device=device)
        F_t = infer_precoder_torch_with_context(
            model,
            H_t,
            mask_t,
            noise_cov_t,
            epsilon,
            nb,
            dk,
            power_limit,
        )
    return F_t.detach().cpu().numpy().astype(np.complex128)


def infer_precoder_numpy_with_blocklength(
    model: nn.Module,
    H_block: Sequence[np.ndarray] | np.ndarray,
    n_kl: int,
    active_mask: Sequence[int | float] | np.ndarray,
    noise_plus_interference_cov: np.ndarray,
    epsilon: float,
    nb: int,
    dk: int,
    power_limit: float,
    *,
    device: torch.device = DEVICE,
) -> np.ndarray:
    with torch.no_grad():
        if isinstance(H_block, np.ndarray) and H_block.ndim == 2:
            H_t = torch.tensor(np.asarray(H_block), dtype=torch.complex64, device=device)
        else:
            H_t = [
                torch.tensor(np.asarray(H_kl), dtype=torch.complex64, device=device)
                for H_kl in list(H_block)
            ]
        mask_t = torch.tensor(np.asarray(active_mask), dtype=torch.float32, device=device)
        noise_cov_t = torch.tensor(np.asarray(noise_plus_interference_cov), dtype=torch.complex64, device=device)
        F_t = infer_precoder_torch_with_blocklength(
            model,
            H_t,
            n_kl,
            mask_t,
            noise_cov_t,
            epsilon,
            nb,
            dk,
            power_limit,
        )
    return F_t.detach().cpu().numpy().astype(np.complex128)


def export_user_model_specs(
    nr: Sequence[int],
    nb: Sequence[int],
    dk: Sequence[int],
    *,
    uses_blocklength_input: bool = False,
    input_mode: str | None = None,
    context_k: int | None = None,
    context_max_nr: int | None = None,
    context_max_nb: int | None = None,
) -> list[dict[str, int | bool]]:
    return [
        {
            "nr": int(nr[k]),
            "nb": int(nb[k]),
            "dk": int(dk[k]),
            "uses_blocklength_input": bool(uses_blocklength_input),
            "input_mode": str(
                input_mode
                if input_mode is not None
                else ("block_context_noise_epsilon_n" if uses_blocklength_input else "channel_only")
            ),
            "context_k": int(context_k if context_k is not None else len(nr)),
            "context_max_nr": int(context_max_nr if context_max_nr is not None else max(nr)),
            "context_max_nb": int(context_max_nb if context_max_nb is not None else max(nb)),
        }
        for k in range(len(nr))
    ]


def export_user_model_states(models: Sequence[nn.Module]) -> list[dict[str, Any]]:
    return [{key: value.detach().cpu() for key, value in model.state_dict().items()} for model in models]


def load_user_precoder_models(
    model_specs: Sequence[dict[str, Any]],
    model_states: Sequence[dict[str, Any]],
    *,
    device: torch.device = DEVICE,
) -> list[nn.Module]:
    models: list[nn.Module] = []
    for spec, state in zip(model_specs, model_states):
        input_mode = str(spec.get("input_mode", "")).strip().lower()
        if input_mode == "block_context_noise_epsilon":
            model = build_user_precoder_net_with_context(
                int(spec["nr"]),
                int(spec["nb"]),
                int(spec["dk"]),
                k_count=int(spec.get("context_k", 1)),
                max_nr=int(spec.get("context_max_nr", spec["nr"])),
                max_nb=int(spec.get("context_max_nb", spec["nb"])),
                device=device,
            )
        elif bool(spec.get("uses_blocklength_input", False)) or input_mode == "block_context_noise_epsilon_n":
            model = build_user_precoder_net_with_blocklength(
                int(spec["nr"]),
                int(spec["nb"]),
                int(spec["dk"]),
                k_count=int(spec.get("context_k", 1)),
                max_nr=int(spec.get("context_max_nr", spec["nr"])),
                max_nb=int(spec.get("context_max_nb", spec["nb"])),
                device=device,
            )
        else:
            model = build_user_precoder_net(
                int(spec["nr"]),
                int(spec["nb"]),
                int(spec["dk"]),
                device=device,
            )
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    return models
