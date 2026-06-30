from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
POWER_PROJECTION_SAFETY_MARGIN = 1e-6
DOWLINK_PRECODER_NET_SCOPES = {"per_user_nets", "bs_shared_net"}


def resolve_downlink_precoder_net_scope(scope: str | None) -> str:
    scope_key = str(scope or "per_user_nets").strip().lower()
    if scope_key not in DOWLINK_PRECODER_NET_SCOPES:
        known = ", ".join(sorted(DOWLINK_PRECODER_NET_SCOPES))
        raise ValueError(
            f"Unknown downlink precoder-net scope '{scope_key}'. Expected one of: {known}"
        )
    return scope_key


def net_output_to_precoder(
    F_out: torch.Tensor,
    nb: int,
    dk: int,
    *,
    output_nb: int | None = None,
    output_dk: int | None = None,
) -> torch.Tensor:
    model_nb = int(output_nb if output_nb is not None else nb)
    model_dk = int(output_dk if output_dk is not None else dk)
    if F_out.dim() == 2:
        F_out = F_out.squeeze(0)
    F_reshaped = F_out.view(2, model_nb, model_dk)
    return (F_reshaped[0, :nb, :dk] + 1j * F_reshaped[1, :nb, :dk]).to(torch.complex64)


def model_outputs_full_bs_precoder(model: nn.Module) -> bool:
    return bool(getattr(model, "outputs_full_bs_precoder", False))


def _pad_complex_matrix(matrix: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    padded = torch.zeros((rows, cols), dtype=torch.complex64, device=matrix.device)
    row_lim = min(int(matrix.shape[0]), int(rows))
    col_lim = min(int(matrix.shape[1]), int(cols))
    padded[:row_lim, :col_lim] = matrix[:row_lim, :col_lim].to(dtype=torch.complex64)
    return padded


def _normalize_block_channels(
    H_block: Sequence[torch.Tensor] | torch.Tensor,
    *,
    k_count: int,
    max_nr: int,
    max_nb: int,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(H_block, torch.Tensor):
        if H_block.dim() == 2:
            block_channels = [H_block]
        elif H_block.dim() == 3:
            block_channels = [H_block[idx] for idx in range(int(H_block.shape[0]))]
        else:
            raise ValueError(f"Unexpected H_block tensor rank: {H_block.dim()}")
    else:
        block_channels = [torch.as_tensor(H_kl, dtype=torch.complex64, device=device) for H_kl in H_block]

    if len(block_channels) == 0:
        raise ValueError("H_block must contain at least one user channel.")

    padded_channels = []
    for idx in range(int(k_count)):
        if idx < len(block_channels):
            padded_channels.append(
                _pad_complex_matrix(
                    block_channels[idx].to(device=device, dtype=torch.complex64),
                    int(max_nr),
                    int(max_nb),
                )
            )
        else:
            padded_channels.append(
                torch.zeros((int(max_nr), int(max_nb)), dtype=torch.complex64, device=device)
            )
    return torch.stack(padded_channels, dim=0)


def _normalize_float_vector(
    values: Sequence[int | float] | torch.Tensor,
    *,
    length: int,
    device: torch.device,
    transform=None,
) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        vector = values.to(device=device, dtype=torch.float32).reshape(1, -1)
    else:
        vector = torch.tensor([list(values)], dtype=torch.float32, device=device)
    if int(vector.shape[1]) < int(length):
        pad_width = int(length) - int(vector.shape[1])
        vector = torch.cat(
            [vector, torch.zeros((1, pad_width), dtype=vector.dtype, device=device)],
            dim=1,
        )
    elif int(vector.shape[1]) > int(length):
        vector = vector[:, : int(length)]
    if transform is not None:
        vector = transform(vector)
    return vector


def split_full_bs_precoder_torch(
    F_out: torch.Tensor,
    nb: Sequence[int],
    dk: Sequence[int],
    *,
    output_nb: int,
    output_slot_dk: int,
) -> list[torch.Tensor]:
    if F_out.dim() == 2:
        F_out = F_out.squeeze(0)
    total_slots = int(len(dk))
    F_reshaped = F_out.view(2, int(output_nb), int(total_slots) * int(output_slot_dk))
    complex_full = (F_reshaped[0] + 1j * F_reshaped[1]).to(torch.complex64)
    beams: list[torch.Tensor] = []
    for k in range(total_slots):
        start = int(k) * int(output_slot_dk)
        stop = start + int(output_slot_dk)
        user_slot = complex_full[: int(nb[k]), start:stop]
        beams.append(user_slot[:, : int(dk[k])].to(torch.complex64))
    return beams


def project_precoder_power(Fmat: torch.Tensor, power_limit: float, eps: float = 1e-12) -> torch.Tensor:
    fro = torch.linalg.norm(Fmat, ord="fro").real
    if float(fro.detach().cpu()) <= float(eps):
        fallback = torch.zeros_like(Fmat)
        diag_dim = min(int(Fmat.shape[0]), int(Fmat.shape[1]))
        fallback[:diag_dim, :diag_dim] = torch.eye(diag_dim, dtype=Fmat.dtype, device=Fmat.device)
        fro = torch.linalg.norm(fallback, ord="fro").real
        return fallback * (
            (
                torch.sqrt(torch.tensor(float(power_limit), device=Fmat.device, dtype=torch.float32)) / (fro + eps)
            ) * (1.0 - float(POWER_PROJECTION_SAFETY_MARGIN))
        ).to(Fmat.dtype)

    scale = (
        torch.sqrt(torch.tensor(float(power_limit), device=Fmat.device, dtype=torch.float32)) / (fro + eps)
    ) * (1.0 - float(POWER_PROJECTION_SAFETY_MARGIN))
    return Fmat * scale.to(Fmat.dtype)


def infer_raw_precoder_torch(
    model: nn.Module,
    H_kl: torch.Tensor,
    nb: int,
    dk: int,
    *,
    user_index: int | float | None = None,
) -> torch.Tensor:
    F_out = model(H_kl, user_index=user_index)
    return net_output_to_precoder(
        F_out,
        nb,
        dk,
        output_nb=int(getattr(model, "output_nb", nb)),
        output_dk=int(getattr(model, "output_dk", dk)),
    )


def infer_raw_precoder_torch_with_context(
    model: nn.Module,
    H_block: Sequence[torch.Tensor] | torch.Tensor,
    active_mask: Sequence[int | float] | torch.Tensor,
    noise_plus_interference_cov: torch.Tensor,
    epsilon: float,
    nb: int,
    dk: int,
    *,
    user_index: int | float | None = None,
) -> torch.Tensor:
    F_out = model(
        H_block,
        active_mask,
        noise_plus_interference_cov,
        float(epsilon),
        user_index=user_index,
    )
    return net_output_to_precoder(
        F_out,
        nb,
        dk,
        output_nb=int(getattr(model, "output_nb", nb)),
        output_dk=int(getattr(model, "output_dk", dk)),
    )


def infer_raw_precoder_torch_with_blocklength(
    model: nn.Module,
    H_block: Sequence[torch.Tensor] | torch.Tensor,
    n_kl: int,
    active_mask: Sequence[int | float] | torch.Tensor,
    noise_plus_interference_cov: torch.Tensor,
    epsilon: float,
    nb: int,
    dk: int,
    *,
    user_index: int | float | None = None,
) -> torch.Tensor:
    F_out = model(
        H_block,
        int(n_kl),
        active_mask,
        noise_plus_interference_cov,
        float(epsilon),
        user_index=user_index,
    )
    return net_output_to_precoder(
        F_out,
        nb,
        dk,
        output_nb=int(getattr(model, "output_nb", nb)),
        output_dk=int(getattr(model, "output_dk", dk)),
    )


def infer_raw_bs_precoders_torch(
    model: nn.Module,
    H_block: Sequence[torch.Tensor] | torch.Tensor,
    active_mask: Sequence[int | float] | torch.Tensor,
    nb: Sequence[int],
    dk: Sequence[int],
) -> list[torch.Tensor]:
    if not model_outputs_full_bs_precoder(model):
        raise ValueError("infer_raw_bs_precoders_torch requires a full-BS-output model.")
    F_out = model(H_block, active_mask)
    return split_full_bs_precoder_torch(
        F_out,
        nb,
        dk,
        output_nb=int(getattr(model, "output_nb", max(nb) if len(nb) > 0 else 0)),
        output_slot_dk=int(getattr(model, "output_dk", max(dk) if len(dk) > 0 else 0)),
    )


def infer_raw_bs_precoders_torch_with_blocklength(
    model: nn.Module,
    H_block: Sequence[torch.Tensor] | torch.Tensor,
    n_targets: Sequence[int] | torch.Tensor,
    active_mask: Sequence[int | float] | torch.Tensor,
    sigma2: Sequence[float] | torch.Tensor,
    epsilon: Sequence[float] | torch.Tensor,
    nb: Sequence[int],
    dk: Sequence[int],
) -> list[torch.Tensor]:
    if not model_outputs_full_bs_precoder(model):
        raise ValueError("infer_raw_bs_precoders_torch_with_blocklength requires a full-BS-output model.")
    F_out = model(H_block, n_targets, active_mask, sigma2, epsilon)
    return split_full_bs_precoder_torch(
        F_out,
        nb,
        dk,
        output_nb=int(getattr(model, "output_nb", max(nb) if len(nb) > 0 else 0)),
        output_slot_dk=int(getattr(model, "output_dk", max(dk) if len(dk) > 0 else 0)),
    )


class ChannelToPrecoderNet(nn.Module):
    def __init__(
        self,
        nr: int,
        nb: int,
        dk: int,
        *,
        max_nr: int | None = None,
        max_nb: int | None = None,
        max_dk: int | None = None,
        include_user_index: bool = False,
        user_count: int = 1,
    ):
        super().__init__()
        self.nr = int(nr)
        self.nb = int(nb)
        self.dk = int(dk)
        self.max_nr = int(max_nr if max_nr is not None else nr)
        self.max_nb = int(max_nb if max_nb is not None else nb)
        self.max_dk = int(max_dk if max_dk is not None else dk)
        self.output_nb = int(self.max_nb)
        self.output_dk = int(self.max_dk)
        self.include_user_index = bool(include_user_index)
        self.user_count = max(1, int(user_count))

        in_dim = 2 * self.max_nr * self.max_nb + (1 if self.include_user_index else 0)
        out_dim = 2 * self.output_nb * self.output_dk

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

    def forward(self, H_kl: torch.Tensor, user_index: int | float | None = None) -> torch.Tensor:
        H_pad = self._pad_complex_matrix(H_kl.to(dtype=torch.complex64), self.max_nr, self.max_nb)
        H_flat = H_pad.reshape(1, -1)
        x = torch.cat([H_flat.real, H_flat.imag], dim=1)
        if self.include_user_index:
            denom = float(max(self.user_count - 1, 1))
            x = torch.cat(
                [
                    x,
                    torch.tensor(
                        [[(0.0 if user_index is None else float(user_index)) / denom]],
                        dtype=torch.float32,
                        device=H_kl.device,
                    ),
                ],
                dim=1,
            )
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
        max_dk: int | None = None,
        include_user_index: bool = False,
    ):
        super().__init__()
        self.nr = int(nr)
        self.nb = int(nb)
        self.dk = int(dk)
        self.k_count = int(k_count)
        self.max_nr = int(max_nr if max_nr is not None else nr)
        self.max_nb = int(max_nb if max_nb is not None else nb)
        self.max_dk = int(max_dk if max_dk is not None else dk)
        self.output_nb = int(self.max_nb)
        self.output_dk = int(self.max_dk)
        self.include_user_index = bool(include_user_index)

        in_dim = (
            2 * self.k_count * self.max_nr * self.max_nb
            + self.k_count
            + 2 * self.max_nr * self.max_nr
            + 1
            + (1 if self.include_user_index else 0)
        )
        out_dim = 2 * self.output_nb * self.output_dk

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
        user_index: int | float | None = None,
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
        if self.include_user_index:
            denom = float(max(self.k_count - 1, 1))
            x_meta = torch.cat(
                [
                    x_meta,
                    torch.tensor(
                        [[(0.0 if user_index is None else float(user_index)) / denom]],
                        dtype=torch.float32,
                        device=device,
                    ),
                ],
                dim=1,
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
        max_dk: int | None = None,
        include_user_index: bool = False,
    ):
        super().__init__()
        self.nr = int(nr)
        self.nb = int(nb)
        self.dk = int(dk)
        self.k_count = int(k_count)
        self.max_nr = int(max_nr if max_nr is not None else nr)
        self.max_nb = int(max_nb if max_nb is not None else nb)
        self.max_dk = int(max_dk if max_dk is not None else dk)
        self.output_nb = int(self.max_nb)
        self.output_dk = int(self.max_dk)
        self.include_user_index = bool(include_user_index)

        in_dim = (
            2 * self.k_count * self.max_nr * self.max_nb
            + self.k_count
            + 2 * self.max_nr * self.max_nr
            + 2
            + (1 if self.include_user_index else 0)
        )
        out_dim = 2 * self.output_nb * self.output_dk

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
        user_index: int | float | None = None,
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
        if self.include_user_index:
            denom = float(max(self.k_count - 1, 1))
            x_meta = torch.cat(
                [
                    x_meta,
                    torch.tensor(
                        [[(0.0 if user_index is None else float(user_index)) / denom]],
                        dtype=torch.float32,
                        device=device,
                    ),
                ],
                dim=1,
            )
        return self.net(torch.cat([x_h, mask_vals, x_noise, x_meta], dim=1))


class FullBlockToBsPrecoderNet(nn.Module):
    def __init__(
        self,
        *,
        k_count: int,
        max_nr: int,
        max_nb: int,
        max_dk: int,
    ):
        super().__init__()
        self.k_count = int(k_count)
        self.max_nr = int(max_nr)
        self.max_nb = int(max_nb)
        self.max_dk = int(max_dk)
        self.output_nb = int(self.max_nb)
        self.output_dk = int(self.max_dk)
        self.output_total_dk = int(self.k_count * self.max_dk)
        self.outputs_full_bs_precoder = True

        in_dim = 2 * self.k_count * self.max_nr * self.max_nb + self.k_count
        out_dim = 2 * self.output_nb * self.output_total_dk

        h1 = max(512, 4 * out_dim)
        h2 = max(256, 2 * out_dim)
        h3 = max(128, out_dim)

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
        H_block: Sequence[torch.Tensor] | torch.Tensor,
        active_mask: Sequence[int | float] | torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(H_block, torch.Tensor):
            device = H_block.device
        else:
            first = next(iter(H_block), None)
            device = first.device if isinstance(first, torch.Tensor) else DEVICE
        H_stack = _normalize_block_channels(
            H_block,
            k_count=self.k_count,
            max_nr=self.max_nr,
            max_nb=self.max_nb,
            device=device,
        )
        x_h = torch.cat([H_stack.real.reshape(1, -1), H_stack.imag.reshape(1, -1)], dim=1)
        x_mask = _normalize_float_vector(active_mask, length=self.k_count, device=device)
        return self.net(torch.cat([x_h, x_mask], dim=1))


class FullBlockAndBlocklengthToBsPrecoderNet(nn.Module):
    def __init__(
        self,
        *,
        k_count: int,
        max_nr: int,
        max_nb: int,
        max_dk: int,
    ):
        super().__init__()
        self.k_count = int(k_count)
        self.max_nr = int(max_nr)
        self.max_nb = int(max_nb)
        self.max_dk = int(max_dk)
        self.output_nb = int(self.max_nb)
        self.output_dk = int(self.max_dk)
        self.output_total_dk = int(self.k_count * self.max_dk)
        self.outputs_full_bs_precoder = True

        in_dim = 2 * self.k_count * self.max_nr * self.max_nb + 4 * self.k_count
        out_dim = 2 * self.output_nb * self.output_total_dk

        h1 = max(512, 4 * out_dim)
        h2 = max(256, 2 * out_dim)
        h3 = max(128, out_dim)

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
        H_block: Sequence[torch.Tensor] | torch.Tensor,
        n_targets: Sequence[int] | torch.Tensor,
        active_mask: Sequence[int | float] | torch.Tensor,
        sigma2: Sequence[float] | torch.Tensor,
        epsilon: Sequence[float] | torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(H_block, torch.Tensor):
            device = H_block.device
        else:
            first = next(iter(H_block), None)
            device = first.device if isinstance(first, torch.Tensor) else DEVICE
        H_stack = _normalize_block_channels(
            H_block,
            k_count=self.k_count,
            max_nr=self.max_nr,
            max_nb=self.max_nb,
            device=device,
        )
        x_h = torch.cat([H_stack.real.reshape(1, -1), H_stack.imag.reshape(1, -1)], dim=1)
        x_mask = _normalize_float_vector(active_mask, length=self.k_count, device=device)
        x_n = _normalize_float_vector(
            n_targets,
            length=self.k_count,
            device=device,
            transform=lambda value: torch.log1p(torch.clamp(value, min=1.0)),
        )
        x_sigma2 = _normalize_float_vector(
            sigma2,
            length=self.k_count,
            device=device,
            transform=lambda value: torch.log1p(torch.clamp(value, min=0.0)),
        )
        x_eps = _normalize_float_vector(epsilon, length=self.k_count, device=device)
        return self.net(torch.cat([x_h, x_mask, x_n, x_sigma2, x_eps], dim=1))


def build_user_precoder_net(nr: int, nb: int, dk: int, *, device: torch.device = DEVICE) -> ChannelToPrecoderNet:
    return ChannelToPrecoderNet(
        nr=nr,
        nb=nb,
        dk=dk,
        max_nr=nr,
        max_nb=nb,
        max_dk=dk,
        include_user_index=False,
        user_count=1,
    ).to(device)


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
        max_dk=dk,
        include_user_index=False,
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
        max_dk=dk,
        include_user_index=False,
    ).to(device)


def build_shared_bs_precoder_net(
    *,
    k_count: int,
    max_nr: int,
    max_nb: int,
    max_dk: int,
    device: torch.device = DEVICE,
) -> FullBlockToBsPrecoderNet:
    return FullBlockToBsPrecoderNet(
        k_count=k_count,
        max_nr=max_nr,
        max_nb=max_nb,
        max_dk=max_dk,
    ).to(device)


def build_shared_bs_precoder_net_with_context(
    *,
    k_count: int,
    max_nr: int,
    max_nb: int,
    max_dk: int,
    device: torch.device = DEVICE,
) -> BlockContextAndInterferenceToPrecoderNet:
    return BlockContextAndInterferenceToPrecoderNet(
        nr=max_nr,
        nb=max_nb,
        dk=max_dk,
        k_count=k_count,
        max_nr=max_nr,
        max_nb=max_nb,
        max_dk=max_dk,
        include_user_index=True,
    ).to(device)


def build_shared_bs_precoder_net_with_blocklength(
    *,
    k_count: int,
    max_nr: int,
    max_nb: int,
    max_dk: int,
    device: torch.device = DEVICE,
) -> FullBlockAndBlocklengthToBsPrecoderNet:
    return FullBlockAndBlocklengthToBsPrecoderNet(
        k_count=k_count,
        max_nr=max_nr,
        max_nb=max_nb,
        max_dk=max_dk,
    ).to(device)


def infer_precoder_torch(
    model: nn.Module,
    H_kl: torch.Tensor,
    nb: int,
    dk: int,
    power_limit: float,
    *,
    user_index: int | float | None = None,
) -> torch.Tensor:
    Fmat = infer_raw_precoder_torch(model, H_kl, nb, dk, user_index=user_index)
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
    *,
    user_index: int | float | None = None,
) -> torch.Tensor:
    Fmat = infer_raw_precoder_torch_with_context(
        model,
        H_block,
        active_mask,
        noise_plus_interference_cov,
        epsilon,
        nb,
        dk,
        user_index=user_index,
    )
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
    *,
    user_index: int | float | None = None,
) -> torch.Tensor:
    Fmat = infer_raw_precoder_torch_with_blocklength(
        model,
        H_block,
        n_kl,
        active_mask,
        noise_plus_interference_cov,
        epsilon,
        nb,
        dk,
        user_index=user_index,
    )
    return project_precoder_power(Fmat, power_limit)


def infer_raw_precoder_numpy(
    model: nn.Module,
    H_kl: np.ndarray,
    nb: int,
    dk: int,
    *,
    device: torch.device = DEVICE,
    user_index: int | float | None = None,
) -> np.ndarray:
    with torch.no_grad():
        H_t = torch.tensor(np.asarray(H_kl), dtype=torch.complex64, device=device)
        F_t = infer_raw_precoder_torch(model, H_t, nb, dk, user_index=user_index)
    return F_t.detach().cpu().numpy().astype(np.complex128)


def infer_raw_precoder_numpy_with_context(
    model: nn.Module,
    H_block: Sequence[np.ndarray] | np.ndarray,
    active_mask: Sequence[int | float] | np.ndarray,
    noise_plus_interference_cov: np.ndarray,
    epsilon: float,
    nb: int,
    dk: int,
    *,
    device: torch.device = DEVICE,
    user_index: int | float | None = None,
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
        F_t = infer_raw_precoder_torch_with_context(
            model,
            H_t,
            mask_t,
            noise_cov_t,
            epsilon,
            nb,
            dk,
            user_index=user_index,
        )
    return F_t.detach().cpu().numpy().astype(np.complex128)


def infer_raw_precoder_numpy_with_blocklength(
    model: nn.Module,
    H_block: Sequence[np.ndarray] | np.ndarray,
    n_kl: int,
    active_mask: Sequence[int | float] | np.ndarray,
    noise_plus_interference_cov: np.ndarray,
    epsilon: float,
    nb: int,
    dk: int,
    *,
    device: torch.device = DEVICE,
    user_index: int | float | None = None,
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
        F_t = infer_raw_precoder_torch_with_blocklength(
            model,
            H_t,
            n_kl,
            mask_t,
            noise_cov_t,
            epsilon,
            nb,
            dk,
            user_index=user_index,
        )
    return F_t.detach().cpu().numpy().astype(np.complex128)


def infer_raw_bs_precoders_numpy(
    model: nn.Module,
    H_block: Sequence[np.ndarray] | np.ndarray,
    active_mask: Sequence[int | float] | np.ndarray,
    nb: Sequence[int],
    dk: Sequence[int],
    *,
    device: torch.device = DEVICE,
) -> list[np.ndarray]:
    with torch.no_grad():
        if isinstance(H_block, np.ndarray) and H_block.ndim == 2:
            H_t = torch.tensor(np.asarray(H_block), dtype=torch.complex64, device=device)
        else:
            H_t = [torch.tensor(np.asarray(H_kl), dtype=torch.complex64, device=device) for H_kl in list(H_block)]
        mask_t = torch.tensor(np.asarray(active_mask), dtype=torch.float32, device=device)
        F_list = infer_raw_bs_precoders_torch(model, H_t, mask_t, nb, dk)
    return [F_t.detach().cpu().numpy().astype(np.complex128) for F_t in F_list]


def infer_raw_bs_precoders_numpy_with_blocklength(
    model: nn.Module,
    H_block: Sequence[np.ndarray] | np.ndarray,
    n_targets: Sequence[int] | np.ndarray,
    active_mask: Sequence[int | float] | np.ndarray,
    sigma2: Sequence[float] | np.ndarray,
    epsilon: Sequence[float] | np.ndarray,
    nb: Sequence[int],
    dk: Sequence[int],
    *,
    device: torch.device = DEVICE,
) -> list[np.ndarray]:
    with torch.no_grad():
        if isinstance(H_block, np.ndarray) and H_block.ndim == 2:
            H_t = torch.tensor(np.asarray(H_block), dtype=torch.complex64, device=device)
        else:
            H_t = [torch.tensor(np.asarray(H_kl), dtype=torch.complex64, device=device) for H_kl in list(H_block)]
        n_t = torch.tensor(np.asarray(n_targets), dtype=torch.float32, device=device)
        mask_t = torch.tensor(np.asarray(active_mask), dtype=torch.float32, device=device)
        sigma2_t = torch.tensor(np.asarray(sigma2), dtype=torch.float32, device=device)
        epsilon_t = torch.tensor(np.asarray(epsilon), dtype=torch.float32, device=device)
        F_list = infer_raw_bs_precoders_torch_with_blocklength(
            model,
            H_t,
            n_t,
            mask_t,
            sigma2_t,
            epsilon_t,
            nb,
            dk,
        )
    return [F_t.detach().cpu().numpy().astype(np.complex128) for F_t in F_list]


def infer_precoder_numpy(
    model: nn.Module,
    H_kl: np.ndarray,
    nb: int,
    dk: int,
    power_limit: float,
    *,
    device: torch.device = DEVICE,
    user_index: int | float | None = None,
) -> np.ndarray:
    with torch.no_grad():
        H_t = torch.tensor(np.asarray(H_kl), dtype=torch.complex64, device=device)
        F_t = infer_precoder_torch(model, H_t, nb, dk, power_limit, user_index=user_index)
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
    user_index: int | float | None = None,
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
            user_index=user_index,
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
    user_index: int | float | None = None,
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
            user_index=user_index,
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
    model_scope: str = "per_user_nets",
    context_max_dk: int | None = None,
) -> list[dict[str, int | bool]]:
    scope = resolve_downlink_precoder_net_scope(model_scope)
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
            "context_max_dk": int(context_max_dk if context_max_dk is not None else max(dk)),
            "model_scope": str(scope),
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
    if len(model_specs) == 0:
        return []

    first_scope = resolve_downlink_precoder_net_scope(str(model_specs[0].get("model_scope", "per_user_nets")))
    if first_scope == "bs_shared_net":
        first_spec = model_specs[0]
        input_mode = str(first_spec.get("input_mode", "")).strip().lower()
        if input_mode == "block_context_noise_epsilon":
            shared_model = build_shared_bs_precoder_net_with_context(
                k_count=int(first_spec.get("context_k", len(model_specs))),
                max_nr=int(first_spec.get("context_max_nr", first_spec["nr"])),
                max_nb=int(first_spec.get("context_max_nb", first_spec["nb"])),
                max_dk=int(first_spec.get("context_max_dk", first_spec["dk"])),
                device=device,
            )
        elif bool(first_spec.get("uses_blocklength_input", False)) or input_mode == "block_context_noise_epsilon_n":
            shared_model = build_shared_bs_precoder_net_with_blocklength(
                k_count=int(first_spec.get("context_k", len(model_specs))),
                max_nr=int(first_spec.get("context_max_nr", first_spec["nr"])),
                max_nb=int(first_spec.get("context_max_nb", first_spec["nb"])),
                max_dk=int(first_spec.get("context_max_dk", first_spec["dk"])),
                device=device,
            )
        else:
            shared_model = build_shared_bs_precoder_net(
                k_count=int(first_spec.get("context_k", len(model_specs))),
                max_nr=int(first_spec.get("context_max_nr", first_spec["nr"])),
                max_nb=int(first_spec.get("context_max_nb", first_spec["nb"])),
                max_dk=int(first_spec.get("context_max_dk", first_spec["dk"])),
                device=device,
            )
        shared_model.load_state_dict(model_states[0])
        shared_model.eval()
        return [shared_model for _ in model_specs]

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
