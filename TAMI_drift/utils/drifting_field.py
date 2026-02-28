"""
Drifting field V computation (adapted from "Generative Modeling via Drifting").

This module is intentionally self-contained so it can be reused as an auxiliary
regularizer in TAMI training while keeping all computations on CUDA.
"""

from __future__ import annotations

from typing import List

import torch


def compute_V(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    temperature: float,
    mask_self: bool = False,
) -> torch.Tensor:
    """
    Compute drifting field V.

    Args:
        x: anchor samples, shape (N, D)
        y_pos: positive samples, shape (N_pos, D)
        y_neg: negative samples, shape (N_neg, D)
        temperature: temperature for softmax (smaller = sharper)
        mask_self: if True and y_neg == x with same N, mask self distances.

    Returns:
        V: drifting field, shape (N, D)
    """
    # cdist/softmax are more stable in fp32; also guard against NaN/Inf from upstream models.
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    y_pos = torch.nan_to_num(y_pos.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    y_neg = torch.nan_to_num(y_neg.float(), nan=0.0, posinf=1e4, neginf=-1e4)

    N = x.shape[0]
    N_pos = y_pos.shape[0]
    N_neg = y_neg.shape[0]

    # Pairwise L2 distances
    dist_pos = torch.cdist(x, y_pos, p=2)  # (N, N_pos)
    dist_neg = torch.cdist(x, y_neg, p=2)  # (N, N_neg)

    if mask_self and N == N_neg:
        dist_neg = dist_neg + torch.eye(N, device=x.device, dtype=dist_neg.dtype) * 1e6

    logit_pos = -dist_pos / temperature
    logit_neg = -dist_neg / temperature
    logit = torch.cat([logit_pos, logit_neg], dim=1)  # (N, N_pos + N_neg)
    logit = logit.clamp(min=-50.0, max=50.0)

    # Normalize along both dimensions (geometric mean)
    A_row = torch.softmax(logit, dim=1)
    A_col = torch.softmax(logit, dim=0)
    A = torch.sqrt(A_row * A_col + 1e-12)

    A_pos = A[:, :N_pos]
    A_neg = A[:, N_pos:]

    W_pos = A_pos * A_neg.sum(dim=1, keepdim=True)
    W_neg = A_neg * A_pos.sum(dim=1, keepdim=True)

    drift_pos = W_pos @ y_pos
    drift_neg = W_neg @ y_neg
    return drift_pos - drift_neg


def compute_V_multi_temperature(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    temperatures: List[float] | None = None,
    mask_self: bool = False,
    normalize_each: bool = True,
) -> torch.Tensor:
    """
    Compute drifting field with multiple temperatures.

    Each temperature's V is optionally normalized so E[||V||^2] ~ 1 before summing.
    """
    if temperatures is None:
        temperatures = [0.02, 0.05, 0.2]

    V_total = torch.zeros_like(x)
    for tau in temperatures:
        V_tau = compute_V(x, y_pos, y_neg, tau, mask_self=mask_self)
        if normalize_each:
            V_norm = torch.sqrt(torch.mean(V_tau ** 2) + 1e-8)
            V_tau = V_tau / (V_norm + 1e-8)
        V_total = V_total + V_tau
    return V_total
