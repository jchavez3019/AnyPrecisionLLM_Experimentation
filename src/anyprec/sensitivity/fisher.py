"""Empirical Fisher diagonal of every target weight (ADR 0003, Section 1; spec 0004)."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FisherResult:
    """Empirical Fisher diagonals and calibration diagnostics.

    :param diagonals: Float32 CPU tensors keyed by module name in discovery order, each shaped
        like its weight ``[m, n]``.
    :param losses: Per-sequence mean token NLL, float32 ``[N]``.
    :param seconds: Wall-clock time of the accumulation loop.
    """

    diagonals: dict[str, torch.Tensor]
    losses: torch.Tensor
    seconds: float
