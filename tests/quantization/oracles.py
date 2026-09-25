"""Brute-force reference computations for the k-means kernel tests (spec 0010).

Everything here is written as plain Python loops in float64, deliberately unlike the vectorized
kernels it checks.
"""

import itertools

import torch


def floats(values: torch.Tensor) -> list[float]:
    """Convert a 1-D tensor to a typed list of Python floats.

    :param values: A 1-D tensor.
    :return: Its elements as floats.
    """
    return [float(values[i].item()) for i in range(values.shape[0])]


def ints(values: torch.Tensor) -> list[int]:
    """Convert a 1-D integer tensor to a typed list of Python ints.

    :param values: A 1-D integer tensor.
    :return: Its elements as ints.
    """
    return [int(values[i].item()) for i in range(values.shape[0])]


def conditioned_fisher(weight: torch.Tensor, fisher: torch.Tensor) -> torch.Tensor:
    """Apply ADR 0003's conditioning rule row by row.

    :param weight: Weights ``[R, n]``.
    :param fisher: Raw Fisher diagonal ``[R, n]``.
    :return: Float64 ``[R, n]``: zero where the weight is zero, and all ones for rows whose
        conditioned sensitivities sum to zero.
    """
    rows: list[list[float]] = []
    for r in range(weight.shape[0]):
        pairs = zip(floats(weight[r]), floats(fisher[r]), strict=True)
        row = [f if w != 0.0 else 0.0 for w, f in pairs]
        rows.append(row if sum(row) > 0.0 else [1.0] * len(row))
    return torch.tensor(rows, dtype=torch.float64)


def segment_cost(w: list[float], f: list[float]) -> float:
    """Weighted squared error of a segment about its weighted mean.

    :param w: Segment values.
    :param f: Segment sensitivities.
    :return: ``sum f (w - mean)^2``, or 0 for a massless segment.
    """
    mass = sum(f)
    if mass <= 0.0:
        return 0.0
    mean = sum(fi * wi for fi, wi in zip(f, w, strict=True)) / mass
    return sum(fi * (wi - mean) ** 2 for fi, wi in zip(f, w, strict=True))


def partition_cost(w: list[float], f: list[float], borders: list[int]) -> float:
    """Total cost of a contiguous partition of sorted values.

    :param w: Sorted values.
    :param f: Their sensitivities.
    :param borders: Ascending borders from 0 to ``len(w)``.
    :return: Sum of the segment costs.
    """
    return sum(segment_cost(w[s:e], f[s:e]) for s, e in itertools.pairwise(borders))


def optimal_partition_cost(w: list[float], f: list[float], num_segments: int) -> float:
    """Exact 1-D weighted k-means optimum by enumerating every contiguous partition.

    Empty segments are allowed, so this is also the optimum for at most ``num_segments`` segments.

    :param w: Sorted values.
    :param f: Their sensitivities.
    :param num_segments: Number of segments ``K``.
    :return: The smallest total cost.
    """
    n = len(w)
    return min(
        partition_cost(w, f, [0, *cuts, n])
        for cuts in itertools.combinations_with_replacement(range(n + 1), num_segments - 1)
    )


def reconstruction_cost(
    w: torch.Tensor, f: torch.Tensor, centroids: torch.Tensor, ids: torch.Tensor
) -> float:
    """Weighted squared error of reconstructing ``w`` from ``centroids[ids]``.

    :param w: Values ``[R, n]``.
    :param f: Sensitivities ``[R, n]``.
    :param centroids: Codebooks ``[R, K]``.
    :param ids: Codebook index per value ``[R, n]``.
    :return: ``sum f (w - c[id])^2`` in float64.
    """
    residual = w.double() - centroids.double().gather(1, ids.long())
    return float((f.double() * residual.square()).sum())
