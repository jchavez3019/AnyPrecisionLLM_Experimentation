"""Exact 2-way splits for incremental upscaling (ADR 0003, Section 4)."""

import math

import torch

from anyprec.quantization.rows import PreparedRows, segment_stats


def split_all_segments(
    rows: PreparedRows, borders: torch.Tensor, centroids: torch.Tensor, empty_eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split every segment at its cost-minimizing point.

    Segment ``k`` becomes children ``2k`` (left) and ``2k + 1`` (right), so the new bit is the
    least significant one and indices stay nested. A segment that cannot be split (fewer than
    two points) keeps all its points in the right child, and the empty left child inherits the
    parent centroid.

    :param rows: Prepared rows.
    :param borders: Segment borders at bit-width ``b``, int64 ``[R, K + 1]``.
    :param centroids: Centroids at bit-width ``b``, float64 ``[R, K]``.
    :param empty_eps: Mass at or below which a segment counts as empty.
    :return: Centroids ``[R, 2K]`` and borders ``[R, 2K + 1]`` at bit-width ``b + 1``.
    """
    num_rows, n = rows.w_sorted.shape
    num_segments: int = centroids.shape[1]
    device = borders.device

    # Every interior split point, and the segment that owns it: [n - 1] -> [R, n - 1].

    candidates: torch.Tensor = torch.arange(1, n, device=device).expand(num_rows, -1).contiguous()
    owner: torch.Tensor = torch.searchsorted(borders[:, 1:].contiguous(), candidates, right=True)
    start: torch.Tensor = borders.gather(1, owner)
    end: torch.Tensor = borders.gather(1, owner + 1)

    # Cost of splitting at each candidate; a valid split leaves both children non-empty. [R, n - 1]

    _, _, left_cost = segment_stats(rows, start, candidates, empty_eps)
    _, _, right_cost = segment_stats(rows, candidates, end, empty_eps)
    valid: torch.Tensor = (candidates > start) & (candidates < end)
    total: torch.Tensor = torch.where(valid, left_cost + right_cost, math.inf)

    # Segmented argmin: the minimum per owning segment, then the first candidate attaining it. [R, K]

    best: torch.Tensor = torch.full(
        (num_rows, num_segments), math.inf, dtype=torch.float64, device=device
    )
    best = best.scatter_reduce(1, owner, total, reduce="amin")
    hit: torch.Tensor = total == best.gather(1, owner)
    split: torch.Tensor = torch.full((num_rows, num_segments), n, dtype=torch.long, device=device)
    split = split.scatter_reduce(1, owner, torch.where(hit, candidates, n), reduce="amin")
    split = torch.where(best.isfinite(), split, borders[:, :-1])

    # Interleave parent starts with split points: [R, K] x 2 -> [R, K, 2] -> [R, 2K], then append
    # the final border n -> [R, 2K + 1].

    child_borders: torch.Tensor = torch.stack([borders[:, :-1], split], dim=2).reshape(
        num_rows, 2 * num_segments
    )
    child_borders = torch.cat([child_borders, borders[:, -1:]], dim=1)

    # Children take their segment means; empty children inherit the parent centroid. [R, 2K]

    mass, mean, _ = segment_stats(rows, child_borders[:, :-1], child_borders[:, 1:], empty_eps)
    parent: torch.Tensor = centroids.repeat_interleave(2, dim=1)
    return torch.where(mass > empty_eps, mean, parent), child_borders


def segment_ids(borders: torch.Tensor, n: int) -> torch.Tensor:
    """Map every sorted position to the index of the segment containing it.

    Empty segments own no position, because ``searchsorted(..., right=True)`` skips repeated
    borders.

    :param borders: Segment borders, int64 ``[R, K + 1]``.
    :param n: Row length.
    :return: Segment index of each sorted position, int64 ``[R, n]``.
    """
    # [n] positions broadcast to [R, n], located among the K right borders [R, K].

    positions: torch.Tensor = torch.arange(n, device=borders.device).expand(borders.shape[0], -1)
    return torch.searchsorted(borders[:, 1:].contiguous(), positions.contiguous(), right=True)
