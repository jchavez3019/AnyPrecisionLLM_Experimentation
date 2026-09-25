"""Row preparation and segment statistics for weighted 1-D k-means (ADR 0003, Sections 1-2)."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PreparedRows:
    """Sorted rows, their conditioned sensitivities, and float64 prefix sums (ADR 0003, Section 2).

    :param order: Sorting permutation of each row, int64 ``[R, n]``.
    :param w_sorted: Row values in ascending order, float64 ``[R, n]``.
    :param f_sorted: Conditioned sensitivities in the same order, float64 ``[R, n]``.
    :param p0: Prefix sums of ``f`` with a leading zero, float64 ``[R, n + 1]``.
    :param p1: Prefix sums of ``f * w`` with a leading zero, float64 ``[R, n + 1]``.
    :param p2: Prefix sums of ``f * w**2`` with a leading zero, float64 ``[R, n + 1]``.
    """

    order: torch.Tensor
    w_sorted: torch.Tensor
    f_sorted: torch.Tensor
    p0: torch.Tensor
    p1: torch.Tensor
    p2: torch.Tensor


def prepare_rows(weight: torch.Tensor, fisher: torch.Tensor) -> PreparedRows:
    """Condition sensitivities, sort each row, and build the prefix sums.

    :param weight: Rows to quantize, ``[R, n]``, any floating dtype.
    :param fisher: Fisher diagonal of the same rows, ``[R, n]``, non-negative.
    :return: The prepared rows, on the inputs' device.
    """
    # Zero weights carry no sensitivity; a row with no sensitivity at all falls back to
    # unweighted k-means (ADR 0003, Section 1, sensitivity conditioning).

    f: torch.Tensor = fisher.double() * (weight != 0)
    f = torch.where(f.sum(dim=1, keepdim=True) > 0, f, torch.ones_like(f))

    # Sort each row once; every later step works on contiguous segments of this order.
    # A stable sort keeps tied weights (common in bfloat16) in column order, so which tied
    # column lands on which side of a split does not depend on the sort implementation.
    # [R, n] -> [R, n] permutation, then sorted values and their sensitivities.

    order: torch.Tensor = weight.argsort(dim=1, stable=True)
    w_sorted: torch.Tensor = weight.gather(1, order).double()
    f_sorted: torch.Tensor = f.gather(1, order)

    # [R, n] -> [R, n + 1]: a leading zero makes segment [s, e) equal to P[e] - P[s].
    # Float64 avoids the catastrophic cancellation of Q - S^2 / M in float32.

    zero: torch.Tensor = w_sorted.new_zeros(w_sorted.shape[0], 1)
    p0: torch.Tensor = torch.cat([zero, f_sorted.cumsum(dim=1)], dim=1)
    p1: torch.Tensor = torch.cat([zero, (f_sorted * w_sorted).cumsum(dim=1)], dim=1)
    p2: torch.Tensor = torch.cat([zero, (f_sorted * w_sorted * w_sorted).cumsum(dim=1)], dim=1)
    return PreparedRows(order, w_sorted, f_sorted, p0, p1, p2)


def segment_stats(
    rows: PreparedRows, start: torch.Tensor, end: torch.Tensor, empty_eps: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute mass, weighted mean, and weighted squared error of sorted segments ``[start, end)``.

    :param rows: Prepared rows.
    :param start: Inclusive segment starts, int64 ``[R, K]``.
    :param end: Exclusive segment ends, int64 ``[R, K]``.
    :param empty_eps: Mass at or below which a segment counts as empty.
    :return: Mass, mean, and cost, each float64 ``[R, K]``. Empty segments cost 0, and their
        mean is meaningless (callers substitute an inherited centroid).
    """
    # Gather prefix sums at the borders: [R, n + 1] -> [R, K] per statistic.

    mass: torch.Tensor = rows.p0.gather(1, end) - rows.p0.gather(1, start)
    first: torch.Tensor = rows.p1.gather(1, end) - rows.p1.gather(1, start)
    second: torch.Tensor = rows.p2.gather(1, end) - rows.p2.gather(1, start)
    safe_mass: torch.Tensor = mass.clamp_min(empty_eps)
    mean: torch.Tensor = first / safe_mass

    # Clamping removes the tiny negative round-off that Q - S^2 / M can still produce in float64.

    cost: torch.Tensor = torch.where(
        mass > empty_eps,
        (second - first * first / safe_mass).clamp_min(0.0),
        torch.zeros_like(mass),
    )
    return mass, mean, cost
