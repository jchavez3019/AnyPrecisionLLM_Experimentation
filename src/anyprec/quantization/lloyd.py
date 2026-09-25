"""Weighted Lloyd's algorithm on sorted rows (ADR 0003, Section 3)."""

from dataclasses import dataclass

import torch

from anyprec.quantization.init import weighted_kmeanspp_init
from anyprec.quantization.rows import PreparedRows, segment_stats


@dataclass(frozen=True)
class LloydResult:
    """A fitted codebook per row.

    :param centroids: Ascending centroids, float64 ``[R, K]``.
    :param borders: Segment borders in sorted order, int64 ``[R, K + 1]``, from 0 to ``n``.
    :param iterations: Number of assignment steps run.
    """

    centroids: torch.Tensor
    borders: torch.Tensor
    iterations: int


def weighted_lloyd(
    rows: PreparedRows,
    num_centroids: int,
    generator: torch.Generator,
    max_iter: int,
    empty_eps: float,
) -> LloydResult:
    """Fit weighted 1-D k-means to every row, starting from weighted k-means++.

    :param rows: Prepared rows.
    :param num_centroids: Codebook size ``K``.
    :param generator: Seeded generator for the initialization, on the rows' device.
    :param max_iter: Iteration cap.
    :param empty_eps: Mass at or below which a segment counts as empty.
    :return: Centroids, borders, and the iteration count.
    """
    num_rows, n = rows.w_sorted.shape
    centroids: torch.Tensor = weighted_kmeanspp_init(rows, num_centroids, generator)
    device = centroids.device
    first_border: torch.Tensor = torch.zeros(num_rows, 1, dtype=torch.long, device=device)
    last_border: torch.Tensor = torch.full((num_rows, 1), n, dtype=torch.long, device=device)
    borders: torch.Tensor | None = None
    iterations: int = 0
    for iteration in range(1, max_iter + 1):
        iterations = iteration

        # Assign: [R, K] centroids -> [R, K - 1] midpoints -> sorted positions -> [R, K + 1] borders.

        midpoints: torch.Tensor = 0.5 * (centroids[:, :-1] + centroids[:, 1:])
        interior: torch.Tensor = torch.searchsorted(rows.w_sorted, midpoints.contiguous())
        new_borders: torch.Tensor = torch.cat([first_border, interior, last_border], dim=1)

        # Converged once no border moves; the centroids are already the means of these segments.

        if borders is not None and torch.equal(new_borders, borders):
            break
        borders = new_borders

        # Update: weighted means of non-empty segments; re-sorting keeps the midpoints monotone.

        mass, mean, _ = segment_stats(rows, borders[:, :-1], borders[:, 1:], empty_eps)
        centroids = torch.where(mass > empty_eps, mean, centroids).sort(dim=1).values
    if borders is None:
        raise ValueError(f"max_iter must be positive, got {max_iter}")
    return LloydResult(centroids, borders, iterations)
