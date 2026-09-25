"""Weighted greedy k-means++ initialization (ADR 0003, Section 3)."""

import math

import torch

from anyprec.quantization.rows import PreparedRows


def _row_normalized(values: torch.Tensor) -> torch.Tensor:
    """Scale each row so its maximum is 1, then cast to float32 for ``torch.multinomial``.

    Normalizing first keeps small float64 values from underflowing in float32.

    :param values: Non-negative weights with a positive maximum per row, float64 ``[R, n]``.
    :return: Float32 sampling weights ``[R, n]``.
    """
    return (values / values.amax(dim=1, keepdim=True)).float()


def weighted_kmeanspp_init(
    rows: PreparedRows, num_centroids: int, generator: torch.Generator
) -> torch.Tensor:
    """Initialize centroids with weighted greedy k-means++.

    The first centroid is drawn in proportion to ``f``. Each further centroid is the best of
    ``L = 2 + floor(ln K)`` candidates drawn in proportion to ``f * D``, where ``D`` is the
    squared distance to the nearest centroid chosen so far.

    :param rows: Prepared rows.
    :param num_centroids: Codebook size ``K``.
    :param generator: Seeded generator on the rows' device.
    :return: Initial centroids, float64 ``[R, K]``, ascending.
    """
    n: int = rows.w_sorted.shape[1]
    local_trials: int = 2 + int(math.log(num_centroids))
    w, f = rows.w_sorted, rows.f_sorted

    # First centroid drawn in proportion to sensitivity: [R, n] -> [R, 1] index -> [R, 1] value.

    first: torch.Tensor = torch.multinomial(_row_normalized(f), 1, generator=generator)
    chosen: list[torch.Tensor] = [w.gather(1, first)]
    nearest: torch.Tensor = (w - chosen[0]).square()
    for _ in range(1, num_centroids):
        # Candidates are drawn in proportion to f * D. A row whose weights all coincide with
        # chosen centroids has zero potential and falls back to f, which can only repeat values.

        potential: torch.Tensor = f * nearest
        potential = torch.where(potential.sum(dim=1, keepdim=True) > 0, potential, f)
        candidates: torch.Tensor = torch.multinomial(
            _row_normalized(potential), local_trials, replacement=True, generator=generator
        )
        values: torch.Tensor = w.gather(1, candidates)

        # Nearest-centroid distance if each candidate were added: [R, 1, n] vs [R, L, n] -> [R, L, n].
        # Keep the candidate with the smallest resulting potential: [R, L] -> [R, 1].

        trial: torch.Tensor = torch.minimum(
            nearest[:, None, :], (w[:, None, :] - values[:, :, None]).square()
        )
        best: torch.Tensor = (f[:, None, :] * trial).sum(dim=2).argmin(dim=1, keepdim=True)
        chosen.append(values.gather(1, best))
        nearest = trial.gather(1, best[:, :, None].expand(-1, 1, n)).squeeze(1)

    # K x [R, 1] -> [R, K], ascending so midpoints between neighbours are well defined.

    return torch.cat(chosen, dim=1).sort(dim=1).values
