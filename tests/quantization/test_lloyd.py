"""Tests for weighted Lloyd iterations (spec 0005)."""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from anyprec.quantization.init import weighted_kmeanspp_init
from anyprec.quantization.lloyd import weighted_lloyd
from anyprec.quantization.rows import prepare_rows, segment_stats
from anyprec.quantization.split import segment_ids
from tests.quantization import oracles
from tests.strategies import weighted_rows

EMPTY_EPS: float = 1e-12
MAX_ITER: int = 100


@settings(max_examples=200)
@given(weighted_rows(), st.integers(1, 3), st.integers(0, 2**31))
def test_lloyd_never_increases_cost_over_its_initialization(
    case: tuple[torch.Tensor, torch.Tensor], bits: int, seed: int
) -> None:
    """
    Given: prepared rows and a seed.
    When: Lloyd runs from the k-means++ initialization drawn with that seed.
    Then: its cost is at most the cost of assigning every value to its nearest initial centroid.
    """
    rows = prepare_rows(*case)
    num_centroids = 2**bits
    initial = weighted_kmeanspp_init(rows, num_centroids, torch.Generator().manual_seed(seed))

    result = weighted_lloyd(
        rows, num_centroids, torch.Generator().manual_seed(seed), MAX_ITER, EMPTY_EPS
    )

    # Nearest initial centroid per value: [R, n, 1] vs [R, 1, K] -> [R, n].

    nearest = (rows.w_sorted[:, :, None] - initial[:, None, :]).abs().argmin(dim=2)
    initial_cost = oracles.reconstruction_cost(rows.w_sorted, rows.f_sorted, initial, nearest)
    final_ids = segment_ids(result.borders, rows.w_sorted.shape[1])
    final_cost = oracles.reconstruction_cost(
        rows.w_sorted, rows.f_sorted, result.centroids, final_ids
    )
    assert final_cost <= initial_cost * (1 + 1e-9) + 1e-12


@settings(max_examples=200)
@given(weighted_rows(), st.integers(1, 3), st.integers(0, 2**31))
def test_lloyd_converges_to_a_fixed_point_of_assign_and_update(
    case: tuple[torch.Tensor, torch.Tensor], bits: int, seed: int
) -> None:
    """
    Given: prepared rows and a seed.
    When: Lloyd converges before its iteration cap.
    Then: each non-empty segment's centroid is its weighted mean, and assigning by midpoints of
        the returned centroids reproduces the returned borders.
    """
    rows = prepare_rows(*case)

    result = weighted_lloyd(rows, 2**bits, torch.Generator().manual_seed(seed), MAX_ITER, EMPTY_EPS)

    assert result.iterations < MAX_ITER
    mass, mean, _ = segment_stats(rows, result.borders[:, :-1], result.borders[:, 1:], EMPTY_EPS)
    non_empty = mass > EMPTY_EPS
    torch.testing.assert_close(result.centroids[non_empty], mean[non_empty], rtol=1e-9, atol=1e-12)

    # One more assignment step: [R, K - 1] midpoints -> interior borders.

    midpoints = 0.5 * (result.centroids[:, :-1] + result.centroids[:, 1:])
    interior = torch.searchsorted(rows.w_sorted, midpoints.contiguous())
    assert torch.equal(interior, result.borders[:, 1:-1])


@settings(max_examples=200)
@given(weighted_rows(max_rows=2, max_n=10), st.sampled_from([2, 3]), st.integers(0, 2**31))
def test_lloyd_is_no_better_than_exhaustive_optimum(
    case: tuple[torch.Tensor, torch.Tensor], num_centroids: int, seed: int
) -> None:
    """
    Given: rows of at most 10 values and K of 2 or 3.
    When: Lloyd runs.
    Then: its cost is at least the optimum over every contiguous partition, and its borders
        start at 0, end at n, and never decrease.
    """
    rows = prepare_rows(*case)
    n = rows.w_sorted.shape[1]

    result = weighted_lloyd(
        rows, num_centroids, torch.Generator().manual_seed(seed), MAX_ITER, EMPTY_EPS
    )

    assert bool((result.borders[:, 0] == 0).all())
    assert bool((result.borders[:, -1] == n).all())
    assert bool((result.borders[:, 1:] >= result.borders[:, :-1]).all())
    for r in range(rows.w_sorted.shape[0]):
        # Lloyd is a local method, so it can only match or exceed the exhaustive optimum.

        w, f = oracles.floats(rows.w_sorted[r]), oracles.floats(rows.f_sorted[r])
        achieved = oracles.partition_cost(w, f, oracles.ints(result.borders[r]))
        optimum = oracles.optimal_partition_cost(w, f, num_centroids)
        assert achieved >= optimum - 1e-9 * max(1.0, optimum)


def test_lloyd_rejects_non_positive_iteration_cap() -> None:
    """
    Given: prepared rows.
    When: Lloyd is asked to run zero iterations.
    Then: it raises ValueError instead of returning borders it never computed.
    """
    rows = prepare_rows(torch.tensor([[1.0, 2.0, 3.0]]), torch.ones(1, 3))

    with pytest.raises(ValueError, match="max_iter"):
        weighted_lloyd(rows, 2, torch.Generator().manual_seed(0), 0, EMPTY_EPS)
