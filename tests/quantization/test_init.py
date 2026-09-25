"""Tests for weighted greedy k-means++ initialization (spec 0005)."""

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from anyprec.quantization.init import weighted_kmeanspp_init
from anyprec.quantization.rows import prepare_rows
from tests.quantization import oracles
from tests.strategies import weighted_rows


@settings(max_examples=200)
@given(weighted_rows(), st.integers(1, 3), st.integers(0, 2**31))
def test_kmeanspp_returns_sorted_row_values_distinct_when_enough_sensitive_values(
    case: tuple[torch.Tensor, torch.Tensor], bits: int, seed: int
) -> None:
    """
    Given: prepared rows, a codebook of K = 2**bits centroids, and a seed.
    When: k-means++ initialization runs.
    Then: each row's centroids are ascending values of that row, and they are distinct whenever
        the row has at least K distinct values with positive sensitivity (only those can be drawn).
    """
    weight, fisher = case
    rows = prepare_rows(weight, fisher)
    num_centroids = 2**bits

    centroids = weighted_kmeanspp_init(rows, num_centroids, torch.Generator().manual_seed(seed))

    assert centroids.shape == (weight.shape[0], num_centroids)
    assert bool((centroids[:, 1:] >= centroids[:, :-1]).all())
    for r in range(weight.shape[0]):
        row_values = set(oracles.floats(rows.w_sorted[r]))
        drawable = set(oracles.floats(rows.w_sorted[r][rows.f_sorted[r] > 0]))
        chosen = oracles.floats(centroids[r])
        assert set(chosen) <= row_values
        if len(drawable) >= num_centroids:
            assert len(set(chosen)) == num_centroids


@settings(max_examples=50)
@given(weighted_rows(), st.integers(0, 2**31))
def test_kmeanspp_is_identical_for_the_same_seed(
    case: tuple[torch.Tensor, torch.Tensor], seed: int
) -> None:
    """
    Given: prepared rows.
    When: initialization runs twice from generators seeded with the same value.
    Then: both runs return bitwise-identical centroids.
    """
    rows = prepare_rows(*case)

    first = weighted_kmeanspp_init(rows, 4, torch.Generator().manual_seed(seed))
    second = weighted_kmeanspp_init(rows, 4, torch.Generator().manual_seed(seed))

    assert torch.equal(first, second)


def test_kmeanspp_first_draw_survives_tiny_sensitivities() -> None:
    """
    Given: a row whose sensitivities are all around 1e-300, far below float32's range.
    When: initialization runs.
    Then: it still returns K distinct values instead of failing in torch.multinomial.
    """
    weight = torch.arange(8, dtype=torch.float32)[None, :] + 1.0
    fisher = torch.full((1, 8), 1e-300, dtype=torch.float64)
    rows = prepare_rows(weight, fisher)

    centroids = weighted_kmeanspp_init(rows, 4, torch.Generator().manual_seed(0))

    assert len(set(oracles.floats(centroids[0]))) == 4
