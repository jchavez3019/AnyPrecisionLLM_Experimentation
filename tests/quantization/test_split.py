"""Tests for exact 2-way splits (spec 0005)."""

import itertools

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from anyprec.quantization.rows import PreparedRows, prepare_rows, segment_stats
from anyprec.quantization.split import segment_ids, split_all_segments
from tests.quantization import oracles
from tests.strategies import weighted_rows

EMPTY_EPS: float = 1e-12


def _random_partition(
    rows: PreparedRows, num_segments: int, data: st.DataObject
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw arbitrary borders, empty segments included, and their mean centroids.

    :param rows: Prepared rows.
    :param num_segments: Number of segments ``K``.
    :param data: Hypothesis data object.
    :return: Borders int64 ``[R, K + 1]`` and centroids float64 ``[R, K]``.
    """
    num_rows, n = rows.w_sorted.shape
    cut_lists = [
        sorted(
            data.draw(
                st.lists(st.integers(0, n), min_size=num_segments - 1, max_size=num_segments - 1)
            )
        )
        for _ in range(num_rows)
    ]
    borders = torch.tensor([[0, *cuts, n] for cuts in cut_lists])
    mass, mean, _ = segment_stats(rows, borders[:, :-1], borders[:, 1:], EMPTY_EPS)
    return borders, torch.where(mass > EMPTY_EPS, mean, torch.zeros_like(mean))


@settings(max_examples=200)
@given(weighted_rows(), st.sampled_from([1, 2, 4]), st.data())
def test_split_all_segments_matches_brute_force_optimum(
    case: tuple[torch.Tensor, torch.Tensor], num_segments: int, data: st.DataObject
) -> None:
    """
    Given: prepared rows partitioned into K arbitrary segments, some possibly empty.
    When: every segment is split.
    Then: each segment's children cost the minimum over all of its split points, a segment with
        fewer than two values keeps them all in its right child, and parent borders are kept.
    """
    rows = prepare_rows(*case)
    borders, centroids = _random_partition(rows, num_segments, data)

    _, child_borders = split_all_segments(rows, borders, centroids, EMPTY_EPS)

    assert torch.equal(child_borders[:, ::2], borders)
    for r in range(borders.shape[0]):
        w, f = oracles.floats(rows.w_sorted[r]), oracles.floats(rows.f_sorted[r])
        for k, (start, end) in enumerate(itertools.pairwise(oracles.ints(borders[r]))):
            split = int(child_borders[r, 2 * k + 1])
            if end - start < 2:
                assert split == start
                continue
            achieved = oracles.partition_cost(w, f, [start, split, end])
            best = min(oracles.partition_cost(w, f, [start, s, end]) for s in range(start + 1, end))
            assert start < split < end
            assert achieved <= best + 1e-9 * max(1.0, best)


@settings(max_examples=200)
@given(weighted_rows(), st.sampled_from([1, 2, 4]), st.data())
def test_split_all_segments_never_increases_cost_and_keeps_indices_nested(
    case: tuple[torch.Tensor, torch.Tensor], num_segments: int, data: st.DataObject
) -> None:
    """
    Given: prepared rows partitioned into K arbitrary segments.
    When: every segment is split.
    Then: the reconstruction cost does not increase, every child id shifted right by one bit is
        its parent's id, and empty children inherit their parent centroid.
    """
    rows = prepare_rows(*case)
    n = rows.w_sorted.shape[1]
    borders, centroids = _random_partition(rows, num_segments, data)

    child_centroids, child_borders = split_all_segments(rows, borders, centroids, EMPTY_EPS)

    parent_ids = segment_ids(borders, n)
    child_ids = segment_ids(child_borders, n)
    assert torch.equal(child_ids >> 1, parent_ids)
    before = oracles.reconstruction_cost(rows.w_sorted, rows.f_sorted, centroids, parent_ids)
    after = oracles.reconstruction_cost(rows.w_sorted, rows.f_sorted, child_centroids, child_ids)
    assert after <= before * (1 + 1e-9) + 1e-12
    mass, _, _ = segment_stats(rows, child_borders[:, :-1], child_borders[:, 1:], EMPTY_EPS)
    empty = mass <= EMPTY_EPS
    assert torch.equal(child_centroids[empty], centroids.repeat_interleave(2, dim=1)[empty])
