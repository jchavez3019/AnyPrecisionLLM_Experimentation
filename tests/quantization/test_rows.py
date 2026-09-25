"""Tests for row preparation and segment statistics (spec 0005)."""

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from anyprec.quantization.rows import prepare_rows, segment_stats
from tests.quantization import oracles
from tests.strategies import weighted_rows

EMPTY_EPS: float = 1e-12


@settings(max_examples=200)
@given(weighted_rows())
def test_prepare_rows_sorts_conditions_and_prefix_sums_like_brute_force(
    case: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """
    Given: rows with ties, zeros, dominant sensitivities, and all-zero Fisher rows.
    When: they are prepared.
    Then: values are sorted, sensitivities follow the conditioning rule, and all three prefix
        sums equal running sums computed in a Python loop.
    """
    weight, fisher = case
    rows = prepare_rows(weight, fisher)
    expected_f = oracles.conditioned_fisher(weight, fisher).gather(1, rows.order)

    torch.testing.assert_close(rows.w_sorted, weight.double().sort(dim=1).values)
    torch.testing.assert_close(rows.f_sorted, expected_f, rtol=0.0, atol=0.0)
    for r in range(weight.shape[0]):
        # Running sums of f, f*w, and f*w^2, each with a leading zero.

        sums: list[list[float]] = [[0.0], [0.0], [0.0]]
        for w, f in zip(
            oracles.floats(rows.w_sorted[r]), oracles.floats(rows.f_sorted[r]), strict=True
        ):
            for power, running in enumerate(sums):
                running.append(running[-1] + f * w**power)
        for prefix, running in zip((rows.p0, rows.p1, rows.p2), sums, strict=True):
            torch.testing.assert_close(
                prefix[r], torch.tensor(running, dtype=torch.float64), rtol=1e-9, atol=1e-9
            )


def test_prepare_rows_zeroes_sensitivity_of_zero_weights_and_falls_back_to_uniform() -> None:
    """
    Given: one row whose only zero weight has large Fisher, and one row whose Fisher is nonzero
        only at a zero weight.
    When: the rows are prepared.
    Then: the first row's zero weight gets sensitivity 0, and the second row becomes uniform.
    """
    weight = torch.tensor([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
    fisher = torch.tensor([[9.0, 1.0, 1.0], [5.0, 0.0, 0.0]])

    rows = prepare_rows(weight, fisher)

    expected = torch.tensor([[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
    assert torch.equal(rows.f_sorted, expected)


@settings(max_examples=200)
@given(weighted_rows(), st.data())
def test_segment_stats_matches_direct_sums_including_empty_segments(
    case: tuple[torch.Tensor, torch.Tensor], data: st.DataObject
) -> None:
    """
    Given: prepared rows and random segments [start, end), some of them empty.
    When: segment statistics are computed.
    Then: mass, mean, and cost equal direct sums, and empty segments cost exactly 0. Tolerances
        scale with the row's prefix totals, because a segment statistic is a difference of two
        prefix values and inherits their absolute rounding error.
    """
    weight, fisher = case
    rows = prepare_rows(weight, fisher)
    num_rows, n = weight.shape
    pair = st.lists(st.integers(0, n), min_size=2, max_size=2)
    bounds = [sorted(data.draw(pair)) for _ in range(num_rows)]
    start = torch.tensor([[s] for s, _ in bounds])
    end = torch.tensor([[e] for _, e in bounds])

    mass, mean, cost = segment_stats(rows, start, end, EMPTY_EPS)

    for r, (s, e) in enumerate(bounds):
        # Absolute rounding budgets of the three prefix differences, from the row totals.

        f_row, w_row = rows.f_sorted[r], rows.w_sorted[r]
        tol_m = 1e-12 * float(f_row.sum())
        tol_s = 1e-12 * float((f_row * w_row.abs()).sum())
        tol_q = 1e-12 * float((f_row * w_row.square()).sum())
        w, f = oracles.floats(w_row[s:e]), oracles.floats(f_row[s:e])
        seg_mass = sum(f)
        assert abs(mass[r, 0].item() - seg_mass) <= tol_m
        if seg_mass <= EMPTY_EPS:
            assert cost[r, 0].item() == 0.0
            continue

        # Propagate the budgets through mean = S / M and cost = Q - S^2 / M.

        expected_mean = sum(fi * wi for fi, wi in zip(f, w, strict=True)) / seg_mass
        tol_mean = (tol_s + abs(expected_mean) * tol_m) / seg_mass
        tol_cost = tol_q + 2 * abs(expected_mean) * tol_s + expected_mean**2 * tol_m
        assert abs(mean[r, 0].item() - expected_mean) <= tol_mean + 1e-12
        assert abs(cost[r, 0].item() - oracles.segment_cost(w, f)) <= tol_cost + 1e-12
