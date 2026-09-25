"""Hypothesis strategies for weight rows and sensitivities (spec 0010)."""

import torch
from hypothesis import strategies as st


@st.composite
def weighted_rows(
    draw: st.DrawFn, max_rows: int = 4, min_n: int = 2, max_n: int = 48
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw ``(weight [R, n], fisher [R, n])`` float32 tensors that exercise the hard cases.

    Values come from a small grid, so rows contain ties; zeros appear; one weight may carry a
    dominant sensitivity (like notebook 02's row 58, column 148); and a row's Fisher values may
    all be zero, which triggers the unweighted fallback.

    :param draw: Hypothesis draw function.
    :param max_rows: Largest ``R``.
    :param min_n: Smallest ``n``.
    :param max_n: Largest ``n``.
    :return: Weight and Fisher tensors on the CPU.
    """
    num_rows = draw(st.integers(1, max_rows))
    n = draw(st.integers(min_n, max_n))
    grid = st.integers(-20, 20).map(lambda k: k / 8.0)
    sensitivities = st.sampled_from([0.0, 1e-8, 0.5, 1.0, 3.0, 1e4])

    # Build each row independently so one row's special case does not constrain another.

    weight_rows: list[list[float]] = []
    fisher_rows: list[list[float]] = []
    for _ in range(num_rows):
        weight_rows.append(draw(st.lists(grid, min_size=n, max_size=n)))

        # Each Fisher style targets one hard case: the unweighted fallback, float64 prefix-sum
        # cancellation next to one huge sensitivity, or a mix that includes exact zeros.

        style = draw(st.sampled_from(["mixed", "all_zero", "dominant"]))
        if style == "all_zero":
            fisher_rows.append([0.0] * n)
        elif style == "dominant":
            spike = draw(st.integers(0, n - 1))
            fisher_rows.append([1e6 if j == spike else 1e-3 for j in range(n)])
        else:
            fisher_rows.append(draw(st.lists(sensitivities, min_size=n, max_size=n)))
    return torch.tensor(weight_rows), torch.tensor(fisher_rows)
