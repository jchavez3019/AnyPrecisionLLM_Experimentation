"""Tests for analytic bits per weight (spec 0008)."""

import pytest

from anyprec.evaluation.bits import bits_report, layer_bits_per_weight, parent_bits_per_weight

_GRANITE_LAYER: list[tuple[int, int]] = [
    (1024, 1024),
    (256, 1024),
    (256, 1024),
    (1024, 1024),
    (4096, 1024),
    (1024, 2048),
]
_GRANITE_SHAPES: list[tuple[int, int]] = _GRANITE_LAYER * 28
_GRANITE_TOTAL: int = 352_379_904

_GRANITE_TABLE: dict[int, tuple[float, float]] = {
    3: (3.1103, 6.8713),
    4: (4.2206, 7.6576),
    5: (5.4412, 8.5221),
    6: (6.8824, 9.5427),
    7: (8.7647, 10.8758),
    8: (11.5294, 12.8339),
}


def test_per_layer_figures_match_the_adr_table_for_1024_wide_rows() -> None:
    """
    Given: 1024-wide rows.
    When: layer and parent bits per weight are computed.
    Then: 3 bits gives 3.125, 8 bits gives 12.0, and the 3-to-8 parent gives 15.875 (ADR 0004).
    """
    assert layer_bits_per_weight(3, 1024) == 3.125
    assert layer_bits_per_weight(8, 1024) == 12.0
    assert parent_bits_per_weight(3, 8, 1024) == 15.875


def test_granite_report_reproduces_the_spec_table_to_four_decimals() -> None:
    """
    Given: Granite 4.0 350M's 168 quantized shapes and 352,379,904 total parameters.
    When: the bits report for 3 to 8 bits is built.
    Then: 249,561,088 parameters are quantized, and every per-bits and parent figure matches
        spec 0008 to four decimals, on the quantized layers and on the whole model.
    """
    report = bits_report(_GRANITE_SHAPES, _GRANITE_TOTAL, 3, 8)

    assert report.quantized_params == 249_561_088
    for bits, (layers, whole) in _GRANITE_TABLE.items():
        assert round(report.per_bits[bits], 4) == layers
        assert round(report.per_bits_whole_model[bits], 4) == whole
    assert round(report.parent, 4) == 14.9485
    assert round(report.parent_whole_model, 4) == 15.2553


@pytest.mark.parametrize(("shapes", "total"), [([], 10), ([(4, 4)], 15)])
def test_report_rejects_no_quantized_layers_or_more_than_the_model_holds(
    shapes: list[tuple[int, int]], total: int
) -> None:
    """
    Given: no quantized shapes, or 16 quantized parameters in a 15-parameter model.
    When: the bits report is built.
    Then: ValueError is raised instead of a division by zero or a negative remainder.
    """
    with pytest.raises(ValueError, match="quantized parameters"):
        bits_report(shapes, total, 2, 4)
