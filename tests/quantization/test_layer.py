"""Tests for whole-matrix quantization (spec 0005)."""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from anyprec.config.schemas import QuantizerConfig, QuantizerMode
from anyprec.quantization.layer import LayerQuantization, quantize_layer
from tests import factories
from tests.quantization import oracles
from tests.strategies import weighted_rows

LAYER_ROWS = weighted_rows(max_rows=6, min_n=16, max_n=40)
MODES: list[QuantizerMode] = ["incremental", "standalone"]


def _relative_error_of(
    weight: torch.Tensor, fisher: torch.Tensor, lut: torch.Tensor, ids: torch.Tensor
) -> float:
    """Recompute ``J / sum(f w^2)`` from stored indices and a float16 codebook.

    :param weight: Weights ``[m, n]``.
    :param fisher: Raw Fisher diagonal ``[m, n]``.
    :param lut: Codebooks ``[m, K]``.
    :param ids: Indices ``[m, n]``.
    :return: The relative weighted reconstruction error, or 0 for an all-zero matrix.
    """
    f = oracles.conditioned_fisher(weight, fisher)
    energy = float((f * weight.double().square()).sum())
    cost = oracles.reconstruction_cost(weight, f, lut, ids)
    return cost / energy if energy > 0.0 else cost


def _assert_consistent(
    result: LayerQuantization, weight: torch.Tensor, fisher: torch.Tensor, cfg: QuantizerConfig
) -> None:
    """Check dtypes, ranges, sortedness, nesting, and recorded errors of one result.

    :param result: Output of ``quantize_layer``.
    :param weight: Its weights.
    :param fisher: Its Fisher diagonal.
    :param cfg: Its quantizer settings.
    """
    m, n = weight.shape
    for bits in range(cfg.seed_bits, cfg.parent_bits + 1):
        lut = result.luts[bits]
        assert lut.dtype == torch.float16 and lut.shape == (m, 2**bits)
        assert bool((lut[:, 1:] >= lut[:, :-1]).all())

        # Incremental artifacts store only the parent; lower widths come from the right shift.

        stored = result.indices.get(bits)
        if stored is None:
            stored = result.indices[cfg.parent_bits] >> (cfg.parent_bits - bits)
        assert stored.dtype == torch.uint8 and stored.shape == (m, n)
        assert int(stored.max()) < 2**bits
        recomputed = _relative_error_of(weight, fisher, lut, stored)
        assert recomputed == pytest.approx(result.relative_error[bits], rel=1e-6, abs=1e-12)


@settings(max_examples=25)
@given(LAYER_ROWS, st.sampled_from(MODES), st.sampled_from([1, 4, 16]))
def test_quantize_layer_outputs_are_consistent_for_every_mode_and_row_chunk(
    case: tuple[torch.Tensor, torch.Tensor], mode: QuantizerMode, row_chunk: int
) -> None:
    """
    Given: a matrix, a mode, and a row_chunk that may split the rows into several chunks.
    When: the matrix is quantized from 2 to 4 bits.
    Then: indices are uint8 below 2**b, LUTs are float16 and non-decreasing, the right-shift
        identity reproduces each recorded relative error, and the right widths are stored.
    """
    weight, fisher = case
    cfg = factories.quantizer_config(mode=mode, row_chunk=row_chunk)

    result = quantize_layer(weight, fisher, cfg, generator_seed=5)

    expected_stored = [4] if mode == "incremental" else [2, 3, 4]
    assert sorted(result.indices) == expected_stored
    _assert_consistent(result, weight, fisher, cfg)


@settings(max_examples=25)
@given(LAYER_ROWS)
def test_incremental_relative_error_is_non_increasing_in_bits(
    case: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """
    Given: a matrix quantized incrementally.
    When: relative errors are compared across bit-widths.
    Then: each extra bit never increases the error, up to float16 codebook rounding.
    """
    weight, fisher = case

    errors = quantize_layer(weight, fisher, factories.quantizer_config(), 5).relative_error

    assert errors[3] <= errors[2] * (1 + 1e-3) + 1e-5
    assert errors[4] <= errors[3] * (1 + 1e-3) + 1e-5


@settings(max_examples=25)
@given(LAYER_ROWS, st.integers(0, 2**31))
def test_incremental_seed_level_equals_standalone_fit_bitwise(
    case: tuple[torch.Tensor, torch.Tensor], seed: int
) -> None:
    """
    Given: the same matrix and module seed in both modes.
    When: it is quantized incrementally and standalone.
    Then: the seed-level LUTs are bitwise equal, and the incremental parent indices shifted down
        to the seed width equal the standalone seed-width indices.
    """
    weight, fisher = case

    incremental = quantize_layer(weight, fisher, factories.quantizer_config("incremental"), seed)
    standalone = quantize_layer(weight, fisher, factories.quantizer_config("standalone"), seed)

    assert torch.equal(incremental.luts[2], standalone.luts[2])
    assert torch.equal(incremental.indices[4] >> 2, standalone.indices[2])


def test_quantize_layer_is_bitwise_deterministic(quantizer_config: QuantizerConfig) -> None:
    """
    Given: a random matrix and Fisher from a seeded local generator.
    When: it is quantized twice with the same config and module seed.
    Then: every index tensor, LUT, and error is identical.
    """
    generator = torch.Generator().manual_seed(3)
    weight = torch.randn(40, 32, generator=generator)
    fisher = torch.rand(40, 32, generator=generator)

    first = quantize_layer(weight, fisher, quantizer_config, 11)
    second = quantize_layer(weight, fisher, quantizer_config, 11)

    assert first.relative_error == second.relative_error
    assert all(torch.equal(first.luts[b], second.luts[b]) for b in first.luts)
    assert all(torch.equal(first.indices[b], second.indices[b]) for b in first.indices)


def test_quantize_layer_rejects_rows_shorter_than_parent_codebook(
    quantizer_config: QuantizerConfig,
) -> None:
    """
    Given: a matrix with 8 columns and a 4-bit parent (16 centroids).
    When: it is quantized.
    Then: ValueError is raised, since a row cannot fill 16 distinct centroids.
    """
    with pytest.raises(ValueError, match="cannot hold 16"):
        quantize_layer(torch.randn(4, 8), torch.ones(4, 8), quantizer_config, 0)
