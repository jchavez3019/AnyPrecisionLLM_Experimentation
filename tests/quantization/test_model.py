"""Tests for whole-model quantization (spec 0005)."""

import re
from collections.abc import Iterator
from typing import cast

import pytest
import torch
from torch import nn
from transformers import GraniteMoeHybridForCausalLM

from anyprec.config.schemas import QuantizerConfig
from anyprec.quantization.model import quantize_model
from tests import factories


def _target_weights(model: nn.Module) -> dict[str, torch.Tensor]:
    """Select the tiny model's quantizable weight matrices in module order.

    :param model: The tiny Granite model.
    :return: Qualified module name to its live ``[m, n]`` weight parameter.
    """
    pattern = re.compile(factories.TINY_PATTERN)
    modules = cast("Iterator[tuple[str, nn.Module]]", model.named_modules())
    return {
        name: module.weight
        for name, module in modules
        if pattern.match(name) and isinstance(module, nn.Linear)
    }


def _random_fisher(weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Draw a positive Fisher diagonal per weight from a local generator.

    :param weights: Name to weight matrix.
    :return: Name to ``[m, n]`` Fisher tensor.
    """
    generator = torch.Generator().manual_seed(0)
    return {
        name: torch.rand(weight.shape, generator=generator) + 1e-3
        for name, weight in weights.items()
    }


def test_quantize_model_covers_targets_in_order_and_leaves_weights_untouched(
    tiny_model: GraniteMoeHybridForCausalLM, quantizer_config: QuantizerConfig
) -> None:
    """
    Given: the tiny model's 12 target weights (the live parameters) and a random positive Fisher.
    When: the whole model is quantized.
    Then: results and progress callbacks follow discovery order, each parent LUT has one row
        per output feature, and every model parameter is bitwise unchanged.
    """
    weights = _target_weights(tiny_model)
    before = {name: p.detach().clone() for name, p in tiny_model.named_parameters()}
    seen: list[str] = []

    result = quantize_model(
        weights, _random_fisher(weights), quantizer_config, torch.device("cpu"), seen.append
    )

    # Order and coverage, then row counts, then the read-only guarantee on the live parameters.

    assert len(weights) == factories.TINY_QUANTIZABLE_COUNT
    assert list(result.layers) == list(weights) == seen
    for name, weight in weights.items():
        assert result.layers[name].luts[quantizer_config.parent_bits].shape[0] == weight.shape[0]
    assert all(torch.equal(before[name], p) for name, p in tiny_model.named_parameters())


def test_quantize_model_results_do_not_depend_on_module_order(
    tiny_model: GraniteMoeHybridForCausalLM,
) -> None:
    """
    Given: the tiny model's target weights in discovery order and in reverse order.
    When: both orders are quantized.
    Then: every module gets bitwise-identical codebooks, because its seed depends only on its name.
    """
    weights = _target_weights(tiny_model)
    fisher = _random_fisher(weights)
    cfg = factories.quantizer_config()
    reverse = dict(reversed(list(weights.items())))

    forward_result = quantize_model(weights, fisher, cfg, torch.device("cpu"))
    reverse_result = quantize_model(reverse, fisher, cfg, torch.device("cpu"))

    for name in weights:
        forward_layer, reverse_layer = forward_result.layers[name], reverse_result.layers[name]
        assert torch.equal(forward_layer.luts[4], reverse_layer.luts[4])
        assert torch.equal(forward_layer.indices[4], reverse_layer.indices[4])


def test_quantize_model_raises_when_fisher_shape_differs_from_weight() -> None:
    """
    Given: a [32, 16] weight whose Fisher diagonal is transposed to [16, 32].
    When: the model is quantized.
    Then: ValueError names the module and both shapes, before any kernel runs.
    """
    weights = {"layer.q_proj": torch.randn(32, 16)}
    fisher = {"layer.q_proj": torch.rand(16, 32)}

    with pytest.raises(ValueError, match=r"layer\.q_proj: Fisher shape \(16, 32\)"):
        quantize_model(weights, fisher, factories.quantizer_config(), torch.device("cpu"))
