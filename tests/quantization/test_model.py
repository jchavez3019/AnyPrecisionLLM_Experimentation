"""Tests for whole-model quantization (spec 0005)."""

import re
from collections.abc import Iterator
from typing import cast

import torch
from torch import nn
from transformers import GraniteMoeHybridForCausalLM

from anyprec.config.schemas import QuantizerConfig
from anyprec.quantization.model import quantize_model
from tests import factories


def _targets(model: nn.Module) -> dict[str, nn.Linear]:
    """Select the tiny model's quantizable linears in module order.

    :param model: The tiny Granite model.
    :return: Qualified name to linear layer.
    """
    pattern = re.compile(factories.TINY_PATTERN)
    modules = cast("Iterator[tuple[str, nn.Module]]", model.named_modules())
    return {
        name: module
        for name, module in modules
        if pattern.match(name) and isinstance(module, nn.Linear)
    }


def _random_fisher(targets: dict[str, nn.Linear]) -> dict[str, torch.Tensor]:
    """Draw a positive Fisher diagonal per target from a local generator.

    :param targets: Name to linear layer.
    :return: Name to ``[m, n]`` Fisher tensor.
    """
    generator = torch.Generator().manual_seed(0)
    return {
        name: torch.rand(linear.weight.shape, generator=generator) + 1e-3
        for name, linear in targets.items()
    }


def test_quantize_model_covers_targets_in_order_and_leaves_weights_untouched(
    tiny_model: GraniteMoeHybridForCausalLM, quantizer_config: QuantizerConfig
) -> None:
    """
    Given: the tiny model's 12 target linears and a random positive Fisher.
    When: the whole model is quantized.
    Then: results and progress callbacks follow discovery order, and every model parameter is
        bitwise unchanged.
    """
    targets = _targets(tiny_model)
    before = {name: p.detach().clone() for name, p in tiny_model.named_parameters()}
    seen: list[str] = []

    result = quantize_model(
        targets, _random_fisher(targets), quantizer_config, torch.device("cpu"), seen.append
    )

    assert len(targets) == factories.TINY_QUANTIZABLE_COUNT
    assert list(result.layers) == list(targets) == seen
    assert all(torch.equal(before[name], p) for name, p in tiny_model.named_parameters())
    for name, linear in targets.items():
        assert (
            result.layers[name].luts[quantizer_config.parent_bits].shape[0] == linear.out_features
        )


def test_quantize_model_results_do_not_depend_on_module_order(
    tiny_model: GraniteMoeHybridForCausalLM,
) -> None:
    """
    Given: the tiny model's targets in discovery order and in reverse order.
    When: both orders are quantized.
    Then: every module gets bitwise-identical codebooks, because its seed depends only on its name.
    """
    targets = _targets(tiny_model)
    fisher = _random_fisher(targets)
    cfg = factories.quantizer_config()
    reverse = dict(reversed(list(targets.items())))

    forward_result = quantize_model(targets, fisher, cfg, torch.device("cpu"))
    reverse_result = quantize_model(reverse, fisher, cfg, torch.device("cpu"))

    for name in targets:
        assert torch.equal(forward_result.layers[name].luts[4], reverse_result.layers[name].luts[4])
        assert torch.equal(
            forward_result.layers[name].indices[4], reverse_result.layers[name].indices[4]
        )
