"""Tests for quantizable-module discovery (spec 0003)."""

import pytest
from transformers import GraniteMoeHybridForCausalLM

from anyprec.config.schemas import ModelConfig, QuantizableModules
from anyprec.models.discovery import QuantizableModuleError, find_quantizable_linears
from tests import factories

_EXPECTED_LAYER_ORDER: list[str] = [
    "shared_mlp.input_linear",
    "shared_mlp.output_linear",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
]


def test_finds_the_twelve_targets_in_named_modules_order(
    tiny_model: GraniteMoeHybridForCausalLM, tiny_model_config: ModelConfig
) -> None:
    """
    Given: the tiny Granite model, whose tied LM head is also an nn.Linear.
    When: quantizable linears are discovered with the tiny pattern.
    Then: exactly the 12 attention and shared-MLP projections are returned, layer by layer in
        named_modules() order, and the LM head is not among them.
    """
    found = find_quantizable_linears(tiny_model, tiny_model_config.quantizable_modules)

    expected = [f"model.layers.{i}.{suffix}" for i in range(2) for suffix in _EXPECTED_LAYER_ORDER]
    assert list(found) == expected
    assert all(module is tiny_model.get_submodule(name) for name, module in found.items())


def test_raises_when_the_count_differs_from_the_configuration(
    tiny_model: GraniteMoeHybridForCausalLM,
) -> None:
    """
    Given: a configuration expecting 13 modules for a pattern that matches 12.
    When: modules are discovered.
    Then: QuantizableModuleError reports both counts.
    """
    cfg = QuantizableModules(pattern=factories.TINY_PATTERN, expected_count=13)

    with pytest.raises(QuantizableModuleError, match="expected 13 quantizable linears, found 12"):
        find_quantizable_linears(tiny_model, cfg)


def test_raises_when_two_targets_share_one_weight_tensor(
    tiny_model: GraniteMoeHybridForCausalLM, tiny_model_config: ModelConfig
) -> None:
    """
    Given: the tiny model with layer 0's k_proj tied to its v_proj weight (both [32, 64]).
    When: modules are discovered.
    Then: QuantizableModuleError is raised, since one tensor would be quantized twice.
    """
    attention = tiny_model.get_submodule("model.layers.0.self_attn")
    attention.get_submodule("k_proj").weight = attention.get_submodule("v_proj").weight

    with pytest.raises(QuantizableModuleError, match="share one weight"):
        find_quantizable_linears(tiny_model, tiny_model_config.quantizable_modules)
