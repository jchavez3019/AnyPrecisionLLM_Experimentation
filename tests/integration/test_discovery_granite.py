"""Module discovery on the real Granite checkpoint (spec 0011); downloads allowed."""

import pytest
import torch

from anyprec.models.discovery import find_quantizable_linears
from anyprec.models.loading import load_model
from tests.integration.conftest import compose_quantize

pytestmark = pytest.mark.network

GRANITE_LAYERS: int = 28
GRANITE_PARAMS: int = 352_379_904

# [out_features, in_features] per block (ADR 0002), in the order named_modules() visits them.

GRANITE_SHAPES: dict[str, tuple[int, int]] = {
    "shared_mlp.input_linear": (4096, 1024),
    "shared_mlp.output_linear": (1024, 2048),
    "self_attn.q_proj": (1024, 1024),
    "self_attn.k_proj": (256, 1024),
    "self_attn.v_proj": (256, 1024),
    "self_attn.o_proj": (1024, 1024),
}


def test_granite_discovery_finds_168_linears_in_layer_order_with_the_adr_shapes() -> None:
    """
    Given: the pinned Granite 4.0 350M checkpoint, loaded on the CPU in bfloat16.
    When: the shipped quantizable_modules pattern is applied.
    Then: exactly 168 linears are found, layer by layer, each with its ADR 0002 shape, and the
        model has 352,379,904 parameters with the tied embedding counted once.
    """
    cfg = compose_quantize([]).model

    model = load_model(cfg, torch.bfloat16, torch.device("cpu"))
    targets = find_quantizable_linears(model, cfg.quantizable_modules)

    expected = {
        f"model.layers.{layer}.{suffix}": shape
        for layer in range(GRANITE_LAYERS)
        for suffix, shape in GRANITE_SHAPES.items()
    }
    assert {name: tuple(linear.weight.shape) for name, linear in targets.items()} == expected
    assert list(targets) == list(expected)
    assert sum(p.numel() for p in model.parameters()) == GRANITE_PARAMS
