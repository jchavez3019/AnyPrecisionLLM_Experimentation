"""Tests for the loss, body, and sliced LM head of a causal LM (spec 0003)."""

from types import SimpleNamespace
from typing import cast

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from transformers import GraniteMoeHybridForCausalLM
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

from anyprec.models.heads import (
    SlicedLogitsError,
    body_hidden_states,
    causal_lm_loss,
    check_sliced_logits,
    logit_head,
)

_TOKENS: torch.Tensor = torch.arange(3, 3 + 7 * 12, 7)[None, :] % 256


@pytest.fixture
def scaled_model(tiny_model: GraniteMoeHybridForCausalLM) -> GraniteMoeHybridForCausalLM:
    """The tiny model with Granite's production ``logits_scaling`` of 4."""
    tiny_model.config.logits_scaling = 4.0
    return tiny_model


def _logits(model: GraniteMoeHybridForCausalLM, tokens: torch.Tensor) -> torch.Tensor:
    """Full-forward logits ``[1, T, V]`` of the tiny model."""
    logits = model(input_ids=tokens, use_cache=False).logits
    assert isinstance(logits, torch.Tensor)
    return logits


def test_causal_lm_loss_is_the_mean_nll_of_the_next_token_and_carries_gradients(
    scaled_model: GraniteMoeHybridForCausalLM,
) -> None:
    """
    Given: 12 tokens and the tiny model with logits_scaling 4.
    When: the causal LM loss is computed.
    Then: it equals the mean cross-entropy of positions 0..10 predicting tokens 1..11, and it
        is attached to the autograd graph so Fisher estimation can differentiate it.
    """
    loss = causal_lm_loss(scaled_model, _TOKENS)

    # Shift by one: logits [T - 1, V] at positions 0..T-2 against targets [T - 1] at 1..T-1.

    with torch.no_grad():
        logits = _logits(scaled_model, _TOKENS)[0, :-1]
        expected = F.cross_entropy(logits, _TOKENS[0, 1:])
    assert loss.shape == ()
    assert loss.requires_grad
    torch.testing.assert_close(loss.detach(), expected)


def test_logit_head_divides_the_tied_projection_by_logits_scaling(
    scaled_model: GraniteMoeHybridForCausalLM,
) -> None:
    """
    Given: the tiny model with logits_scaling 4 and 5 random hidden states.
    When: the logit head is applied.
    Then: the result is the output-embedding projection divided by 4, shaped [5, 256].
    """
    hidden = torch.randn(5, 64, generator=torch.Generator().manual_seed(0))
    projection = cast(object, scaled_model.get_output_embeddings())
    assert isinstance(projection, nn.Linear)

    with torch.no_grad():
        logits = logit_head(scaled_model)(hidden)
        expected = projection(hidden) / 4.0

    assert logits.shape == (5, 256)
    torch.testing.assert_close(logits, expected)


@pytest.mark.parametrize("slice_len", [None, 1, 5, 12])
def test_sliced_logits_match_forward_for_any_slice_length(
    scaled_model: GraniteMoeHybridForCausalLM, slice_len: int | None
) -> None:
    """
    Given: the tiny model with logits_scaling 4 and 12 tokens.
    When: sliced logits are checked with slices of all positions, 1, 5 (not a divisor), or 12.
    Then: no error is raised.
    """
    check_sliced_logits(scaled_model, _TOKENS, slice_len)


def _soft_cap(module: nn.Module, args: tuple[object, ...], output: object) -> SimpleNamespace:
    """Forward hook: replace the model's logits with a soft cap at 0.1, as some LMs do."""
    assert isinstance(output, MoeCausalLMOutputWithPast)
    assert isinstance(output.logits, torch.Tensor)
    return SimpleNamespace(logits=0.1 * torch.tanh(output.logits / 0.1))


def test_sliced_logits_detect_a_forward_that_soft_caps_its_logits(
    scaled_model: GraniteMoeHybridForCausalLM,
) -> None:
    """
    Given: a model whose forward soft-caps logits after the head, which body plus head omits.
    When: sliced logits are checked.
    Then: SlicedLogitsError reports the difference.
    """
    handle = scaled_model.register_forward_hook(_soft_cap)

    try:
        with pytest.raises(SlicedLogitsError, match="differ from forward"):
            check_sliced_logits(scaled_model, _TOKENS, 5)
    finally:
        handle.remove()


def test_head_refuses_a_null_logits_scaling(
    scaled_model: GraniteMoeHybridForCausalLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Given: a model whose config sets logits_scaling to None, which the strict config accepts.
    When: the logit head is built.
    Then: SlicedLogitsError names the bad value instead of producing unscaled logits.
    """
    monkeypatch.setattr(scaled_model.config, "logits_scaling", None)

    with pytest.raises(SlicedLogitsError, match="logits_scaling is NoneType"):
        logit_head(scaled_model)


def test_head_refuses_output_embeddings_that_are_not_linear(
    scaled_model: GraniteMoeHybridForCausalLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Given: a model whose output embeddings are an nn.Identity.
    When: the logit head is built.
    Then: SlicedLogitsError is raised, since the head could not be sliced by position.
    """
    monkeypatch.setattr(scaled_model, "get_output_embeddings", nn.Identity)

    with pytest.raises(SlicedLogitsError, match=r"not nn\.Linear"):
        logit_head(scaled_model)


def test_body_refuses_a_model_without_a_decoder_module(
    scaled_model: GraniteMoeHybridForCausalLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Given: a model whose get_decoder() returns None.
    When: the body's hidden states are requested.
    Then: SlicedLogitsError is raised instead of an AttributeError deep inside the call.
    """
    monkeypatch.setattr(scaled_model, "get_decoder", lambda: None)

    with pytest.raises(SlicedLogitsError, match="not a module"):
        body_hidden_states(scaled_model, _TOKENS)
