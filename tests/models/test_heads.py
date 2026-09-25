"""Tests for the loss, body, and sliced LM head of a causal LM (spec 0003)."""

from collections.abc import Callable
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


def _null_scaling(model: GraniteMoeHybridForCausalLM, monkeypatch: pytest.MonkeyPatch) -> None:
    """Break the model: a null logits_scaling, which the strict config still accepts."""
    monkeypatch.setattr(model.config, "logits_scaling", None)


def _identity_head(model: GraniteMoeHybridForCausalLM, monkeypatch: pytest.MonkeyPatch) -> None:
    """Break the model: output embeddings that are not an nn.Linear."""
    monkeypatch.setattr(model, "get_output_embeddings", nn.Identity)


def _no_decoder(model: GraniteMoeHybridForCausalLM, monkeypatch: pytest.MonkeyPatch) -> None:
    """Break the model: get_decoder() returns no module."""
    monkeypatch.setattr(model, "get_decoder", lambda: None)


def _build_head(model: GraniteMoeHybridForCausalLM) -> object:
    """Build the head half of the split."""
    return logit_head(model)


def _run_body(model: GraniteMoeHybridForCausalLM) -> object:
    """Run the body half of the split."""
    return body_hidden_states(model, _TOKENS)


type _Breaker = Callable[[GraniteMoeHybridForCausalLM, pytest.MonkeyPatch], None]


@pytest.mark.parametrize(
    ("break_model", "split", "message"),
    [
        (_null_scaling, _build_head, "logits_scaling is NoneType"),
        (_identity_head, _build_head, "not nn.Linear"),
        (_no_decoder, _run_body, "not a module"),
    ],
)
def test_split_refuses_a_model_that_is_not_body_plus_linear_head(
    scaled_model: GraniteMoeHybridForCausalLM,
    monkeypatch: pytest.MonkeyPatch,
    break_model: _Breaker,
    split: Callable[[GraniteMoeHybridForCausalLM], object],
    message: str,
) -> None:
    """
    Given: a model with a null logits_scaling, a non-linear head, or no decoder body.
    When: the matching half of the body/head split is built.
    Then: SlicedLogitsError names what is missing instead of producing wrong logits.
    """
    break_model(scaled_model, monkeypatch)

    with pytest.raises(SlicedLogitsError, match=message):
        split(scaled_model)
