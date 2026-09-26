"""Tests for sliced KL, agreement, NLL, and their streaming reduction (spec 0008)."""

import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from anyprec.evaluation.metrics import (
    ChunkMetrics,
    ChunkOutputs,
    StreamingMetrics,
    chunk_metrics,
)

_T: int = 11
_H: int = 16
_V: int = 50


def _head(seed: int) -> nn.Linear:
    """A seeded random ``[H] -> [V]`` LM head, drawn from a local generator."""
    head = nn.Linear(_H, _V, bias=False)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        head.weight.copy_(torch.randn(_V, _H, generator=generator) / _H**0.5)
    return head


def _outputs(hidden: torch.Tensor, head: nn.Linear) -> ChunkOutputs:
    """Wrap hidden states and a head, scaled by 4 like Granite's logits."""

    def apply(h: torch.Tensor) -> torch.Tensor:
        """Project ``[S, H]`` to scaled ``[S, V]`` logits."""
        return head(h) / 4.0

    return ChunkOutputs(hidden=hidden, head=apply)


@pytest.fixture
def pair() -> tuple[ChunkOutputs, ChunkOutputs, torch.Tensor]:
    """Reference and quantized outputs for one chunk that differ only in the head, plus tokens."""
    generator = torch.Generator().manual_seed(0)
    hidden = torch.randn(_T, _H, generator=generator)
    tokens = torch.randint(0, _V, (_T,), generator=generator)
    return _outputs(hidden, _head(1)), _outputs(hidden, _head(2)), tokens


def _full_logp(outputs: ChunkOutputs) -> torch.Tensor:
    """Log-probabilities of every position at once: ``[T, V]``."""
    with torch.no_grad():
        return outputs.head(outputs.hidden).float().log_softmax(dim=-1)


def test_identical_models_have_zero_kl_and_full_agreement(
    pair: tuple[ChunkOutputs, ChunkOutputs, torch.Tensor],
) -> None:
    """
    Given: the same outputs used as both the reference and the evaluated model.
    When: chunk metrics are computed in slices of 4.
    Then: every position's KL is within 1e-6 of zero, and every argmax agrees.
    """
    ref, _, tokens = pair

    metrics = chunk_metrics(ref, ref, tokens, 4)

    assert metrics.kl is not None and metrics.agree is not None
    assert float(metrics.kl.abs().max()) <= 1e-6
    assert bool(metrics.agree.all())


@pytest.mark.parametrize("slice_len", [None, 1, 3, 4, _T, 16])
def test_sliced_metrics_equal_the_full_logit_references_for_any_slice_length(
    pair: tuple[ChunkOutputs, ChunkOutputs, torch.Tensor], slice_len: int | None
) -> None:
    """
    Given: two models that differ in their heads, and 11 tokens.
    When: chunk metrics are computed whole, or in slices of 1, 3, 4, 11, or 16 positions.
    Then: KL equals kl_div on the full logits, agreement equals the full argmax comparison,
        and the mean NLL equals cross-entropy with labels shifted by one, which is what
        transformers' labels=input_ids loss computes (checked on the model in test_heads).
    """
    ref, q, tokens = pair
    q_logp, ref_logp = _full_logp(q), _full_logp(ref)

    metrics = chunk_metrics(q, ref, tokens, slice_len)

    # References on full [T, V] log-probabilities; KL is reduced over the vocabulary to [T].

    expected_kl = F.kl_div(q_logp, ref_logp, log_target=True, reduction="none").sum(dim=-1)
    with torch.no_grad():
        logits = q.head(q.hidden)

    # Position t predicts token t + 1: [T - 1, V] logits against [T - 1] targets.

    expected_nll = F.cross_entropy(logits[:-1], tokens[1:])
    assert metrics.kl is not None and metrics.agree is not None
    torch.testing.assert_close(metrics.kl, expected_kl, rtol=1e-5, atol=1e-6)
    assert torch.equal(metrics.agree, ref_logp.argmax(dim=-1) == q_logp.argmax(dim=-1))
    assert metrics.mean_nll == pytest.approx(float(expected_nll), rel=1e-5)
    assert metrics.num_tokens == _T


def test_without_a_reference_only_the_nll_is_computed(
    pair: tuple[ChunkOutputs, ChunkOutputs, torch.Tensor],
) -> None:
    """
    Given: a perplexity-only dataset, with no reference outputs.
    When: chunk metrics are computed.
    Then: KL and agreement are None, and the NLL is still reported.
    """
    _, q, tokens = pair

    metrics = chunk_metrics(q, None, tokens, 4)

    assert metrics.kl is None and metrics.agree is None
    assert metrics.mean_nll > 0.0


def test_chunk_metrics_reject_a_chunk_without_a_next_token_target(
    pair: tuple[ChunkOutputs, ChunkOutputs, torch.Tensor],
) -> None:
    """
    Given: a single-token chunk.
    When: chunk metrics are computed.
    Then: ValueError is raised, since the mean over T - 1 targets is undefined.
    """
    _, q, tokens = pair
    single = ChunkOutputs(hidden=q.hidden[:1], head=q.head)

    with pytest.raises(ValueError, match="T >= 2"):
        chunk_metrics(single, None, tokens[:1], None)


def _chunk(mean_nll: float, kl: list[float] | None, num_tokens: int) -> ChunkMetrics:
    """A hand-built chunk result; agreement is True where KL is below 0.5."""
    kl_tensor = None if kl is None else torch.tensor(kl)
    agree = None if kl_tensor is None else kl_tensor < 0.5
    return ChunkMetrics(kl=kl_tensor, agree=agree, mean_nll=mean_nll, num_tokens=num_tokens)


def test_perplexity_is_exp_of_the_mean_of_per_chunk_means_not_of_tokens() -> None:
    """
    Given: a 3-token chunk with mean NLL 1 and an 11-token chunk with mean NLL 3.
    When: the stream is summarized.
    Then: perplexity is exp(2), the documented per-chunk mean, not exp(32 / 12) per token.
    """
    stream = StreamingMetrics(quantile=0.5)
    stream.update(_chunk(1.0, None, 3))
    stream.update(_chunk(3.0, None, 11))

    summary = stream.summary()

    assert summary.perplexity == pytest.approx(math.exp(2.0))
    assert summary.perplexity != pytest.approx(math.exp(32 / 12))
    assert (summary.num_chunks, summary.num_tokens) == (2, 14)
    assert summary.kl_mean is None and summary.top1_agreement is None


def test_kl_fields_reduce_over_every_position_of_every_chunk() -> None:
    """
    Given: two chunks of per-position KL values.
    When: the stream is summarized at the 0.75 quantile.
    Then: the mean, the torch.quantile, and the agreement are over all positions together.
    """
    stream = StreamingMetrics(quantile=0.75)
    stream.update(_chunk(2.0, [0.1, 0.9, 0.2], 3))
    stream.update(_chunk(2.0, [0.4, 0.6], 2))
    everything = torch.tensor([0.1, 0.9, 0.2, 0.4, 0.6])

    summary = stream.summary()

    assert summary.kl_mean == pytest.approx(float(everything.mean()))
    assert summary.kl_quantile == pytest.approx(float(torch.quantile(everything, 0.75)))
    assert summary.top1_agreement == pytest.approx(3 / 5)


def test_kl_fields_are_none_if_any_chunk_lacked_a_reference() -> None:
    """
    Given: one chunk with KL values and one without.
    When: the stream is summarized.
    Then: the KL fields are None rather than a mean over part of the dataset.
    """
    stream = StreamingMetrics(quantile=0.5)
    stream.update(_chunk(2.0, [0.1, 0.2], 2))
    stream.update(_chunk(2.0, None, 2))

    assert stream.summary().kl_quantile is None


def test_summary_refuses_an_empty_stream_and_an_inexact_quantile() -> None:
    """
    Given: an empty stream, and a stream with 2**24 + 1 KL values.
    When: each is summarized.
    Then: both raise ValueError; the second instead of silently subsampling.
    """
    with pytest.raises(ValueError, match="no chunks"):
        StreamingMetrics(quantile=0.5).summary()

    stream = StreamingMetrics(quantile=0.5)
    size = 2**24 + 1
    stream.update(
        ChunkMetrics(
            kl=torch.zeros(size),
            agree=torch.ones(size, dtype=torch.bool),
            mean_nll=1.0,
            num_tokens=size,
        )
    )
    with pytest.raises(ValueError, match="2\\*\\*24"):
        stream.summary()
