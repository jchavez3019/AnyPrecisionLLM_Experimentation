"""Tests for evaluation token streams and chunking (spec 0003)."""

import torch

from anyprec.config.schemas import EvalDatasetConfig
from anyprec.data.calibration import Encoder
from anyprec.data.evaluation_text import iter_chunks, load_eval_tokens


def _dataset(max_tokens: int | None) -> EvalDatasetConfig:
    """An in-memory evaluation dataset joined by blank lines."""
    return EvalDatasetConfig(
        path="in-memory", split="test", text_field="text", joiner="\n\n", max_tokens=max_tokens
    )


def test_without_a_budget_the_stream_is_the_joined_text_tokenized_once(
    char_encoder: Encoder,
) -> None:
    """
    Given: three documents and no token budget.
    When: the evaluation stream is built.
    Then: it equals encoding the documents joined by the joiner, as int64.
    """
    texts = ["first", "second doc", "third"]

    tokens = load_eval_tokens(texts, char_encoder, _dataset(None))

    assert tokens.dtype == torch.long
    assert torch.equal(tokens, torch.tensor(char_encoder("first\n\nsecond doc\n\nthird")))


def test_budget_is_filled_across_document_blocks_as_a_prefix_of_the_full_stream(
    char_encoder: Encoder,
) -> None:
    """
    Given: 2,500 ten-byte documents, so the first 1,000-document block holds about 12,000 tokens.
    When: a budget of 15,000 tokens is requested, which needs a second block.
    Then: exactly 15,000 tokens are returned, and they are a prefix of the whole joined stream.
    """
    texts = [f"doc{i:06d}." for i in range(2500)]
    full = char_encoder("\n\n".join(texts))

    tokens = load_eval_tokens(texts, char_encoder, _dataset(15_000))

    assert torch.equal(tokens, torch.tensor(full[:15_000]))


def test_budget_larger_than_the_corpus_returns_the_whole_stream(char_encoder: Encoder) -> None:
    """
    Given: two short documents and a budget far above their length.
    When: the evaluation stream is built.
    Then: every token of the joined text is returned, and nothing more.
    """
    tokens = load_eval_tokens(["ab", "cd"], char_encoder, _dataset(1000))

    assert torch.equal(tokens, torch.tensor(char_encoder("ab\n\ncd")))


def test_chunks_drop_the_remainder_and_are_views_of_the_stream() -> None:
    """
    Given: a stream of 23 tokens.
    When: it is cut into 5-token chunks.
    Then: 4 chunks of shape [1, 5] cover tokens 0..19, and each shares the stream's storage.
    """
    tokens = torch.arange(23)

    chunks = list(iter_chunks(tokens, 5, None))

    assert [tuple(c.shape) for c in chunks] == [(1, 5)] * 4
    assert torch.equal(torch.cat(chunks, dim=1)[0], tokens[:20])
    assert all(
        c.untyped_storage().data_ptr() == tokens.untyped_storage().data_ptr() for c in chunks
    )


def test_chunks_stop_at_max_chunks() -> None:
    """
    Given: a stream with room for 4 chunks.
    When: at most 2 are requested.
    Then: only the first 2 chunks are yielded.
    """
    chunks = list(iter_chunks(torch.arange(23), 5, 2))

    assert [c[0, 0].item() for c in chunks] == [0, 5]
