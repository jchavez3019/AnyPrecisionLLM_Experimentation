"""Tests for calibration sampling and the tokenizer boundary (spec 0003)."""

import json

import pytest
import torch
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

from anyprec.config.schemas import CalibrationConfig
from anyprec.data.calibration import (
    CalibrationError,
    Encoder,
    make_encoder,
    sample_calibration,
)
from tests import factories

# A word-level tokenizer in the tokenizers JSON format whose post-processor prepends <s>, like
# tokenizers that add a BOS token by default. It is built from JSON because the builder classes
# of the tokenizers package are untyped.

_BOS_WORD_TOKENIZER: dict[str, object] = {
    "version": "1.0",
    "added_tokens": [
        {
            "id": 0,
            "content": "<s>",
            "special": True,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
        }
    ],
    "pre_tokenizer": {"type": "Whitespace"},
    "post_processor": {
        "type": "TemplateProcessing",
        "single": [
            {"SpecialToken": {"id": "<s>", "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}},
        ],
        "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 0}}],
        "special_tokens": {"<s>": {"id": "<s>", "ids": [0], "tokens": ["<s>"]}},
    },
    "model": {
        "type": "WordLevel",
        "vocab": {"<s>": 0, "<unk>": 1, "alpha": 2, "beta": 3},
        "unk_token": "<unk>",
    },
}


def _config(num_sequences: int = 4, seed: int = 0) -> CalibrationConfig:
    """The tiny calibration settings (32-token sequences) with overrides."""
    return factories.calibration_config().model_copy(
        update={"num_sequences": num_sequences, "seed": seed}
    )


def test_sampling_skips_short_documents_and_keeps_each_accepted_prefix(
    char_encoder: Encoder,
) -> None:
    """
    Given: 12 documents, a third of them shorter than 32 bytes.
    When: 4 sequences of 32 tokens are sampled.
    Then: the result is int64 [4, 32], and every row is the first 32 tokens of a distinct
        document that is at least 32 tokens long.
    """
    texts = factories.corpus()
    encoded = [char_encoder(t) for t in texts]
    long_prefixes = torch.tensor([ids[:32] for ids in encoded if len(ids) >= 32])

    tokens = sample_calibration(texts, char_encoder, _config())

    # Row-by-prefix equality: [4, 1, 32] against [1, K, 32] -> [4, K] matches per row.

    matches = (tokens[:, None, :] == long_prefixes[None, :, :]).all(dim=2)
    assert tokens.shape == (4, 32)
    assert tokens.dtype == torch.long
    assert bool((matches.sum(dim=1) == 1).all())
    assert int(matches.any(dim=0).sum()) == 4


def test_sampling_is_reproducible_per_seed_and_changes_with_the_seed(
    char_encoder: Encoder,
) -> None:
    """
    Given: the same documents and settings.
    When: sampling runs twice with seed 0 and once with seed 1.
    Then: both seed-0 samples are identical and the seed-1 sample differs.
    """
    texts = factories.corpus()

    first = sample_calibration(texts, char_encoder, _config(seed=0))
    second = sample_calibration(texts, char_encoder, _config(seed=0))
    other = sample_calibration(texts, char_encoder, _config(seed=1))

    assert torch.equal(first, second)
    assert not torch.equal(first, other)


def test_sampling_raises_when_too_few_documents_are_long_enough(char_encoder: Encoder) -> None:
    """
    Given: 12 documents of which 8 are at least 32 tokens long.
    When: 9 sequences are requested.
    Then: CalibrationError reports how many qualified.
    """
    with pytest.raises(CalibrationError, match="only 8 of 12 documents"):
        sample_calibration(factories.corpus(), char_encoder, _config(num_sequences=9))


def test_encoder_drops_the_special_tokens_the_tokenizer_would_add() -> None:
    """
    Given: an in-memory word-level tokenizer whose post-processor prepends a BOS token.
    When: text is encoded through make_encoder.
    Then: only the word ids are returned, since calibration and evaluation use raw text.
    """
    backend = Tokenizer.from_str(json.dumps(_BOS_WORD_TOKENIZER))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, bos_token="<s>", unk_token="<unk>"
    )
    assert tokenizer("alpha beta")["input_ids"] == [0, 2, 3]

    assert make_encoder(tokenizer)("alpha beta gamma") == [2, 3, 1]
