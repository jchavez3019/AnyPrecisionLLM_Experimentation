"""Calibration sampling from the real C4 shard (spec 0011); downloads allowed."""

import pytest
import torch

from anyprec.data.calibration import make_encoder, sample_calibration
from anyprec.data.hub import load_texts
from anyprec.models.loading import load_tokenizer
from tests.integration.conftest import NUM_SEQUENCES, compose_quantize

pytestmark = pytest.mark.network

GRANITE_VOCAB: int = 100_352


def test_c4_calibration_gives_reproducible_in_vocabulary_sequences() -> None:
    """
    Given: the shipped C4 calibration config, reduced to 8 sequences, and the Granite tokenizer.
    When: the calibration set is sampled twice from the same loaded texts.
    Then: both samples are the same [8, 512] int64 tensor, and every id is inside the vocabulary.
    """
    cfg = compose_quantize([f"calibration.num_sequences={NUM_SEQUENCES}"])
    cal = cfg.calibration
    texts = load_texts(cal.path, cal.name, cal.data_files, cal.split, cal.text_field)
    encode = make_encoder(load_tokenizer(cfg.model))

    first = sample_calibration(texts, encode, cal)
    second = sample_calibration(texts, encode, cal)

    assert first.shape == (NUM_SEQUENCES, cal.seq_len)
    assert first.dtype == torch.int64
    assert torch.equal(first, second)
    assert int(first.min()) >= 0
    assert int(first.max()) < GRANITE_VOCAB
