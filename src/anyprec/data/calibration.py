"""Calibration sampling and the tokenizer boundary (ADR 0002, spec 0003)."""

from collections.abc import Callable, Sequence
from typing import cast

import torch
from transformers import PreTrainedTokenizerBase

from anyprec.config.schemas import CalibrationConfig

type Encoder = Callable[[str], list[int]]


class CalibrationError(ValueError):
    """Too few documents are long enough to fill the calibration set."""


def make_encoder(tokenizer: PreTrainedTokenizerBase) -> Encoder:
    """Wrap a tokenizer as raw-text encoding with no special tokens (ADR 0002).

    :param tokenizer: A Hugging Face tokenizer.
    :return: A function from text to token ids.
    """

    def encode(text: str) -> list[int]:
        """Tokenize raw text without adding BOS, EOS, or other special tokens."""
        # The tokenizer's __call__ is untyped; this cast is the single place it is narrowed.

        ids = cast(object, tokenizer(text, add_special_tokens=False)["input_ids"])
        return cast("list[int]", ids)

    return encode


def sample_calibration(
    texts: Sequence[str], encode: Encoder, cfg: CalibrationConfig
) -> torch.Tensor:
    """Sample calibration sequences with ADR 0002's rule, identical to notebook 02.

    Documents are visited in a seeded permutation; each one shorter than ``seq_len`` tokens is
    skipped, and the first ``seq_len`` tokens of each accepted document are kept.

    :param texts: Raw documents.
    :param encode: Text-to-token-ids function.
    :param cfg: Number of sequences, their length, and the seed.
    :return: int64 token ids ``[cfg.num_sequences, cfg.seq_len]`` on the CPU.
    :raises CalibrationError: If fewer than ``num_sequences`` documents are long enough.
    """
    # A CPU generator makes the document order identical on every machine.

    generator = torch.Generator().manual_seed(cfg.seed)
    order = [int(index) for index in torch.randperm(len(texts), generator=generator)]
    accepted: list[torch.Tensor] = []
    for index in order:
        ids = encode(texts[index])
        if len(ids) < cfg.seq_len:
            continue
        accepted.append(torch.tensor(ids[: cfg.seq_len], dtype=torch.long))
        if len(accepted) == cfg.num_sequences:
            break
    if len(accepted) < cfg.num_sequences:
        raise CalibrationError(
            f"only {len(accepted)} of {len(texts)} documents have at least {cfg.seq_len} tokens; "
            f"{cfg.num_sequences} are needed"
        )

    # num_sequences x [seq_len] -> [num_sequences, seq_len]

    return torch.stack(accepted)
