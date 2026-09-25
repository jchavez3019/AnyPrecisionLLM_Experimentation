"""Evaluation token streams and their chunks (ADR 0004, spec 0003)."""

from collections.abc import Iterator, Sequence

import torch

from anyprec.config.schemas import EvalDatasetConfig
from anyprec.data.calibration import Encoder

_DOCUMENT_BLOCK: int = 1000


def load_eval_tokens(texts: Sequence[str], encode: Encoder, cfg: EvalDatasetConfig) -> torch.Tensor:
    """Join documents with ``cfg.joiner``, tokenize once, and truncate to ``cfg.max_tokens``.

    :param texts: Raw documents, in dataset order.
    :param encode: Text-to-token-ids function.
    :param cfg: The joiner and the optional token budget.
    :return: int64 token ids ``[num_tokens]`` on the CPU.
    """
    if cfg.max_tokens is None:
        return torch.tensor(encode(cfg.joiner.join(texts)), dtype=torch.long)

    # Tokenizing the whole C4 validation shard is wasteful when only 2**19 tokens are needed.
    # Grow the joined prefix in blocks of documents until it is long enough, then truncate.
    # Block boundaries do not change the result once the budget is reached.

    count = 0
    while True:
        count = min(count + _DOCUMENT_BLOCK, len(texts))
        ids = encode(cfg.joiner.join(texts[:count]))
        if len(ids) >= cfg.max_tokens or count == len(texts):
            return torch.tensor(ids[: cfg.max_tokens], dtype=torch.long)


def iter_chunks(
    tokens: torch.Tensor, chunk_len: int, max_chunks: int | None
) -> Iterator[torch.Tensor]:
    """Yield non-overlapping chunks of a token stream; the trailing remainder is dropped.

    :param tokens: int64 ``[num_tokens]``.
    :param chunk_len: Tokens per chunk.
    :param max_chunks: Upper bound on the number of chunks, or ``None`` for all.
    :return: An iterator of ``[1, chunk_len]`` views of ``tokens``.
    """
    num_chunks = tokens.numel() // chunk_len
    if max_chunks is not None:
        num_chunks = min(num_chunks, max_chunks)

    # [num_tokens] -> [chunk_len] slice -> [1, chunk_len], a batch of one sequence.

    for i in range(num_chunks):
        yield tokens[i * chunk_len : (i + 1) * chunk_len].unsqueeze(0)
