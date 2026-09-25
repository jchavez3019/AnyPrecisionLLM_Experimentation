"""KL divergence, top-1 agreement, and perplexity, computed one LM-head slice at a time (spec 0008).

Everything here is pure ``torch``: a model enters only as final-norm hidden states and a head
function, so the arithmetic is unit-tested without loading one.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

type LogitHead = Callable[[torch.Tensor], torch.Tensor]

_QUANTILE_LIMIT: int = 2**24


@dataclass(frozen=True)
class ChunkOutputs:
    """One model's output for one chunk, before the LM head.

    :param hidden: Final-norm hidden states ``[T, H]``.
    :param head: Maps ``[S, H]`` to logits ``[S, V]``, including any logit scaling.
    """

    hidden: torch.Tensor
    head: LogitHead


@dataclass(frozen=True)
class ChunkMetrics:
    """Metrics of one chunk for the evaluated model.

    :param kl: Per-position ``KL(p_ref || p_q)``, float32 ``[T]`` on the CPU, or ``None``
        without a reference.
    :param agree: Per-position top-1 agreement, bool ``[T]`` on the CPU, or ``None``.
    :param mean_nll: Mean next-token NLL over the chunk's ``T - 1`` targets.
    :param num_tokens: Positions in the chunk, ``T``.
    """

    kl: torch.Tensor | None
    agree: torch.Tensor | None
    mean_nll: float
    num_tokens: int


def chunk_metrics(
    q: ChunkOutputs, ref: ChunkOutputs | None, tokens: torch.Tensor, slice_len: int | None
) -> ChunkMetrics:
    """Per-position KL, top-1 agreement, and the chunk's mean next-token NLL for model ``q``.

    Position ``t`` predicts token ``t + 1``, so the last position has no NLL target, while KL
    and agreement use all ``T`` positions. Only one slice of logits per model exists at a time.

    :param q: The evaluated model's outputs.
    :param ref: The reference model's outputs, or ``None`` for perplexity-only datasets.
    :param tokens: int64 ``[T]`` on the hidden states' device.
    :param slice_len: Positions per head application, ``eval.lm_head_chunk_tokens``; ``None``
        processes all ``T`` positions at once.
    :return: The chunk's metrics.
    :raises ValueError: If ``T < 2`` or the tokens and hidden states disagree in length.
    """
    num_tokens = tokens.shape[0]
    if num_tokens < 2 or q.hidden.shape[0] != num_tokens:
        raise ValueError(f"need T >= 2 tokens matching hidden [T, H], got {num_tokens} tokens")

    # A null setting is one slice spanning the whole chunk, so both modes share one code path.

    step = num_tokens if slice_len is None else slice_len
    kl_parts: list[torch.Tensor] = []
    agree_parts: list[torch.Tensor] = []
    nll_sum = 0.0
    with torch.no_grad():
        for s in range(0, num_tokens, step):
            # Apply the LM head to [S, H] -> [S, V], then work in float32 log-probabilities.

            e = min(s + step, num_tokens)
            q_logp = q.head(q.hidden[s:e]).float().log_softmax(dim=-1)

            # Next-token NLL for positions whose target lies inside the chunk: [S', 1] gathered
            # from [S, V], with S' = S, or S - 1 on the last slice.

            targets = tokens[s + 1 : e + 1]
            picked = q_logp[: targets.shape[0]].gather(1, targets[:, None])
            nll_sum -= float(picked.sum())
            if ref is None:
                continue

            # KL(p_ref || p_q) per position, reduced over the vocabulary: [S, V] -> [S].

            ref_logp = ref.head(ref.hidden[s:e]).float().log_softmax(dim=-1)
            kl_parts.append((ref_logp.exp() * (ref_logp - q_logp)).sum(dim=-1).cpu())
            agree_parts.append((ref_logp.argmax(dim=-1) == q_logp.argmax(dim=-1)).cpu())

    # [S] slices -> [T] per-position values.

    return ChunkMetrics(
        kl=torch.cat(kl_parts) if kl_parts else None,
        agree=torch.cat(agree_parts) if agree_parts else None,
        mean_nll=nll_sum / (num_tokens - 1),
        num_tokens=num_tokens,
    )


@dataclass(frozen=True)
class MetricSummary:
    """Metrics of one (model, dataset) pair over all its chunks.

    :param perplexity: ``exp`` of the mean of the per-chunk mean NLLs (ADR 0004).
    :param mean_nll: Mean of the per-chunk mean NLLs.
    :param num_chunks: Chunks evaluated.
    :param num_tokens: Positions evaluated.
    :param kl_mean: Mean KL over all positions, or ``None`` without a reference.
    :param kl_quantile: KL at the configured quantile over all positions, or ``None``.
    :param top1_agreement: Fraction of positions whose argmax agrees, or ``None``.
    """

    perplexity: float
    mean_nll: float
    num_chunks: int
    num_tokens: int
    kl_mean: float | None
    kl_quantile: float | None
    top1_agreement: float | None


class StreamingMetrics:
    """Collect chunk metrics for one (model, dataset) pair.

    Per-position KL values stay on the CPU so the quantile is exact rather than estimated.

    :param quantile: The KL quantile to report, ``eval.kl.quantile``.
    """

    def __init__(self, quantile: float) -> None:
        self._quantile = quantile
        self._nlls: list[float] = []
        self._num_tokens = 0
        self._kl: list[torch.Tensor] = []
        self._agree: list[torch.Tensor] = []
        self._missing_reference = False

    def update(self, chunk: ChunkMetrics) -> None:
        """Add one chunk's metrics.

        :param chunk: Output of ``chunk_metrics``.
        """
        self._nlls.append(chunk.mean_nll)
        self._num_tokens += chunk.num_tokens
        if chunk.kl is None or chunk.agree is None:
            self._missing_reference = True
            return
        self._kl.append(chunk.kl)
        self._agree.append(chunk.agree)

    def summary(self) -> MetricSummary:
        """Reduce the collected chunks.

        :return: The summary; KL fields are ``None`` if any chunk lacked a reference.
        :raises ValueError: If no chunk was added, or there are more KL values than
            ``torch.quantile`` accepts.
        """
        if not self._nlls:
            raise ValueError("no chunks were added")
        mean_nll = sum(self._nlls) / len(self._nlls)
        kl_mean: float | None = None
        kl_quantile: float | None = None
        agreement: float | None = None

        # Concatenate per-position values across chunks: C x [T] -> [C * T].

        if not self._missing_reference:
            kl = torch.cat(self._kl)
            if kl.numel() > _QUANTILE_LIMIT:
                raise ValueError(f"{kl.numel()} KL values exceed torch.quantile's 2**24 limit")
            kl_mean = float(kl.double().mean())
            kl_quantile = float(torch.quantile(kl, self._quantile))
            agreement = float(torch.cat(self._agree).double().mean())
        return MetricSummary(
            perplexity=math.exp(mean_nll),
            mean_nll=mean_nll,
            num_chunks=len(self._nlls),
            num_tokens=self._num_tokens,
            kl_mean=kl_mean,
            kl_quantile=kl_quantile,
            top1_agreement=agreement,
        )
