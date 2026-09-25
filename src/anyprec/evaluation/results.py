"""The ``results.json`` schema and the builders that fill it from metric summaries (spec 0008)."""

from collections.abc import Mapping
from datetime import datetime
from typing import Self

from pydantic import Field, PositiveInt, model_validator

from anyprec.config.schemas import (
    BitWidth,
    EvalDatasetName,
    EvaluateRunConfig,
    FrozenModel,
    QuantizerMode,
)
from anyprec.evaluation.bits import BitsReport
from anyprec.evaluation.metrics import MetricSummary

RESULTS_SCHEMA_VERSION: int = 1


class PerplexityResult(FrozenModel):
    """Perplexity of one model on one dataset.

    :param dataset: Dataset name.
    :param perplexity: ``exp(mean_nll)``.
    :param mean_nll: Mean of the per-chunk mean NLLs.
    :param num_chunks: Chunks evaluated.
    :param num_tokens: Positions evaluated.
    """

    dataset: EvalDatasetName
    perplexity: float
    mean_nll: float
    num_chunks: PositiveInt
    num_tokens: PositiveInt


class ReferenceEntry(FrozenModel):
    """Metrics of the unquantized reference model.

    :param perplexity: One result per dataset.
    """

    perplexity: list[PerplexityResult]


class QuantizedEntry(FrozenModel):
    """Metrics of one (mode, bit-width) pair.

    :param mode: Quantizer mode.
    :param bits: Bit-width.
    :param artifact_key: Full key of the evaluated quantized artifact.
    :param perplexity: One result per dataset.
    :param kl_dataset: Dataset on which the KL fields were measured.
    :param kl_mean: Mean KL over all positions.
    :param kl_quantile: KL at the configured quantile.
    :param top1_agreement: Fraction of positions whose argmax agrees with the reference.
    :param bits_per_weight: Over the quantized layers.
    :param bits_per_weight_whole_model: Over all parameters.
    """

    mode: QuantizerMode
    bits: BitWidth
    artifact_key: str
    perplexity: list[PerplexityResult]
    kl_dataset: EvalDatasetName
    kl_mean: float
    kl_quantile: float
    top1_agreement: float = Field(ge=0.0, le=1.0)
    bits_per_weight: float
    bits_per_weight_whole_model: float


class Results(FrozenModel):
    """One evaluation run: self-contained, with the resolved config and the reference metrics.

    :param schema_version: ``RESULTS_SCHEMA_VERSION`` at write time.
    :param config: The resolved run config.
    :param model_id: Hub repository ID.
    :param revision: Hub revision.
    :param fisher_key: Full key of the Fisher both modes were built from.
    :param versions: Library versions.
    :param created_at: Creation time, UTC.
    :param bits: Analytic bits per weight.
    :param reference: Reference-model metrics.
    :param entries: One entry per (mode, bits), ordered by mode then bits.
    :param seconds: Wall-clock time of the evaluation.
    """

    schema_version: int
    config: EvaluateRunConfig
    model_id: str
    revision: str
    fisher_key: str
    versions: dict[str, str]
    created_at: datetime
    bits: BitsReport
    reference: ReferenceEntry
    entries: list[QuantizedEntry]
    seconds: float

    @model_validator(mode="after")
    def _entries_match_config(self) -> Self:
        """Require every entry's mode, bit-width, and KL dataset to come from the config."""
        for entry in self.entries:
            if entry.mode not in self.config.modes or entry.bits not in self.config.eval.bits:
                raise ValueError(f"entry ({entry.mode}, {entry.bits}) is not in the config")
            if entry.kl_dataset != self.config.eval.kl.dataset:
                raise ValueError(f"entry KL dataset {entry.kl_dataset!r} differs from the config")
        return self


def _perplexities(summaries: Mapping[EvalDatasetName, MetricSummary]) -> list[PerplexityResult]:
    """Convert per-dataset summaries into perplexity results, in mapping order."""
    return [
        PerplexityResult(
            dataset=name,
            perplexity=s.perplexity,
            mean_nll=s.mean_nll,
            num_chunks=s.num_chunks,
            num_tokens=s.num_tokens,
        )
        for name, s in summaries.items()
    ]


def make_reference_entry(summaries: Mapping[EvalDatasetName, MetricSummary]) -> ReferenceEntry:
    """Build the reference entry from its per-dataset summaries.

    :param summaries: Dataset name to the reference model's summary.
    :return: The entry.
    """
    return ReferenceEntry(perplexity=_perplexities(summaries))


def make_quantized_entry(
    mode: QuantizerMode,
    bits: int,
    artifact_key: str,
    summaries: Mapping[EvalDatasetName, MetricSummary],
    kl_dataset: EvalDatasetName,
    report: BitsReport,
) -> QuantizedEntry:
    """Build one (mode, bits) entry: perplexity everywhere, KL fields from the KL dataset.

    :param mode: Quantizer mode.
    :param bits: Bit-width.
    :param artifact_key: Full key of the evaluated artifact.
    :param summaries: Dataset name to the quantized model's summary.
    :param kl_dataset: The dataset whose summary holds the KL fields.
    :param report: Bits per weight of this model.
    :return: The entry.
    :raises ValueError: If the KL dataset's summary has no KL values, which only a loop that
        skipped the reference can cause.
    """
    kl = summaries[kl_dataset]
    if kl.kl_mean is None or kl.kl_quantile is None or kl.top1_agreement is None:
        raise ValueError(f"the {kl_dataset} summary has no KL values")
    return QuantizedEntry(
        mode=mode,
        bits=bits,
        artifact_key=artifact_key,
        perplexity=_perplexities(summaries),
        kl_dataset=kl_dataset,
        kl_mean=kl.kl_mean,
        kl_quantile=kl.kl_quantile,
        top1_agreement=kl.top1_agreement,
        bits_per_weight=report.per_bits[bits],
        bits_per_weight_whole_model=report.per_bits_whole_model[bits],
    )
