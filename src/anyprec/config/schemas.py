"""Pydantic schemas for every Hydra config group (ADR 0002, spec 0002).

Every schema is frozen and rejects unknown keys, so a typo in YAML or on the command line is a
validation error rather than a silently ignored field. The schemas are also the single source
of truth for test fixtures.
"""

import re
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, model_validator

from anyprec.utils.dtypes import DTypeName

type QuantizerMode = Literal["incremental", "standalone"]
type EvalDatasetName = Literal["wikitext2", "c4"]

BitWidth = Annotated[int, Field(ge=1, le=8)]


class FrozenModel(BaseModel):
    """Base for every config, manifest, and results schema: immutable, and unknown keys are rejected.

    It is public so the artifact manifests (spec 0006) and the results schema (spec 0008) share
    the exact same validation rules as the configs.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class QuantizableModules(FrozenModel):
    """Which linear layers to quantize, and how many there must be.

    :param pattern: Regular expression matched against qualified module names.
    :param expected_count: Required number of matches; a mismatch is a hard error.
    """

    pattern: str
    expected_count: PositiveInt

    @model_validator(mode="after")
    def _pattern_compiles(self) -> Self:
        """Reject patterns that are not valid regular expressions."""
        try:
            re.compile(self.pattern)
        except re.error as error:
            raise ValueError(f"invalid quantizable_modules.pattern: {error}") from error
        return self


class ModelConfig(FrozenModel):
    """The Hugging Face checkpoint and its quantizable modules (ADR 0002).

    :param model_id: Hub repository ID.
    :param revision: Hub revision, pinned to a commit hash.
    :param dtype: Dtype used for Fisher estimation.
    :param eval_dtype: Dtype used for simulated inference and evaluation.
    :param quantizable_modules: Module selection rule.
    """

    model_id: str
    revision: str
    dtype: DTypeName
    eval_dtype: DTypeName
    quantizable_modules: QuantizableModules


class CalibrationConfig(FrozenModel):
    """Calibration source and sampling rule (ADR 0002).

    :param path: ``datasets.load_dataset`` path.
    :param data_files: Optional data files mapping.
    :param name: Optional dataset configuration name.
    :param split: Split to sample from.
    :param text_field: Column holding raw text.
    :param num_sequences: Number of calibration sequences ``N``.
    :param seq_len: Tokens per sequence ``T``.
    :param seed: Seed of the document permutation.
    """

    path: str
    data_files: dict[str, str] | None = None
    name: str | None = None
    split: str
    text_field: str
    num_sequences: PositiveInt
    seq_len: PositiveInt
    seed: int


class QuantizerConfig(FrozenModel):
    """Fisher-weighted k-means settings (ADR 0003, Section 7).

    :param mode: ``incremental`` (nested, the any-precision method) or ``standalone`` (baseline).
    :param seed_bits: Seed bit-width ``b_0``.
    :param parent_bits: Parent bit-width ``B``; ``uint8`` indices cap it at 8.
    :param seed: Base k-means++ seed, expanded per module and per row chunk.
    :param lloyd_max_iter: Lloyd iteration cap.
    :param empty_eps: Segment mass at or below which a segment counts as empty.
    :param row_chunk: Rows per kernel call; bounds peak memory.
    """

    mode: QuantizerMode
    seed_bits: BitWidth
    parent_bits: BitWidth
    seed: int
    lloyd_max_iter: PositiveInt
    empty_eps: PositiveFloat
    row_chunk: PositiveInt

    @model_validator(mode="after")
    def _bit_range(self) -> Self:
        """Require ``seed_bits <= parent_bits``."""
        if self.seed_bits > self.parent_bits:
            raise ValueError(
                f"seed_bits ({self.seed_bits}) must not exceed parent_bits ({self.parent_bits})"
            )
        return self


class RotationNone(FrozenModel):
    """No rotation before clustering (ADR 0006)."""

    kind: Literal["none"]


class RotationHadamard(FrozenModel):
    """Randomized Hadamard rotation before clustering; validated but not implemented (ADR 0006).

    :param axis: Rotation axis; only ``in_features`` is defined.
    :param randomized_signs: Whether to apply random sign flips.
    :param seed: Seed of the per-module sign vectors.
    """

    kind: Literal["hadamard"]
    axis: Literal["in_features"]
    randomized_signs: bool
    seed: int


type RotationConfig = Annotated[RotationNone | RotationHadamard, Field(discriminator="kind")]


class OutputConfig(FrozenModel):
    """Output locations.

    :param base_dir: Root of Hydra run directories and the cache.
    :param cache_dir: Root of the artifact cache.
    """

    base_dir: Path
    cache_dir: Path


class EvalDatasetConfig(FrozenModel):
    """One evaluation text source (ADR 0004, Metric 2).

    :param path: ``datasets.load_dataset`` path.
    :param name: Optional dataset configuration name.
    :param data_files: Optional data files mapping.
    :param split: Split to evaluate.
    :param text_field: Column holding raw text.
    :param joiner: String placed between documents before tokenizing.
    :param max_tokens: Optional truncation of the token stream.
    """

    path: str
    name: str | None = None
    data_files: dict[str, str] | None = None
    split: str
    text_field: str
    joiner: str
    max_tokens: PositiveInt | None = None


class KLConfig(FrozenModel):
    """KL divergence settings (ADR 0004, Metric 1).

    :param dataset: Dataset on which KL and top-1 agreement are measured.
    :param quantile: Reported upper quantile of per-token KL.
    """

    dataset: EvalDatasetName
    quantile: float = Field(gt=0.0, lt=1.0)


class EvalConfig(FrozenModel):
    """Evaluation protocol (ADR 0004).

    :param chunk_len: Tokens per evaluation chunk.
    :param max_chunks: Optional cap on chunks per dataset, for smoke runs.
    :param lm_head_chunk_tokens: Positions per LM-head slice, a power of two; ``None`` applies
        the head to the whole chunk at once (spec 0008).
    :param datasets: Evaluation text sources.
    :param kl: KL settings.
    :param bits: Bit-widths to evaluate, ascending and unique.
    """

    chunk_len: PositiveInt
    max_chunks: PositiveInt | None = None
    lm_head_chunk_tokens: PositiveInt | None = 256
    datasets: dict[EvalDatasetName, EvalDatasetConfig]
    kl: KLConfig
    bits: list[BitWidth]

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        """Check the slice length, the KL dataset, and the bit-width list."""
        slice_len = self.lm_head_chunk_tokens
        if slice_len is not None and slice_len & (slice_len - 1) != 0:
            raise ValueError(
                f"lm_head_chunk_tokens must be a power of two or null, got {slice_len}"
            )
        if self.kl.dataset not in self.datasets:
            raise ValueError(
                f"kl.dataset {self.kl.dataset!r} is not one of the configured datasets"
            )
        if not self.bits or self.bits != sorted(set(self.bits)):
            raise ValueError(f"bits must be non-empty, ascending, and unique, got {self.bits}")
        return self


class QuantizeRunConfig(FrozenModel):
    """Validated configuration of one quantization run.

    :param seed: Global seed for ``random``, NumPy, and PyTorch.
    :param device: Device name.
    :param model: Model settings.
    :param calibration: Calibration settings.
    :param quantizer: Quantizer settings.
    :param rotation: Rotation settings.
    :param output: Output locations.
    """

    seed: int
    device: str
    model: ModelConfig
    calibration: CalibrationConfig
    quantizer: QuantizerConfig
    rotation: RotationConfig
    output: OutputConfig


class EvaluateRunConfig(FrozenModel):
    """Validated configuration of one evaluation run.

    :param seed: Global seed for ``random``, NumPy, and PyTorch.
    :param device: Device name.
    :param modes: Quantizer modes whose artifacts are evaluated.
    :param model: Model settings.
    :param calibration: Calibration settings, needed to recompute the Fisher key.
    :param quantizer: Quantizer settings, needed to recompute the artifact keys.
    :param rotation: Rotation settings.
    :param eval: Evaluation protocol.
    :param output: Output locations.
    """

    seed: int
    device: str
    modes: list[QuantizerMode]
    model: ModelConfig
    calibration: CalibrationConfig
    quantizer: QuantizerConfig
    rotation: RotationConfig
    eval: EvalConfig
    output: OutputConfig

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        """Require unique modes and evaluated bit-widths inside the quantizer's range."""
        if not self.modes or len(set(self.modes)) != len(self.modes):
            raise ValueError(f"modes must be non-empty and unique, got {self.modes}")
        low, high = self.quantizer.seed_bits, self.quantizer.parent_bits
        outside = [b for b in self.eval.bits if not low <= b <= high]
        if outside:
            raise ValueError(f"eval.bits {outside} lie outside the quantizer range [{low}, {high}]")
        return self
