"""Manifest and statistics schemas of cached artifacts (ADR 0005, spec 0006)."""

from datetime import datetime
from typing import Literal

from pydantic import PositiveInt

from anyprec.config.schemas import FrozenModel, QuantizerMode
from anyprec.utils.hashing import JsonValue


class ModuleEntry(FrozenModel):
    """One quantizable module, as recorded in a manifest.

    :param name: Qualified module name.
    :param shape: Weight shape ``(m, n)``.
    """

    name: str
    shape: tuple[PositiveInt, PositiveInt]


class ManifestBase(FrozenModel):
    """Fields shared by both artifact kinds.

    :param schema_version: Artifact schema version (spec 0002).
    :param key: Full 64-hex SHA-256 of ``config_snapshot``.
    :param model_id: Hub repository ID.
    :param revision: Hub revision.
    :param modules: Modules in discovery order.
    :param config_snapshot: The exact configuration the artifact depends on.
    :param device: Device type that produced the artifact, ``"cuda"`` or ``"cpu"``.
    :param versions: Library versions at creation time.
    :param created_at: Creation time, UTC.
    """

    schema_version: int
    key: str
    model_id: str
    revision: str
    modules: list[ModuleEntry]
    config_snapshot: dict[str, JsonValue]
    device: str
    versions: dict[str, str]
    created_at: datetime


class FisherManifest(ManifestBase):
    """Manifest of a Fisher artifact.

    :param kind: Always ``"fisher"``.
    :param num_sequences: Calibration sequences ``N``.
    :param seq_len: Tokens per sequence ``T``.
    :param mean_loss: Mean of the per-sequence calibration NLLs.
    :param seconds: Wall-clock time of the accumulation loop.
    """

    kind: Literal["fisher"]
    num_sequences: PositiveInt
    seq_len: PositiveInt
    mean_loss: float
    seconds: float


class QuantizedManifest(ManifestBase):
    """Manifest of a quantized artifact.

    :param kind: Always ``"quantized"``.
    :param mode: Quantizer mode.
    :param seed_bits: Seed bit-width.
    :param parent_bits: Parent bit-width.
    :param parent_key: Full key of the Fisher artifact it was built from.
    :param seconds: Wall-clock time of the k-means loop.
    """

    kind: Literal["quantized"]
    mode: QuantizerMode
    seed_bits: int
    parent_bits: int
    parent_key: str
    seconds: float


class ArtifactStats(FrozenModel):
    """Per-module diagnostics of a quantized artifact (``stats.json``).

    :param relative_error: Module to bit-width to ``J / sum(f w^2)``.
    :param lloyd_iterations: Module to its largest Lloyd iteration count.
    """

    relative_error: dict[str, dict[int, float]]
    lloyd_iterations: dict[str, int]
