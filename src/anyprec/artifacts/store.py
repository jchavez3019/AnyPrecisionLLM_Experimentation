"""Content-addressed, atomically written artifact cache (ADR 0005, spec 0006).

Fisher diagonals are a get-or-compute cache; quantized artifacts are write-once, read-many.
Every lookup either returns an artifact that matches the request exactly, or raises.
"""

import os
import shutil
import tempfile
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

import torch
from pydantic import ValidationError
from safetensors.torch import load, save

from anyprec.artifacts.keys import (
    FISHER_SCHEMA_VERSION,
    QUANTIZED_SCHEMA_VERSION,
    fisher_key,
)
from anyprec.artifacts.manifest import (
    ArtifactStats,
    FisherManifest,
    ManifestBase,
    ModuleEntry,
    QuantizedManifest,
    module_entries,
)
from anyprec.config.schemas import QuantizerMode, QuantizeRunConfig
from anyprec.quantization.model import ModelQuantization
from anyprec.sensitivity.fisher import FisherResult
from anyprec.utils.hashing import JsonValue, sha256_key
from anyprec.utils.versions import library_versions

_MANIFEST: str = "manifest.json"
_STATS: str = "stats.json"
_CREATE_HINT: str = (
    "run quantization/quantize_any_precision.py with the same model, calibration, quantizer, "
    "and rotation settings to create it"
)

type _Shape = tuple[int, ...]


class ArtifactNotFoundError(FileNotFoundError):
    """No artifact exists for the requested key."""


class ArtifactMismatchError(RuntimeError):
    """An artifact exists for the key but does not match the request."""


@dataclass(frozen=True)
class FisherMeta:
    """Fisher manifest fields the store cannot derive from the result.

    :param model_id: Hub repository ID.
    :param revision: Hub revision.
    :param device: Device type that computed the Fisher.
    :param num_sequences: Calibration sequences ``N``.
    :param seq_len: Tokens per sequence ``T``.
    """

    model_id: str
    revision: str
    device: str
    num_sequences: int
    seq_len: int

    @classmethod
    def from_config(cls, cfg: QuantizeRunConfig, device: torch.device) -> Self:
        """Build the metadata of a quantization run.

        :param cfg: The validated run config.
        :param device: The resolved compute device.
        :return: The Fisher metadata.
        """
        return cls(
            model_id=cfg.model.model_id,
            revision=cfg.model.revision,
            device=device.type,
            num_sequences=cfg.calibration.num_sequences,
            seq_len=cfg.calibration.seq_len,
        )


@dataclass(frozen=True)
class QuantizedMeta:
    """Quantized manifest fields the store cannot derive from the result.

    :param model_id: Hub repository ID.
    :param revision: Hub revision.
    :param device: Device type that ran the k-means.
    :param mode: Quantizer mode.
    :param seed_bits: Seed bit-width.
    :param parent_bits: Parent bit-width.
    :param parent_key: Full key of the Fisher artifact.
    """

    model_id: str
    revision: str
    device: str
    mode: QuantizerMode
    seed_bits: int
    parent_bits: int
    parent_key: str

    @classmethod
    def from_config(cls, cfg: QuantizeRunConfig, device: torch.device) -> Self:
        """Build the metadata of a quantization run.

        :param cfg: The validated run config.
        :param device: The resolved compute device.
        :return: The quantized-artifact metadata.
        """
        return cls(
            model_id=cfg.model.model_id,
            revision=cfg.model.revision,
            device=device.type,
            mode=cfg.quantizer.mode,
            seed_bits=cfg.quantizer.seed_bits,
            parent_bits=cfg.quantizer.parent_bits,
            parent_key=fisher_key(cfg.model, cfg.calibration, cfg.rotation),
        )


@dataclass(frozen=True)
class QuantizedArtifact:
    """A loaded quantized artifact; every tensor is on the CPU (ADR 0005).

    :param manifest: The validated manifest.
    :param indices: Stored bit-width to module to uint8 ``[m, n]``.
    :param luts: Bit-width to module to float16 ``[m, 2**bits]``.
    :param stats: Per-module diagnostics, or ``None`` if ``stats.json`` is absent.
    """

    manifest: QuantizedManifest
    indices: dict[int, dict[str, torch.Tensor]]
    luts: dict[int, dict[str, torch.Tensor]]
    stats: ArtifactStats | None

    @property
    def module_names(self) -> list[str]:
        """Module names in discovery order."""
        return [entry.name for entry in self.manifest.modules]


class ArtifactStore:
    """Read and write artifacts under one cache directory.

    :param cache_dir: Root of the cache, ``output.cache_dir``.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._root = cache_dir

    def fisher_dir(self, key: str) -> Path:
        """Directory of a Fisher artifact.

        :param key: Full Fisher key.
        :return: ``cache_dir / "fisher" / key[:16]``.
        """
        return self._root / "fisher" / key[:16]

    def quantized_dir(self, key: str) -> Path:
        """Directory of a quantized artifact.

        :param key: Full quantized key.
        :return: ``cache_dir / "quantized" / key[:16]``.
        """
        return self._root / "quantized" / key[:16]

    def has_fisher(self, key: str) -> bool:
        """Whether a Fisher manifest exists and parses; the snapshot is not checked.

        :param key: Full Fisher key.
        :return: ``True`` if ``load_fisher`` would get past its first two checks.
        """
        return _manifest_parses(self.fisher_dir(key), FisherManifest)

    def has_quantized(self, key: str) -> bool:
        """Whether a quantized manifest exists and parses; the snapshot is not checked.

        :param key: Full quantized key.
        :return: ``True`` if ``load_quantized`` would get past its first two checks.
        """
        return _manifest_parses(self.quantized_dir(key), QuantizedManifest)

    def save_fisher(
        self, key: str, snapshot: dict[str, JsonValue], result: FisherResult, meta: FisherMeta
    ) -> Path:
        """Write a Fisher artifact atomically.

        :param key: Full Fisher key; must be the hash of ``snapshot``.
        :param snapshot: The Fisher config snapshot.
        :param result: The estimated diagonals and calibration losses.
        :param meta: Manifest fields not derivable from ``result``.
        :return: The artifact directory.
        :raises ValueError: If the key does not hash the snapshot, or the losses do not have one
            entry per calibration sequence.
        :raises FileExistsError: If an artifact already exists for the key.
        """
        _require_key_matches(key, snapshot)
        if tuple(result.losses.shape) != (meta.num_sequences,):
            raise ValueError(
                f"losses shape {tuple(result.losses.shape)} != ({meta.num_sequences},)"
            )

        # The manifest records the modules in the order the result holds them, which is the
        # discovery order every later lookup compares against.

        manifest = FisherManifest(
            schema_version=FISHER_SCHEMA_VERSION,
            key=key,
            model_id=meta.model_id,
            revision=meta.revision,
            modules=module_entries(result.diagonals),
            config_snapshot=snapshot,
            device=meta.device,
            versions=library_versions(),
            created_at=datetime.now(UTC),
            kind="fisher",
            num_sequences=meta.num_sequences,
            seq_len=meta.seq_len,
            mean_loss=float(result.losses.double().mean()),
            seconds=result.seconds,
        )

        # Tensors first and the manifest last, all inside the temporary directory.

        final = self.fisher_dir(key)
        with _atomic_directory(final) as tmp:
            _save_tensors(result.diagonals, tmp / "fisher.safetensors")
            _save_tensors({"losses": result.losses}, tmp / "losses.safetensors")
            (tmp / _MANIFEST).write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return final

    def load_fisher(
        self, key: str, snapshot: dict[str, JsonValue], modules: Sequence[ModuleEntry]
    ) -> dict[str, torch.Tensor]:
        """Load Fisher diagonals after every check of spec 0006.

        :param key: Full Fisher key.
        :param snapshot: The Fisher snapshot the caller expects.
        :param modules: The modules the caller discovered, in order.
        :return: Module name to float32 CPU ``[m, n]``, in discovery order.
        :raises ArtifactNotFoundError: If no artifact exists for the key.
        :raises ArtifactMismatchError: If the artifact does not match the request.
        """
        directory = self.fisher_dir(key)
        manifest = _load_manifest(directory, FisherManifest)
        _check_identity(manifest, FISHER_SCHEMA_VERSION, key, snapshot, modules)

        # The losses file is validated as well, so a partial artifact is never reported as valid.

        diagonals = _load_tensors(
            directory / "fisher.safetensors",
            {entry.name: entry.shape for entry in manifest.modules},
            torch.float32,
        )
        _load_tensors(
            directory / "losses.safetensors", {"losses": (manifest.num_sequences,)}, torch.float32
        )
        return diagonals

    def save_quantized(
        self,
        key: str,
        snapshot: dict[str, JsonValue],
        result: ModelQuantization,
        meta: QuantizedMeta,
    ) -> Path:
        """Write a quantized artifact atomically, including ``stats.json``.

        :param key: Full quantized key; must be the hash of ``snapshot``.
        :param snapshot: The quantized config snapshot.
        :param result: Per-module indices, codebooks, and errors.
        :param meta: Manifest fields not derivable from ``result``.
        :return: The artifact directory.
        :raises ValueError: If the key does not hash the snapshot, or a layer's stored bit-widths
            do not match the mode and bit range in ``meta``.
        :raises FileExistsError: If an artifact already exists for the key.
        """
        _require_key_matches(key, snapshot)
        bit_widths = list(range(meta.seed_bits, meta.parent_bits + 1))
        stored = _stored_bits(meta.mode, meta.seed_bits, meta.parent_bits)
        for name, layer in result.layers.items():
            if sorted(layer.indices) != stored or sorted(layer.luts) != bit_widths:
                raise ValueError(f"{name}: stored bit-widths do not match mode {meta.mode!r}")

        # Transpose per-module results into per-bit-width files: module -> bits -> tensor
        # becomes bits -> module -> tensor, which is the on-disk layout of ADR 0005.

        parents = {name: layer.indices[meta.parent_bits] for name, layer in result.layers.items()}
        manifest = QuantizedManifest(
            schema_version=QUANTIZED_SCHEMA_VERSION,
            key=key,
            model_id=meta.model_id,
            revision=meta.revision,
            modules=module_entries(parents),
            config_snapshot=snapshot,
            device=meta.device,
            versions=library_versions(),
            created_at=datetime.now(UTC),
            kind="quantized",
            mode=meta.mode,
            seed_bits=meta.seed_bits,
            parent_bits=meta.parent_bits,
            parent_key=meta.parent_key,
            seconds=result.seconds,
        )
        stats = ArtifactStats(
            relative_error={n: dict(layer.relative_error) for n, layer in result.layers.items()},
            lloyd_iterations={n: layer.lloyd_iterations for n, layer in result.layers.items()},
        )

        # Tensors first, then the optional statistics, and the manifest last.

        final = self.quantized_dir(key)
        with _atomic_directory(final) as tmp:
            for bits in stored:
                tensors = {n: layer.indices[bits] for n, layer in result.layers.items()}
                _save_tensors(tensors, tmp / _indices_file(meta.mode, bits))
            for bits in bit_widths:
                tensors = {n: layer.luts[bits] for n, layer in result.layers.items()}
                _save_tensors(tensors, tmp / f"lut_{bits}.safetensors")
            (tmp / _STATS).write_text(stats.model_dump_json(indent=2), encoding="utf-8")
            (tmp / _MANIFEST).write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return final

    def load_quantized(
        self, key: str, snapshot: dict[str, JsonValue], modules: Sequence[ModuleEntry]
    ) -> QuantizedArtifact:
        """Load a quantized artifact after every check of spec 0006.

        :param key: Full quantized key.
        :param snapshot: The quantized snapshot the caller expects.
        :param modules: The modules the caller discovered, in order.
        :return: The artifact, with every tensor on the CPU.
        :raises ArtifactNotFoundError: If no artifact exists for the key.
        :raises ArtifactMismatchError: If the artifact does not match the request.
        """
        directory = self.quantized_dir(key)
        manifest = _load_manifest(directory, QuantizedManifest)
        _check_identity(manifest, QUANTIZED_SCHEMA_VERSION, key, snapshot, modules)

        # Every stored index file and every codebook file must hold exactly the manifest's
        # modules: uint8 [m, n] indices, and float16 [m, 2**bits] codebooks.

        entries = manifest.modules
        indices = {
            bits: _load_tensors(
                directory / _indices_file(manifest.mode, bits),
                {entry.name: entry.shape for entry in entries},
                torch.uint8,
            )
            for bits in _stored_bits(manifest.mode, manifest.seed_bits, manifest.parent_bits)
        }
        luts = {
            bits: _load_tensors(
                directory / f"lut_{bits}.safetensors",
                {entry.name: (entry.shape[0], 2**bits) for entry in entries},
                torch.float16,
            )
            for bits in range(manifest.seed_bits, manifest.parent_bits + 1)
        }
        stats = _load_stats(directory / _STATS, [entry.name for entry in entries])
        return QuantizedArtifact(manifest, indices, luts, stats)


def _stored_bits(mode: QuantizerMode, seed_bits: int, parent_bits: int) -> list[int]:
    """Bit-widths whose indices are stored: only the parent when nested, else every width."""
    return [parent_bits] if mode == "incremental" else list(range(seed_bits, parent_bits + 1))


def _indices_file(mode: QuantizerMode, bits: int) -> str:
    """File name of the indices at one stored bit-width (ADR 0005 layout)."""
    return "indices.safetensors" if mode == "incremental" else f"indices_{bits}.safetensors"


def _require_key_matches(key: str, snapshot: dict[str, JsonValue]) -> None:
    """Refuse to write an artifact under a key that is not its snapshot's hash."""
    if sha256_key(snapshot) != key:
        raise ValueError("key is not the SHA-256 of the snapshot")


def _save_tensors(tensors: Mapping[str, torch.Tensor], path: Path) -> None:
    """Write contiguous CPU copies with ``safetensors``.

    :param tensors: Name to tensor, on any device.
    :param path: Destination file.
    """
    path.write_bytes(save({name: t.detach().contiguous().cpu() for name, t in tensors.items()}))


@contextmanager
def _atomic_directory(final: Path) -> Generator[Path]:
    """Yield a temporary directory next to ``final``, and rename it into place on success.

    :param final: The artifact directory to create.
    :raises FileExistsError: If ``final`` already exists.
    """
    # An existing final directory means another run already wrote this key. Keys are content
    # hashes, so the existing artifact is equivalent, and overwriting it would only add risk.

    if final.exists():
        raise FileExistsError(f"artifact already exists: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{final.name}.", suffix=".tmp", dir=final.parent))

    # A single rename publishes the complete directory; any failure removes the partial one.

    try:
        yield tmp
        os.replace(tmp, final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def _load_manifest[M: ManifestBase](directory: Path, schema: type[M]) -> M:
    """Read and validate ``manifest.json``: the first two load-time checks of spec 0006.

    :param directory: The artifact directory.
    :param schema: The expected manifest schema, which also fixes ``kind``.
    :return: The parsed manifest.
    :raises ArtifactNotFoundError: If the directory or its manifest is missing.
    :raises ArtifactMismatchError: If the manifest does not parse into ``schema``.
    """
    path = directory / _MANIFEST
    if not path.is_file():
        raise ArtifactNotFoundError(f"no artifact at {directory}; {_CREATE_HINT}")
    try:
        return schema.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as error:
        raise ArtifactMismatchError(f"{path} is not a valid {schema.__name__}: {error}") from error


def _manifest_parses(directory: Path, schema: type[ManifestBase]) -> bool:
    """Whether ``manifest.json`` exists and parses into ``schema``."""
    try:
        _load_manifest(directory, schema)
    except (ArtifactNotFoundError, ArtifactMismatchError):
        return False
    return True


def _check_identity(
    manifest: ManifestBase,
    schema_version: int,
    key: str,
    snapshot: dict[str, JsonValue],
    modules: Sequence[ModuleEntry],
) -> None:
    """Run the version, key, snapshot, and module checks of spec 0006, in that order.

    :param manifest: The parsed manifest.
    :param schema_version: The current schema version of this artifact kind.
    :param key: The requested full key.
    :param snapshot: The requested snapshot.
    :param modules: The discovered modules.
    :raises ArtifactMismatchError: On the first failing check.
    """
    if manifest.schema_version != schema_version:
        raise ArtifactMismatchError(
            f"schema_version {manifest.schema_version} != current {schema_version}"
        )

    # The directory name is only a 16-hex prefix, so the full key guards against collisions.

    if manifest.key != key:
        raise ArtifactMismatchError(f"manifest key {manifest.key} != requested {key}")
    if manifest.config_snapshot != snapshot:
        differing = sorted(
            name
            for name in set(manifest.config_snapshot) | set(snapshot)
            if manifest.config_snapshot.get(name) != snapshot.get(name)
        )
        raise ArtifactMismatchError(f"config snapshot differs in {differing}")

    # A changed module pattern can keep the count but change the names or shapes (spec 0006).

    if manifest.modules != list(modules):
        raise ArtifactMismatchError("recorded modules differ from the discovered modules")


def _load_tensors(
    path: Path, expected: Mapping[str, _Shape], dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    """Load a tensor file on the CPU and check its names, dtype, and shapes exactly.

    :param path: A ``.safetensors`` file.
    :param expected: Tensor name to shape, in the order to return.
    :param dtype: The dtype every tensor must have.
    :return: The tensors in ``expected`` order.
    :raises ArtifactMismatchError: If the file is missing or its contents differ.
    """
    if not path.is_file():
        raise ArtifactMismatchError(f"missing tensor file {path}")
    tensors = load(path.read_bytes())
    if set(tensors) != set(expected):
        raise ArtifactMismatchError(f"{path.name} holds different tensor names than the manifest")

    # Check every tensor before returning any, so the caller never sees a partial artifact.

    for name, shape in expected.items():
        tensor = tensors[name]
        if tensor.dtype != dtype or tuple(tensor.shape) != shape:
            raise ArtifactMismatchError(
                f"{path.name}[{name}] is {tensor.dtype} {tuple(tensor.shape)}, "
                f"expected {dtype} {shape}"
            )
    return {name: tensors[name] for name in expected}


def _load_stats(path: Path, names: list[str]) -> ArtifactStats | None:
    """Load the optional ``stats.json`` and check it covers exactly the manifest's modules.

    :param path: The statistics file.
    :param names: Module names from the manifest.
    :return: The statistics, or ``None`` if the file is absent.
    :raises ArtifactMismatchError: If the file does not parse or names other modules.
    """
    if not path.is_file():
        return None
    try:
        stats = ArtifactStats.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as error:
        raise ArtifactMismatchError(f"{path} is not valid ArtifactStats: {error}") from error
    if set(stats.relative_error) != set(names) or set(stats.lloyd_iterations) != set(names):
        raise ArtifactMismatchError(f"{path.name} covers different modules than the manifest")
    return stats
