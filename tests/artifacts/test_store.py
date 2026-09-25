"""Tests for the artifact store: layout, atomic writes, and load-time checks (spec 0006)."""

import dataclasses
import json
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from safetensors.torch import save

import anyprec.artifacts.store as store_module
from anyprec.artifacts.keys import fisher_key, quantized_snapshot
from anyprec.artifacts.manifest import ModuleEntry
from anyprec.artifacts.store import (
    ArtifactMismatchError,
    ArtifactNotFoundError,
    ArtifactStore,
    QuantizedMeta,
)
from anyprec.config.schemas import QuantizerMode, QuantizeRunConfig
from anyprec.quantization.model import ModelQuantization, quantize_model
from anyprec.utils.hashing import JsonValue, sha256_key
from tests import factories

type Snapshot = dict[str, JsonValue]


class _Saved:
    """A quantized artifact written to a temporary store, plus everything needed to reload it.

    :param root: Temporary directory.
    :param mode: Quantizer mode.
    """

    def __init__(self, root: Path, mode: QuantizerMode) -> None:
        self.config: QuantizeRunConfig = factories.quantize_run_config(root, mode)
        weights = factories.target_weights(factories.tiny_model())
        self.result: ModelQuantization = quantize_model(
            weights, factories.random_fisher(weights), self.config.quantizer, torch.device("cpu")
        )
        f_key = fisher_key(self.config.model, self.config.calibration, self.config.rotation)
        self.snapshot: Snapshot = quantized_snapshot(f_key, self.config.quantizer)
        self.key: str = sha256_key(self.snapshot)
        self.modules: list[ModuleEntry] = [
            ModuleEntry(name=n, shape=(w.shape[0], w.shape[1])) for n, w in weights.items()
        ]
        self.store = ArtifactStore(self.config.output.cache_dir)
        self.meta = QuantizedMeta.from_config(self.config, torch.device("cpu"))
        self.directory: Path = self.store.save_quantized(
            self.key, self.snapshot, self.result, self.meta
        )

    def load(self) -> None:
        """Reload with the original request, so a test can check whether it still passes."""
        self.store.load_quantized(self.key, self.snapshot, self.modules)


def _edit_manifest(directory: Path, edit: Callable[[dict[str, object]], None]) -> None:
    """Apply an in-place edit to an artifact's manifest.json on disk.

    :param directory: The artifact directory.
    :param edit: Mutates the parsed JSON object.
    """
    path = directory / "manifest.json"
    data: dict[str, object] = json.loads(path.read_text())
    edit(data)
    path.write_text(json.dumps(data))


@pytest.fixture(params=["incremental", "standalone"])
def saved(request: pytest.FixtureRequest, tmp_path: Path) -> _Saved:
    """A freshly saved tiny artifact in each mode."""
    mode: QuantizerMode = request.param
    return _Saved(tmp_path, mode)


def test_quantized_artifact_round_trips_bitwise_in_its_mode_layout(saved: _Saved) -> None:
    """
    Given: a saved tiny artifact.
    When: it is loaded with the original request.
    Then: every index and codebook tensor is bitwise equal and on the CPU, the manifest records
        the run, stats hold the recorded errors, and the file layout matches the mode.
    """
    artifact = saved.store.load_quantized(saved.key, saved.snapshot, saved.modules)

    # Tensors, bit-width by bit-width, against the in-memory result.

    for name, layer in saved.result.layers.items():
        for bits, indices in layer.indices.items():
            assert torch.equal(artifact.indices[bits][name], indices)
            assert artifact.indices[bits][name].device.type == "cpu"
        for bits, lut in layer.luts.items():
            assert torch.equal(artifact.luts[bits][name], lut)
    assert artifact.stats is not None
    assert artifact.stats.relative_error == {
        n: layer.relative_error for n, layer in saved.result.layers.items()
    }

    # Manifest identity and the two index layouts of ADR 0005.

    assert artifact.manifest.key == saved.key
    assert artifact.manifest.parent_key == saved.meta.parent_key
    assert artifact.module_names == [entry.name for entry in saved.modules]
    names = sorted(path.name for path in saved.directory.iterdir())
    if saved.config.quantizer.mode == "incremental":
        assert "indices.safetensors" in names and not any(n.startswith("indices_") for n in names)
    else:
        assert [n for n in names if n.startswith("indices")] == [
            "indices_2.safetensors",
            "indices_3.safetensors",
            "indices_4.safetensors",
        ]


def test_artifact_without_stats_loads_with_stats_none(saved: _Saved) -> None:
    """
    Given: a saved artifact whose optional stats.json was deleted.
    When: it is loaded.
    Then: loading succeeds and stats is None.
    """
    (saved.directory / "stats.json").unlink()

    artifact = saved.store.load_quantized(saved.key, saved.snapshot, saved.modules)

    assert artifact.stats is None


def test_saving_over_an_existing_key_raises_and_keeps_the_original(saved: _Saved) -> None:
    """
    Given: a saved artifact.
    When: the same key is saved again.
    Then: FileExistsError is raised, and the original still loads unchanged.
    """
    before = (saved.directory / "manifest.json").read_bytes()

    with pytest.raises(FileExistsError):
        saved.store.save_quantized(saved.key, saved.snapshot, saved.result, saved.meta)

    assert (saved.directory / "manifest.json").read_bytes() == before
    saved.load()


def test_failure_midway_through_save_leaves_no_artifact_and_no_temporary_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Given: a tensor writer that fails on its second file.
    When: a Fisher artifact is saved.
    Then: the error propagates, and the cache holds neither the key directory nor a temp dir.
    """
    config = factories.quantize_run_config(tmp_path)
    weights = factories.target_weights(factories.tiny_model())
    result = factories.fisher_result(weights, config.calibration.num_sequences)
    snapshot, key = factories.fisher_snapshot_and_key(config)
    store = ArtifactStore(config.output.cache_dir)
    calls: list[Path] = []

    def failing_save(tensors: dict[str, torch.Tensor], path: Path) -> None:
        """Write the first file, then fail, as a crash between two files would."""
        calls.append(path)
        if len(calls) == 2:
            raise OSError("disk full")
        path.write_bytes(save(tensors))

    monkeypatch.setattr(store_module, "_save_tensors", failing_save)

    with pytest.raises(OSError, match="disk full"):
        store.save_fisher(key, snapshot, result, factories.fisher_meta(config))

    assert list((config.output.cache_dir / "fisher").iterdir()) == []


def test_save_rejects_a_key_that_is_not_the_snapshot_hash(saved: _Saved) -> None:
    """
    Given: a valid quantization result.
    When: it is saved under a key that does not hash its snapshot.
    Then: ValueError is raised and nothing is written.
    """
    wrong_key = "0" * 64

    with pytest.raises(ValueError, match="SHA-256"):
        saved.store.save_quantized(wrong_key, saved.snapshot, saved.result, saved.meta)

    assert not saved.store.quantized_dir(wrong_key).exists()


def test_save_rejects_layers_whose_bit_widths_do_not_match_the_mode(saved: _Saved) -> None:
    """
    Given: a result produced in one mode.
    When: it is saved with metadata claiming the other mode.
    Then: ValueError names a module, since the stored index widths cannot match.
    """
    other: QuantizerMode = (
        "standalone" if saved.config.quantizer.mode == "incremental" else "incremental"
    )
    meta = dataclasses.replace(saved.meta, mode=other)

    with pytest.raises(ValueError, match="stored bit-widths"):
        saved.store.save_quantized(saved.key, saved.snapshot, saved.result, meta)


def test_load_raises_not_found_naming_the_command_that_creates_it(tmp_path: Path) -> None:
    """
    Given: an empty cache.
    When: a quantized artifact is requested.
    Then: ArtifactNotFoundError names the quantization entry script, and has_quantized is False.
    """
    store = ArtifactStore(tmp_path)

    with pytest.raises(ArtifactNotFoundError, match=r"quantize_any_precision\.py"):
        store.load_quantized("a" * 64, {}, [])
    assert not store.has_quantized("a" * 64)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("kind", "fisher", "not a valid QuantizedManifest"),
        ("schema_version", 99, "schema_version"),
        ("key", "f" * 64, "manifest key"),
    ],
)
def test_load_raises_mismatch_when_manifest_identity_is_edited(
    saved: _Saved, field: str, value: object, message: str
) -> None:
    """
    Given: a saved artifact whose manifest kind, schema version, or key is edited on disk.
    When: it is loaded.
    Then: ArtifactMismatchError explains which check failed.
    """
    _edit_manifest(saved.directory, lambda data: data.__setitem__(field, value))

    with pytest.raises(ArtifactMismatchError, match=message):
        saved.load()


def test_load_raises_mismatch_listing_the_differing_snapshot_section(saved: _Saved) -> None:
    """
    Given: a saved artifact.
    When: it is requested with a snapshot whose quantizer section differs by one field.
    Then: ArtifactMismatchError lists "quantizer" as the differing top-level key.
    """
    quantizer = saved.snapshot["quantizer"]
    assert isinstance(quantizer, dict)
    changed: Snapshot = {**saved.snapshot, "quantizer": {**quantizer, "seed": 1}}

    with pytest.raises(ArtifactMismatchError, match=r"\['quantizer'\]"):
        saved.store.load_quantized(saved.key, changed, saved.modules)


def test_load_raises_mismatch_when_a_module_is_renamed(saved: _Saved) -> None:
    """
    Given: a saved artifact.
    When: it is requested with a module list in which one module has a different name.
    Then: ArtifactMismatchError is raised, even though the count is unchanged.
    """
    renamed = [saved.modules[0].model_copy(update={"name": "renamed"}), *saved.modules[1:]]

    with pytest.raises(ArtifactMismatchError, match="modules differ"):
        saved.store.load_quantized(saved.key, saved.snapshot, renamed)


def test_load_raises_mismatch_when_a_tensor_file_is_missing_or_has_wrong_dtype(
    saved: _Saved,
) -> None:
    """
    Given: a saved artifact.
    When: one codebook file is deleted, and separately rewritten with float32 tensors.
    Then: each load raises ArtifactMismatchError naming the problem.
    """
    lut_path = saved.directory / "lut_3.safetensors"
    original = {n: layer.luts[3] for n, layer in saved.result.layers.items()}
    lut_path.unlink()

    with pytest.raises(ArtifactMismatchError, match="missing tensor file"):
        saved.load()

    # Rewrite with the right names and shapes but the wrong dtype.

    lut_path.write_bytes(save({n: t.float().contiguous() for n, t in original.items()}))
    with pytest.raises(ArtifactMismatchError, match=r"expected torch\.float16"):
        saved.load()

    # Rewrite with the right dtype but one tensor under a foreign name.

    renamed = {**original, "foreign": original[saved.modules[0].name]}
    del renamed[saved.modules[0].name]
    lut_path.write_bytes(save(renamed))
    with pytest.raises(ArtifactMismatchError, match="different tensor names"):
        saved.load()


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("not json", "not valid ArtifactStats"),
        ('{"relative_error": {}, "lloyd_iterations": {}}', "covers different modules"),
    ],
)
def test_load_raises_mismatch_when_stats_are_invalid_or_cover_other_modules(
    saved: _Saved, contents: str, message: str
) -> None:
    """
    Given: a saved artifact whose optional stats.json is unparsable, or names no modules.
    When: it is loaded.
    Then: ArtifactMismatchError is raised, because present stats must describe this artifact.
    """
    (saved.directory / "stats.json").write_text(contents)

    with pytest.raises(ArtifactMismatchError, match=message):
        saved.load()


def test_fisher_save_rejects_losses_of_the_wrong_length(tmp_path: Path) -> None:
    """
    Given: a Fisher result with one loss fewer than calibration sequences.
    When: it is saved.
    Then: ValueError is raised before anything is written.
    """
    config = factories.quantize_run_config(tmp_path)
    weights = factories.target_weights(factories.tiny_model())
    result = factories.fisher_result(weights, config.calibration.num_sequences - 1)
    snapshot, key = factories.fisher_snapshot_and_key(config)
    store = ArtifactStore(config.output.cache_dir)

    with pytest.raises(ValueError, match="losses shape"):
        store.save_fisher(key, snapshot, result, factories.fisher_meta(config))
    assert not store.has_fisher(key)
