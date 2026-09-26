"""Tests for ``run_quantization`` and its entry script (spec 0009), offline on the tiny model."""

import json
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

import pytest
import torch

from anyprec.artifacts.keys import fisher_snapshot, quantized_snapshot
from anyprec.artifacts.manifest import module_entries
from anyprec.artifacts.store import ArtifactStore
from anyprec.config.schemas import QuantizeRunConfig, RotationHadamard
from anyprec.quantization import pipeline
from anyprec.quantization.pipeline import SUMMARY_FILE, run_quantization
from tests import factories
from tests.offline import OfflineLoaders

REPO_ROOT: Path = Path(__file__).resolve().parents[2]


def _fail_if_called(*args: object, **kwargs: object) -> NoReturn:
    """Stand in for an expensive stage that a cached run must never reach."""
    raise AssertionError("an expensive stage ran although its artifact was cached")


def _forbid_recomputation(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    """Replace the named pipeline stages with spies that fail the test if called.

    :param monkeypatch: The test's monkeypatch.
    :param names: Attribute names in ``anyprec.quantization.pipeline``.
    """
    for name in names:
        monkeypatch.setattr(pipeline, name, _fail_if_called)


def test_run_on_an_empty_cache_computes_and_saves_both_artifacts(
    tmp_path: Path, quantize_run_config: QuantizeRunConfig, offline_loaders: OfflineLoaders
) -> None:
    """
    Given: an empty cache and the tiny model behind offline loaders.
    When: run_quantization runs in incremental mode.
    Then: the Fisher and the artifact are saved under their keys and reload against the
        discovered modules, calibration text is loaded once, and the summary's median error
        falls from the seed width to the parent width.
    """
    cfg = quantize_run_config

    outcome = run_quantization(cfg, tmp_path / "run")

    assert not outcome.fisher_reused and not outcome.quantized_reused
    assert offline_loaders.calls.count("load_texts:in-memory:train") == 1

    # Both artifacts reload through every identity check, against a fresh discovery.

    store = ArtifactStore(cfg.output.cache_dir)
    modules = module_entries(factories.target_weights(factories.tiny_model()))
    f_snapshot = fisher_snapshot(cfg.model, cfg.calibration, cfg.rotation)
    q_snapshot = quantized_snapshot(outcome.fisher_key, cfg.quantizer)
    diagonals = store.load_fisher(outcome.fisher_key, f_snapshot, modules)
    artifact = store.load_quantized(outcome.quantized_key, q_snapshot, modules)
    assert all(bool((d >= 0).all()) and bool(d.sum() > 0) for d in diagonals.values())
    assert artifact.manifest.parent_key == outcome.fisher_key
    assert outcome.quantized_dir == store.quantized_dir(outcome.quantized_key)

    # The summary is for people; it must still name the keys and show a falling error curve.

    summary = json.loads((tmp_path / "run" / SUMMARY_FILE).read_text())
    assert summary["quantized_key"] == outcome.quantized_key
    median = summary["median_relative_error"]
    assert median["4"] < median["2"]


def test_second_identical_run_reuses_both_artifacts_without_recomputing(
    tmp_path: Path,
    quantize_run_config: QuantizeRunConfig,
    offline_loaders: OfflineLoaders,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Given: a cache already holding this config's Fisher and artifact.
    When: run_quantization runs again with the identical config.
    Then: both are reported reused, with the same keys, and neither the Fisher pass, k-means,
        nor the calibration text loader is called.
    """
    first = run_quantization(quantize_run_config, tmp_path / "first")
    offline_loaders.calls.clear()
    _forbid_recomputation(monkeypatch, "estimate_fisher", "quantize_model")

    second = run_quantization(quantize_run_config, tmp_path / "second")

    assert second.fisher_reused and second.quantized_reused
    assert (second.fisher_key, second.quantized_key) == (first.fisher_key, first.quantized_key)
    assert offline_loaders.calls == ["load_model"]


def test_changing_only_the_mode_reuses_the_fisher_and_adds_a_second_artifact(
    tmp_path: Path,
    quantize_run_config: QuantizeRunConfig,
    offline_loaders: OfflineLoaders,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Given: a cache holding the incremental artifact and its Fisher.
    When: run_quantization runs with quantizer.mode=standalone and nothing else changed.
    Then: the Fisher is reused without a new Fisher pass, and a second artifact with a
        different key is created from it.
    """
    incremental = run_quantization(quantize_run_config, tmp_path / "incremental")
    _forbid_recomputation(monkeypatch, "estimate_fisher")
    standalone_cfg = factories.quantize_run_config(tmp_path, mode="standalone")

    standalone = run_quantization(standalone_cfg, tmp_path / "standalone")

    assert standalone.fisher_reused and not standalone.quantized_reused
    assert standalone.fisher_key == incremental.fisher_key
    assert standalone.quantized_key != incremental.quantized_key
    store = ArtifactStore(quantize_run_config.output.cache_dir)
    assert store.has_quantized(incremental.quantized_key)
    assert store.has_quantized(standalone.quantized_key)


def test_hadamard_rotation_fails_before_any_loader_is_called(
    tmp_path: Path, quantize_run_config: QuantizeRunConfig, offline_loaders: OfflineLoaders
) -> None:
    """
    Given: a config selecting the Hadamard rotation, which ADR 0006 has not implemented.
    When: run_quantization is called.
    Then: NotImplementedError is raised before any model, tokenizer, or text is loaded.
    """
    hadamard = RotationHadamard(kind="hadamard", axis="in_features", randomized_signs=True, seed=0)
    cfg = quantize_run_config.model_copy(update={"rotation": hadamard})

    with pytest.raises(NotImplementedError, match="Hadamard"):
        run_quantization(cfg, tmp_path / "run")
    assert offline_loaders.calls == []


def test_cuda_request_without_cuda_fails_before_any_loader_is_called(
    tmp_path: Path,
    quantize_run_config: QuantizeRunConfig,
    offline_loaders: OfflineLoaders,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Given: a machine where CUDA is unavailable, and a config requesting device=cuda.
    When: run_quantization is called.
    Then: RuntimeError asks for device=cpu explicitly, and nothing is loaded.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = quantize_run_config.model_copy(update={"device": "cuda"})

    with pytest.raises(RuntimeError, match="device=cpu"):
        run_quantization(cfg, tmp_path / "run")
    assert offline_loaders.calls == []


def test_entry_script_resolves_its_config_path_and_prints_the_job_config() -> None:
    """
    Given: the quantization entry script and the repository's configs directory.
    When: it is run as a subprocess with Hydra's --cfg job flag, which skips the pipeline.
    Then: it exits 0 and prints the composed model config, proving config_path is correct.
    """
    script = REPO_ROOT / "quantization" / "quantize_any_precision.py"

    completed = subprocess.run(
        [sys.executable, str(script), "--cfg", "job"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    assert "model_id: ibm-granite/granite-4.0-350m" in completed.stdout
