"""Tests for ``run_evaluation`` (spec 0009), offline on the tiny model."""

from pathlib import Path

import pytest

from anyprec.artifacts.store import ArtifactNotFoundError
from anyprec.config.schemas import EvaluateRunConfig, QuantizerMode
from anyprec.evaluation.pipeline import RESULTS_FILE, SameWeightsError, run_evaluation
from anyprec.evaluation.results import Results
from anyprec.quantization.pipeline import run_quantization
from tests import factories
from tests.offline import OfflineLoaders


def _quantize(tmp_path: Path, *modes: QuantizerMode) -> dict[QuantizerMode, str]:
    """Run the quantization pipeline once per mode into the shared tiny cache.

    :param tmp_path: The test's temporary directory, which holds the cache.
    :param modes: Modes to quantize.
    :return: Mode to its full artifact key.
    """
    keys: dict[QuantizerMode, str] = {}
    for mode in modes:
        cfg = factories.quantize_run_config(tmp_path, mode=mode)
        keys[mode] = run_quantization(cfg, tmp_path / f"quantize-{mode}").quantized_key
    return keys


def test_run_writes_results_with_one_entry_per_mode_and_bits_in_order(
    tmp_path: Path, evaluate_run_config: EvaluateRunConfig, offline_loaders: OfflineLoaders
) -> None:
    """
    Given: both tiny artifacts in the cache, and a config evaluating both modes at 2 to 4 bits.
    When: run_evaluation runs.
    Then: results.json re-validates to the returned Results, whose entries are ordered by mode
        then bits and point at the artifacts that were built, and whose reference covers both
        datasets.
    """
    keys = _quantize(tmp_path, "incremental", "standalone")

    results = run_evaluation(evaluate_run_config, tmp_path / "run")

    written = Results.model_validate_json((tmp_path / "run" / RESULTS_FILE).read_text())
    assert written == results
    assert [(e.mode, e.bits) for e in results.entries] == [
        (mode, bits) for mode in ("incremental", "standalone") for bits in (2, 3, 4)
    ]
    assert all(e.artifact_key == keys[e.mode] for e in results.entries)
    assert [p.dataset for p in results.reference.perplexity] == ["wikitext2", "c4"]


def test_missing_standalone_artifact_fails_before_tokenizing_and_names_the_command(
    tmp_path: Path, evaluate_run_config: EvaluateRunConfig, offline_loaders: OfflineLoaders
) -> None:
    """
    Given: a cache holding only the incremental artifact, and a config requesting both modes.
    When: run_evaluation runs.
    Then: ArtifactNotFoundError names quantizer.mode=standalone, no evaluation text was loaded
        or tokenized, and no results.json exists.
    """
    _quantize(tmp_path, "incremental")
    offline_loaders.calls.clear()

    with pytest.raises(ArtifactNotFoundError, match=r"quantizer\.mode=standalone"):
        run_evaluation(evaluate_run_config, tmp_path / "run")

    assert offline_loaders.calls == ["load_model", "load_model"]
    assert not (tmp_path / "run" / RESULTS_FILE).exists()


def test_disagreeing_model_instances_fail_the_same_weights_check(
    tmp_path: Path, evaluate_run_config: EvaluateRunConfig, offline_loaders: OfflineLoaders
) -> None:
    """
    Given: both artifacts cached, but a reference and a quantized instance that load different
        weights (seeds 0 and 1).
    When: run_evaluation runs its start-up checks.
    Then: SameWeightsError is raised before any precision is set, and no results.json exists.
    """
    _quantize(tmp_path, "incremental", "standalone")
    offline_loaders.model_seeds = [0, 1]

    with pytest.raises(SameWeightsError, match="before quantization"):
        run_evaluation(evaluate_run_config, tmp_path / "run")

    assert not (tmp_path / "run" / RESULTS_FILE).exists()
