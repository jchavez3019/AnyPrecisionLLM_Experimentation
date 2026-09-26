"""Shared pytest configuration and fixtures (spec 0010)."""

import random
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch
from hypothesis import HealthCheck, settings
from transformers import GraniteMoeHybridConfig, GraniteMoeHybridForCausalLM

from anyprec.artifacts.store import ArtifactStore, QuantizedArtifact
from anyprec.config.schemas import (
    EvaluateRunConfig,
    ModelConfig,
    QuantizerConfig,
    QuantizerMode,
    QuantizeRunConfig,
)
from anyprec.data.calibration import Encoder
from anyprec.evaluation import pipeline as evaluation_pipeline
from anyprec.quantization import pipeline as quantization_pipeline
from tests import factories
from tests.offline import OfflineLoaders

# Derandomized hypothesis runs make every failure reproducible from the test name alone.

settings.register_profile(
    "ci",
    derandomize=True,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("ci")


@pytest.fixture(scope="session")
def tiny_granite_config() -> GraniteMoeHybridConfig:
    """The verified two-layer Granite configuration; configs are only read, never mutated."""
    return factories.tiny_granite_config()


@pytest.fixture
def tiny_model() -> GraniteMoeHybridForCausalLM:
    """A fresh float32 tiny Granite model; function-scoped so no test sees another's edits."""
    return factories.tiny_model()


@pytest.fixture(scope="session")
def char_encoder() -> Encoder:
    """The byte-level encoder, a stand-in for a tokenizer over the tiny model's 256 ids."""
    return factories.char_encode


@pytest.fixture(scope="session")
def tiny_model_config() -> ModelConfig:
    """A frozen ``ModelConfig`` matching the tiny model's 12 linears."""
    return factories.model_config()


@pytest.fixture(params=["incremental", "standalone"])
def quantizer_config(request: pytest.FixtureRequest) -> QuantizerConfig:
    """Tiny-model quantizer settings, parametrized over both modes."""
    mode: QuantizerMode = request.param
    return factories.quantizer_config(mode=mode)


@pytest.fixture
def quantize_run_config(tmp_path: Path) -> QuantizeRunConfig:
    """A quantization run config writing under ``tmp_path``."""
    return factories.quantize_run_config(tmp_path)


@pytest.fixture
def evaluate_run_config(tmp_path: Path) -> EvaluateRunConfig:
    """An evaluation run config writing under ``tmp_path``."""
    return factories.evaluate_run_config(tmp_path)


@pytest.fixture
def offline_loaders(monkeypatch: pytest.MonkeyPatch) -> Iterator[OfflineLoaders]:
    """Replace the Hub loaders in both pipeline modules with recording offline stand-ins.

    Both pipelines call ``seed_everything``, so the global generators are restored afterwards,
    and a pipeline test never changes another test's random draws.
    """
    loaders = OfflineLoaders()
    for module in (quantization_pipeline, evaluation_pipeline):
        for name in ("load_model", "load_tokenizer", "make_encoder", "load_texts"):
            monkeypatch.setattr(module, name, getattr(loaders, name))

    # Save every generator seed_everything touches; CUDA generators are seeded lazily and
    # never drawn from by the offline suite.

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.default_generator.get_state()
    yield loaders
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.default_generator.set_state(torch_state)


@pytest.fixture(params=["incremental", "standalone"])
def tiny_artifact(
    request: pytest.FixtureRequest, tmp_path: Path, tiny_model: GraniteMoeHybridForCausalLM
) -> QuantizedArtifact:
    """A tiny artifact built from ``tiny_model``, saved and reloaded, in each quantizer mode."""
    mode: QuantizerMode = request.param
    config = factories.quantize_run_config(tmp_path, mode)
    return factories.stored_tiny_artifact(
        tiny_model, config, ArtifactStore(config.output.cache_dir)
    )
