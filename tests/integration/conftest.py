"""Shared helpers for the integration tests on the real Granite checkpoint (spec 0011).

Every config is composed from the shipped YAML through Hydra, with the smallest overrides that
keep a test to minutes, so these tests also prove the defaults the acceptance run uses.
"""

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from anyprec.config.loading import load_evaluate_config, load_quantize_config
from anyprec.config.schemas import EvaluateRunConfig, QuantizerMode, QuantizeRunConfig
from anyprec.quantization.pipeline import QuantizationOutcome, run_quantization

CONFIG_DIR: Path = Path(__file__).resolve().parents[2] / "configs"
NUM_SEQUENCES: int = 8
GIB: int = 2**30


def compose_quantize(overrides: list[str]) -> QuantizeRunConfig:
    """Compose and validate ``configs/quantize.yaml`` with command-line style overrides.

    :param overrides: Hydra overrides, for example ``["calibration.num_sequences=8"]``.
    :return: The validated run config.
    """
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return load_quantize_config(compose(config_name="quantize", overrides=overrides))


def compose_evaluate(overrides: list[str]) -> EvaluateRunConfig:
    """Compose and validate ``configs/evaluate.yaml`` with command-line style overrides.

    :param overrides: Hydra overrides.
    :return: The validated run config.
    """
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return load_evaluate_config(compose(config_name="evaluate", overrides=overrides))


def small_run_overrides(base_dir: Path) -> list[str]:
    """Overrides shared by the quantize and evaluate integration runs: 8 sequences, a temp cache.

    :param base_dir: Temporary output root; the cache lives under it.
    :return: Hydra overrides.
    """
    return [f"calibration.num_sequences={NUM_SEQUENCES}", f"output.base_dir={base_dir}"]


@pytest.fixture
def cuda_device() -> torch.device:
    """The CUDA device; skips the test with a clear reason on a machine without one."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available; GPU integration tests need a CUDA device")
    return torch.device("cuda")


@dataclass(frozen=True)
class GraniteRun:
    """Both Granite artifacts quantized from one 8-sequence Fisher into a temporary cache.

    :param base_dir: Output root holding the cache.
    :param outcomes: Mode to the outcome of its first quantization run.
    """

    base_dir: Path
    outcomes: dict[QuantizerMode, QuantizationOutcome]


@pytest.fixture(scope="session")
def granite_run(tmp_path_factory: pytest.TempPathFactory) -> GraniteRun:
    """Quantize Granite in both modes once per session; the quantize and evaluate tests share it.

    The run takes several minutes of k-means per mode, which is why it is session-scoped. Tests
    only read the returned outcomes and the cache; none of them writes into it except through
    the pipelines, which never overwrite an existing artifact.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available; GPU integration tests need a CUDA device")
    base_dir = tmp_path_factory.mktemp("granite")
    modes: list[QuantizerMode] = ["incremental", "standalone"]
    outcomes: dict[QuantizerMode, QuantizationOutcome] = {}
    for mode in modes:
        cfg = compose_quantize([*small_run_overrides(base_dir), f"quantizer.mode={mode}"])
        outcomes[mode] = run_quantization(cfg, base_dir / f"quantize-{mode}")
    return GraniteRun(base_dir=base_dir, outcomes=outcomes)
