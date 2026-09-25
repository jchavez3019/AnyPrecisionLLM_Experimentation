"""Every shipped Hydra config composes and validates (spec 0002)."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from anyprec.config.loading import load_evaluate_config, load_quantize_config
from anyprec.config.schemas import RotationHadamard, RotationNone

CONFIG_DIR: Path = Path(__file__).resolve().parents[2] / "configs"


@pytest.fixture(autouse=True)
def _hydra_context() -> Iterator[None]:
    """Initialize Hydra on the repository's config directory, and clear it afterwards."""
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        yield
    GlobalHydra.instance().clear()


def test_quantize_yaml_composes_into_the_documented_defaults() -> None:
    """
    Given: configs/quantize.yaml with its default groups.
    When: it is composed and validated.
    Then: the pinned Granite revision, the ADR 0003 quantizer defaults, and no rotation result.
    """
    config = load_quantize_config(compose(config_name="quantize"))

    assert config.model.revision == "bd8a1497065c0d6ba1ef19af6b0d2b14bacf71c2"
    assert config.model.quantizable_modules.expected_count == 168
    assert (config.quantizer.seed_bits, config.quantizer.parent_bits) == (3, 8)
    assert config.quantizer.row_chunk == 1024
    assert isinstance(config.rotation, RotationNone)
    assert config.output.cache_dir == Path("outputs/cache")


def test_evaluate_yaml_composes_into_the_documented_defaults() -> None:
    """
    Given: configs/evaluate.yaml with its default groups.
    When: it is composed and validated.
    Then: both modes, bit-widths 3 to 8, and 256-position LM-head slices are configured.
    """
    config = load_evaluate_config(compose(config_name="evaluate"))

    assert config.modes == ["incremental", "standalone"]
    assert config.eval.bits == [3, 4, 5, 6, 7, 8]
    assert config.eval.lm_head_chunk_tokens == 256
    assert config.eval.datasets["c4"].max_tokens == 256 * 2048


def test_command_line_overrides_reach_the_validated_config() -> None:
    """
    Given: overrides selecting Hadamard rotation, standalone mode, and whole-chunk LM-head logits.
    When: the evaluation config is composed with them.
    Then: the validated config reflects every override.
    """
    overrides = ["rotation=hadamard", "quantizer.mode=standalone", "eval.lm_head_chunk_tokens=null"]

    config = load_evaluate_config(compose(config_name="evaluate", overrides=overrides))

    assert isinstance(config.rotation, RotationHadamard)
    assert config.quantizer.mode == "standalone"
    assert config.eval.lm_head_chunk_tokens is None
