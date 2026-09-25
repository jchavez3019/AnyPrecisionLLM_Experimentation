"""Tests for cache keys: each key changes exactly with its snapshot fields (spec 0002)."""

from pathlib import Path

import pytest

from anyprec.artifacts.keys import fisher_key, quantized_key
from anyprec.config.schemas import QuantizeRunConfig, RotationHadamard
from tests import factories


def _keys(config: QuantizeRunConfig) -> tuple[str, str]:
    """Compute the Fisher and quantized keys of a run config.

    :param config: A quantization run config.
    :return: ``(fisher_key, quantized_key)``.
    """
    f_key = fisher_key(config.model, config.calibration, config.rotation)
    return f_key, quantized_key(f_key, config.quantizer)


def test_keys_are_stable_for_equal_configs(tmp_path: Path) -> None:
    """
    Given: two separately built but equal run configs.
    When: their keys are computed.
    Then: both keys match and are 64 hexadecimal characters.
    """
    first = _keys(factories.quantize_run_config(tmp_path))
    second = _keys(factories.quantize_run_config(tmp_path / "other"))

    assert first == second
    assert all(len(key) == 64 and int(key, 16) >= 0 for key in first)


@pytest.mark.parametrize(
    ("section", "field", "value", "fisher_changes"),
    [
        ("model", "revision", "1" * 40, True),
        ("model", "dtype", "bfloat16", True),
        ("calibration", "num_sequences", 8, True),
        ("calibration", "seed", 1, True),
        ("quantizer", "mode", "standalone", False),
        ("quantizer", "seed", 1, False),
        ("quantizer", "row_chunk", 8, False),
        ("model", "eval_dtype", "bfloat16", None),
    ],
)
def test_one_field_changes_exactly_the_dependent_keys(
    tmp_path: Path, section: str, field: str, value: object, fisher_changes: bool | None
) -> None:
    """
    Given: a baseline run config.
    When: one field of one section is changed.
    Then: Fisher-snapshot fields change both keys, quantizer fields change only the quantized
        key, and fields outside every snapshot (fisher_changes None) change neither.
    """
    baseline = factories.quantize_run_config(tmp_path)
    changed_section = getattr(baseline, section).model_copy(update={field: value})
    changed = baseline.model_copy(update={section: changed_section})

    (base_f, base_q), (new_f, new_q) = _keys(baseline), _keys(changed)

    if fisher_changes is None:
        assert (new_f, new_q) == (base_f, base_q)
    else:
        assert (new_f != base_f) == fisher_changes
        assert new_q != base_q


def test_rotation_is_part_of_the_fisher_key(tmp_path: Path) -> None:
    """
    Given: the same run config with rotation none and hadamard.
    When: Fisher keys are computed.
    Then: they differ, since rotated Fisher diagonals differ (ADR 0006).
    """
    baseline = factories.quantize_run_config(tmp_path)
    rotated = baseline.model_copy(
        update={
            "rotation": RotationHadamard(
                kind="hadamard", axis="in_features", randomized_signs=True, seed=0
            )
        }
    )

    assert _keys(baseline)[0] != _keys(rotated)[0]
