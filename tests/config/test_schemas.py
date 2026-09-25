"""Tests for the pydantic config schemas (spec 0002)."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from anyprec.config.schemas import (
    EvalConfig,
    EvaluateRunConfig,
    QuantizableModules,
    QuantizerConfig,
)
from tests import factories


def test_quantizer_config_rejects_unknown_key() -> None:
    """
    Given: valid quantizer settings plus a misspelled key.
    When: they are validated.
    Then: validation fails instead of silently ignoring the typo.
    """
    settings = factories.quantizer_config().model_dump() | {"row_chunks": 8}

    with pytest.raises(ValidationError, match="row_chunks"):
        QuantizerConfig.model_validate(settings)


@pytest.mark.parametrize(
    ("seed_bits", "parent_bits"),
    [(5, 4), (3, 9), (0, 4)],
    ids=["seed-above-parent", "parent-above-uint8", "seed-zero"],
)
def test_quantizer_config_rejects_invalid_bit_range(seed_bits: int, parent_bits: int) -> None:
    """
    Given: a bit range that is inverted, exceeds uint8 indices, or starts at zero.
    When: the quantizer settings are validated.
    Then: validation fails.
    """
    settings = factories.quantizer_config().model_dump()
    settings |= {"seed_bits": seed_bits, "parent_bits": parent_bits}

    with pytest.raises(ValidationError):
        QuantizerConfig.model_validate(settings)


def test_quantizable_modules_rejects_invalid_regex() -> None:
    """
    Given: a module pattern with an unbalanced parenthesis.
    When: it is validated.
    Then: validation fails with the regex error.
    """
    with pytest.raises(ValidationError, match=r"invalid quantizable_modules\.pattern"):
        QuantizableModules(pattern="(q_proj", expected_count=1)


@pytest.mark.parametrize("slice_len", [0, 300, -256])
def test_eval_config_rejects_lm_head_chunk_tokens_not_power_of_two(slice_len: int) -> None:
    """
    Given: an LM-head slice length that is zero, negative, or not a power of two.
    When: the evaluation settings are validated.
    Then: validation fails.
    """
    settings = factories.eval_config().model_dump() | {"lm_head_chunk_tokens": slice_len}

    with pytest.raises(ValidationError):
        EvalConfig.model_validate(settings)


@pytest.mark.parametrize("slice_len", [None, 1, 256, 2048])
def test_eval_config_accepts_null_or_power_of_two_lm_head_chunk_tokens(
    slice_len: int | None,
) -> None:
    """
    Given: an LM-head slice length that is null or a power of two.
    When: the evaluation settings are validated.
    Then: the value is kept as given.
    """
    settings = factories.eval_config().model_dump() | {"lm_head_chunk_tokens": slice_len}

    assert EvalConfig.model_validate(settings).lm_head_chunk_tokens == slice_len


def test_eval_config_rejects_kl_dataset_missing_from_datasets() -> None:
    """
    Given: evaluation settings whose KL dataset is not configured.
    When: they are validated.
    Then: validation fails.
    """
    settings = factories.eval_config().model_dump()
    del settings["datasets"]["wikitext2"]

    with pytest.raises(ValidationError, match=r"kl\.dataset"):
        EvalConfig.model_validate(settings)


@pytest.mark.parametrize("bits", [[4, 3], [3, 3], []], ids=["descending", "duplicate", "empty"])
def test_eval_config_rejects_unsorted_duplicate_or_empty_bits(bits: list[int]) -> None:
    """
    Given: an evaluated bit-width list that is unsorted, repeated, or empty.
    When: the evaluation settings are validated.
    Then: validation fails.
    """
    settings = factories.eval_config().model_dump() | {"bits": bits}

    with pytest.raises(ValidationError):
        EvalConfig.model_validate(settings)


def test_evaluate_run_config_rejects_bits_outside_quantizer_range(tmp_path: Path) -> None:
    """
    Given: a tiny-model evaluation config whose quantizer covers 2 to 4 bits.
    When: 5 bits is requested for evaluation.
    Then: validation fails, naming the out-of-range width.
    """
    settings = factories.evaluate_run_config(tmp_path).model_dump()
    settings["eval"]["bits"] = [2, 5]

    with pytest.raises(ValidationError, match=r"\[5\]"):
        EvaluateRunConfig.model_validate(settings)


def test_evaluate_run_config_rejects_duplicate_modes(tmp_path: Path) -> None:
    """
    Given: an evaluation config listing the same mode twice.
    When: it is validated.
    Then: validation fails.
    """
    settings = factories.evaluate_run_config(tmp_path).model_dump()
    settings["modes"] = ["incremental", "incremental"]

    with pytest.raises(ValidationError, match="modes"):
        EvaluateRunConfig.model_validate(settings)


def test_run_configs_are_frozen(tmp_path: Path) -> None:
    """
    Given: a validated quantization run config.
    When: a field is assigned.
    Then: pydantic refuses the mutation.
    """
    config = factories.quantize_run_config(tmp_path)

    with pytest.raises(ValidationError):
        config.seed = 1  # pyright: ignore[reportAttributeAccessIssue]
