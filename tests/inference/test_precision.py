"""Tests for writing a bit-width's weights into a model (spec 0007)."""

import dataclasses
from pathlib import Path

import pytest
import torch
from transformers import GraniteMoeHybridForCausalLM

from anyprec.artifacts.store import ArtifactStore, QuantizedArtifact
from anyprec.inference.precision import (
    PrecisionError,
    restore_weights,
    set_precision,
    snapshot_weights,
)
from tests import factories


def _all_weights(model: GraniteMoeHybridForCausalLM) -> dict[str, torch.Tensor]:
    """Clone every parameter, so later writes can be detected bitwise."""
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def _max_distinct_per_row(weight: torch.Tensor) -> int:
    """Largest number of distinct values in any row of a ``[m, n]`` matrix."""
    # Sorting each row makes distinct values the positions where a row changes: [m, n - 1].

    values: torch.Tensor = torch.sort(weight, dim=1).values
    changes = (values[:, 1:] != values[:, :-1]).sum(dim=1)
    return int(changes.max()) + 1


def test_each_row_holds_only_its_own_codebook_values_at_every_width(
    tiny_model: GraniteMoeHybridForCausalLM, tiny_artifact: QuantizedArtifact
) -> None:
    """
    Given: a tiny artifact and the model it was built from.
    When: each bit-width from 2 to 4 is set.
    Then: every row of every target weight has at most 2**bits distinct values, all drawn from
        that row's codebook, and the artifact's tensors are still on the CPU.
    """
    for bits in range(2, 5):
        set_precision(tiny_model, tiny_artifact, bits)

        # Row-wise membership: [m, n, 1] against [m, 1, 2**bits] broadcasts to [m, n, 2**bits].

        for name, weight in factories.target_weights(tiny_model).items():
            lut = tiny_artifact.luts[bits][name].float()
            assert _max_distinct_per_row(weight) <= 2**bits
            assert bool((weight[:, :, None] == lut[:, None, :]).any(dim=2).all())
    tensors = [
        t
        for group in (tiny_artifact.indices, tiny_artifact.luts)
        for d in group.values()
        for t in d.values()
    ]
    assert all(t.device.type == "cpu" for t in tensors)


def test_result_depends_only_on_the_artifact_not_on_the_previous_width(
    tiny_model: GraniteMoeHybridForCausalLM, tiny_artifact: QuantizedArtifact
) -> None:
    """
    Given: a tiny artifact.
    When: the model goes 4 -> 2 -> 4.
    Then: the final weights equal those after setting 4 once.
    """
    set_precision(tiny_model, tiny_artifact, 4)
    once = _all_weights(tiny_model)

    set_precision(tiny_model, tiny_artifact, 2)
    set_precision(tiny_model, tiny_artifact, 4)

    assert all(torch.equal(once[n], p) for n, p in tiny_model.named_parameters())


def test_standalone_uses_its_own_indices_rather_than_the_shifted_parent(
    tiny_model: GraniteMoeHybridForCausalLM, tmp_path: Path
) -> None:
    """
    Given: a standalone artifact whose 2-bit indices are replaced with all zeros, which the
        shifted parent indices are not.
    When: 2 bits is set.
    Then: every row equals column 0 of its 2-bit codebook.
    """
    config = factories.quantize_run_config(tmp_path, "standalone")
    stored = factories.stored_tiny_artifact(
        tiny_model, config, ArtifactStore(config.output.cache_dir)
    )
    zeros = {name: torch.zeros_like(idx) for name, idx in stored.indices[2].items()}
    artifact = dataclasses.replace(stored, indices={**stored.indices, 2: zeros})

    set_precision(tiny_model, artifact, 2)

    # Column 0 of each [m, 4] codebook, broadcast over the row: [m, 1] -> [m, n].

    for name, weight in factories.target_weights(tiny_model).items():
        assert torch.equal(weight, artifact.luts[2][name][:, :1].float().expand_as(weight))


@pytest.mark.parametrize("bits", [1, 5])
def test_out_of_range_bits_raise_without_writing_any_weight(
    tiny_model: GraniteMoeHybridForCausalLM, tiny_artifact: QuantizedArtifact, bits: int
) -> None:
    """
    Given: a tiny artifact spanning 2 to 4 bits.
    When: a width just outside that range is requested.
    Then: PrecisionError names the range and no parameter changes.
    """
    before = _all_weights(tiny_model)

    with pytest.raises(PrecisionError, match=r"outside \[2, 4\]"):
        set_precision(tiny_model, tiny_artifact, bits)

    assert all(torch.equal(before[n], p) for n, p in tiny_model.named_parameters())


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"name": "model.layers.0.missing"}, "is not an nn.Linear"),
        ({"shape": (1, 1)}, r"!= manifest \(1, 1\)"),
    ],
)
def test_mismatched_last_module_raises_before_any_earlier_module_is_written(
    tiny_model: GraniteMoeHybridForCausalLM,
    tiny_artifact: QuantizedArtifact,
    update: dict[str, object],
    message: str,
) -> None:
    """
    Given: an artifact whose last manifest module is missing from the model or shaped differently.
    When: a bit-width is set.
    Then: PrecisionError is raised and even the first module's weight is unchanged.
    """
    modules = [
        *tiny_artifact.manifest.modules[:-1],
        tiny_artifact.manifest.modules[-1].model_copy(update=update),
    ]
    broken = dataclasses.replace(
        tiny_artifact, manifest=tiny_artifact.manifest.model_copy(update={"modules": modules})
    )
    before = _all_weights(tiny_model)

    with pytest.raises(PrecisionError, match=message):
        set_precision(tiny_model, broken, 3)

    assert all(torch.equal(before[n], p) for n, p in tiny_model.named_parameters())


def test_restore_returns_bitwise_original_weights_after_quantization(
    tiny_model: GraniteMoeHybridForCausalLM, tiny_artifact: QuantizedArtifact
) -> None:
    """
    Given: a snapshot of the original target weights.
    When: 2 bits is set and the snapshot is restored.
    Then: every parameter is bitwise equal to the original, and the snapshot is on the CPU.
    """
    original = _all_weights(tiny_model)
    saved = snapshot_weights(tiny_model, tiny_artifact.module_names)

    set_precision(tiny_model, tiny_artifact, 2)
    restore_weights(tiny_model, saved)

    assert all(t.device.type == "cpu" for t in saved.values())
    assert all(torch.equal(original[n], p) for n, p in tiny_model.named_parameters())


def test_restore_with_a_wrong_shape_raises_without_writing(
    tiny_model: GraniteMoeHybridForCausalLM, tiny_artifact: QuantizedArtifact
) -> None:
    """
    Given: a snapshot in which the last weight lost its final input column.
    When: it is restored after quantizing to 2 bits.
    Then: PrecisionError is raised and the quantized weights are all still in place.
    """
    saved = snapshot_weights(tiny_model, tiny_artifact.module_names)
    last = tiny_artifact.module_names[-1]
    saved[last] = saved[last][:, :-1].clone()
    set_precision(tiny_model, tiny_artifact, 2)
    quantized = _all_weights(tiny_model)

    with pytest.raises(PrecisionError, match="saved shape"):
        restore_weights(tiny_model, saved)

    assert all(torch.equal(quantized[n], p) for n, p in tiny_model.named_parameters())
