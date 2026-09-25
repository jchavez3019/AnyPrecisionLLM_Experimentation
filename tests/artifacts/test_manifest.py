"""Tests for the manifest and statistics schemas (spec 0006)."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from anyprec.artifacts.manifest import ArtifactStats, ModuleEntry, QuantizedManifest


def _manifest() -> QuantizedManifest:
    """A valid quantized manifest with a nested snapshot."""
    return QuantizedManifest(
        schema_version=1,
        key="a" * 64,
        model_id="ibm-granite/granite-4.0-350m",
        revision="main",
        modules=[ModuleEntry(name="model.layers.0.self_attn.q_proj", shape=(64, 64))],
        config_snapshot={"quantizer": {"seed": 0, "mode": "incremental"}, "parent_key": "b" * 64},
        device="cpu",
        versions={"torch": "2.x"},
        created_at=datetime(2026, 9, 25, tzinfo=UTC),
        kind="quantized",
        mode="incremental",
        seed_bits=2,
        parent_bits=4,
        parent_key="b" * 64,
        seconds=0.5,
    )


def test_manifest_json_round_trip_preserves_snapshot_and_timestamp_exactly() -> None:
    """
    Given: a quantized manifest with a nested snapshot and a UTC timestamp.
    When: it is dumped to JSON and parsed back.
    Then: the parsed manifest equals the original, so snapshot comparisons after a reload are exact.
    """
    manifest = _manifest()

    assert QuantizedManifest.model_validate_json(manifest.model_dump_json()) == manifest


def test_manifest_rejects_unknown_fields_and_the_other_kind() -> None:
    """
    Given: manifest JSON with an extra field, and separately with kind "fisher".
    When: each is parsed as a quantized manifest.
    Then: both are rejected, so a stale or foreign manifest cannot load as this kind.
    """
    data = _manifest().model_dump(mode="json")

    with pytest.raises(ValidationError, match="extra"):
        QuantizedManifest.model_validate({**data, "unexpected": 1})
    with pytest.raises(ValidationError, match="kind"):
        QuantizedManifest.model_validate({**data, "kind": "fisher"})


def test_module_entry_rejects_a_non_positive_dimension() -> None:
    """
    Given: a module shape with a zero dimension.
    When: a module entry is built.
    Then: validation fails.
    """
    with pytest.raises(ValidationError):
        ModuleEntry(name="x", shape=(0, 4))


def test_stats_restore_integer_bit_width_keys_after_json() -> None:
    """
    Given: statistics keyed by integer bit-widths, which JSON turns into strings.
    When: they are dumped and parsed back.
    Then: the keys are integers again, so lookups by ``bits`` keep working after a reload.
    """
    stats = ArtifactStats(relative_error={"q": {2: 0.1, 4: 0.01}}, lloyd_iterations={"q": 7})

    loaded = ArtifactStats.model_validate_json(stats.model_dump_json())

    assert loaded == stats
    assert list(loaded.relative_error["q"]) == [2, 4]
