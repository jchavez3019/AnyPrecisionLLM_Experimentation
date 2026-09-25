"""Cache keys for Fisher and quantized artifacts (ADR 0002, spec 0002).

A key is the SHA-256 of a snapshot holding exactly the configuration an artifact depends on,
plus a schema version. Changing any other field (the evaluation dtype, the device, the module
pattern) reuses the artifact; changing a field in the snapshot produces a new key.
"""

from typing import cast

from pydantic import TypeAdapter

from anyprec.config.schemas import (
    CalibrationConfig,
    ModelConfig,
    QuantizerConfig,
    RotationConfig,
)
from anyprec.utils.hashing import JsonValue, sha256_key

FISHER_SCHEMA_VERSION: int = 1
QUANTIZED_SCHEMA_VERSION: int = 1

_ROTATION_ADAPTER: TypeAdapter[RotationConfig] = TypeAdapter(RotationConfig)


def fisher_snapshot(
    model: ModelConfig, calibration: CalibrationConfig, rotation: RotationConfig
) -> dict[str, JsonValue]:
    """Collect the configuration a Fisher artifact depends on (ADR 0002, caching table).

    :param model: Model settings; only the checkpoint identity and the Fisher dtype are used.
    :param calibration: Calibration settings, all of which affect the Fisher.
    :param rotation: Rotation settings, since rotated Fisher diagonals differ.
    :return: The JSON snapshot stored in the manifest and hashed into the key.
    """
    return {
        "schema_version": FISHER_SCHEMA_VERSION,
        "model": {"model_id": model.model_id, "revision": model.revision, "dtype": model.dtype},
        "calibration": cast(JsonValue, calibration.model_dump(mode="json")),
        "rotation": cast(JsonValue, _ROTATION_ADAPTER.dump_python(rotation, mode="json")),
    }


def quantized_snapshot(fisher_key: str, quantizer: QuantizerConfig) -> dict[str, JsonValue]:
    """Collect the configuration a quantized artifact depends on.

    :param fisher_key: Full key of the Fisher artifact the quantization reads.
    :param quantizer: Quantizer settings, all of which affect the codebooks.
    :return: The JSON snapshot stored in the manifest and hashed into the key.
    """
    return {
        "schema_version": QUANTIZED_SCHEMA_VERSION,
        "fisher_key": fisher_key,
        "quantizer": cast(JsonValue, quantizer.model_dump(mode="json")),
    }


def fisher_key(model: ModelConfig, calibration: CalibrationConfig, rotation: RotationConfig) -> str:
    """Compute the full Fisher cache key.

    :param model: Model settings.
    :param calibration: Calibration settings.
    :param rotation: Rotation settings.
    :return: 64-character hexadecimal key.
    """
    return sha256_key(fisher_snapshot(model, calibration, rotation))


def quantized_key(fisher_key: str, quantizer: QuantizerConfig) -> str:
    """Compute the full quantized-artifact cache key.

    :param fisher_key: Full key of the Fisher artifact.
    :param quantizer: Quantizer settings.
    :return: 64-character hexadecimal key.
    """
    return sha256_key(quantized_snapshot(fisher_key, quantizer))
