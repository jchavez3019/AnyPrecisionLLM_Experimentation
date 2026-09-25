"""Canonical JSON, SHA-256 cache keys, and stable per-name seeds (spec 0002)."""

import hashlib
import json

type JsonValue = str | int | float | bool | list[JsonValue] | dict[str, JsonValue] | None

_SEED_MASK: int = (1 << 63) - 1


def canonical_json(value: JsonValue) -> str:
    """Serialize a JSON value with sorted keys and no whitespace.

    Equal values always produce equal strings, so their hashes are equal too.

    :param value: A JSON-compatible value.
    :return: The canonical JSON text.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_key(value: JsonValue) -> str:
    """Hash the canonical JSON of a value.

    :param value: A JSON-compatible value.
    :return: The full 64-character hexadecimal SHA-256 digest.
    """
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def stable_seed(seed: int, name: str) -> int:
    """Derive a reproducible generator seed from a base seed and a name.

    The first 8 bytes of ``sha256(f"{seed}:{name}")`` are read as an unsigned big-endian
    integer and masked to 63 bits, a range ``torch.Generator.manual_seed`` always accepts.
    Unlike ``hash()``, the result does not change between Python processes.

    :param seed: Base seed, for example ``quantizer.seed``.
    :param name: Distinguishing name, for example a module name or ``"chunk3"``.
    :return: A seed in ``[0, 2**63)``.
    """
    digest: bytes = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & _SEED_MASK
