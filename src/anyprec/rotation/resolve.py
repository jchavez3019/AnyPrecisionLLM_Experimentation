"""Rotation resolution: accept ``none`` and fail fast on ``hadamard`` (ADR 0006)."""

from anyprec.config.schemas import RotationConfig, RotationHadamard, RotationNone


def resolve_rotation(cfg: RotationConfig) -> None:
    """Accept ``rotation=none`` and reject ``rotation=hadamard``, which is not implemented yet.

    Pipelines call this right after config validation, before loading any model or data.

    :param cfg: The validated rotation config.
    :raises NotImplementedError: For ``rotation=hadamard``.
    """
    match cfg:
        case RotationNone():
            return None
        case RotationHadamard():
            raise NotImplementedError(
                "Hadamard rotation is specified in ADR 0006 but not implemented yet; "
                "use rotation=none."
            )
