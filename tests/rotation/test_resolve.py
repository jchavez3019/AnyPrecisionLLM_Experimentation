"""Tests for rotation resolution (ADR 0006)."""

import pytest

from anyprec.config.schemas import RotationHadamard, RotationNone
from anyprec.rotation.resolve import resolve_rotation


def test_resolve_rotation_accepts_none() -> None:
    """
    Given: the rotation=none config.
    When: it is resolved.
    Then: no error is raised.
    """
    resolve_rotation(RotationNone(kind="none"))


def test_resolve_rotation_raises_not_implemented_for_hadamard() -> None:
    """
    Given: a valid rotation=hadamard config.
    When: it is resolved.
    Then: NotImplementedError points the user to rotation=none.
    """
    config = RotationHadamard(kind="hadamard", axis="in_features", randomized_signs=True, seed=0)

    with pytest.raises(NotImplementedError, match="rotation=none"):
        resolve_rotation(config)
