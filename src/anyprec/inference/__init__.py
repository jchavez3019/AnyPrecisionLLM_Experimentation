"""Simulated inference from quantized artifacts (ADR 0005, spec 0007)."""

from anyprec.inference.precision import (
    PrecisionError,
    restore_weights,
    set_precision,
    snapshot_weights,
)

__all__ = ["PrecisionError", "restore_weights", "set_precision", "snapshot_weights"]
