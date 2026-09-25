"""Simulated inference: write a bit-width's dequantized weights into a model (ADR 0005, spec 0007).

Only ``torch`` is needed: the model is reached through ``named_modules()``, so the tiny test
model and Granite are handled identically.
"""

from collections.abc import Iterator, Mapping, Sequence
from typing import cast

import torch
from torch import nn

from anyprec.artifacts.manifest import ModuleEntry
from anyprec.artifacts.store import QuantizedArtifact


class PrecisionError(ValueError):
    """The requested bit-width or the artifact's module list does not fit the model."""


def set_precision(model: nn.Module, artifact: QuantizedArtifact, bits: int) -> None:
    """Overwrite every quantized weight with its ``bits``-wide dequantization.

    The artifact stays on the CPU; one module's indices and codebook are moved to the weight's
    device at a time. The result depends only on the artifact, never on the current weights.

    :param model: Any module tree containing the artifact's linears.
    :param artifact: A loaded quantized artifact.
    :param bits: Bit-width in ``[seed_bits, parent_bits]``.
    :raises PrecisionError: If ``bits`` is out of range, or a module is missing, not an
        ``nn.Linear``, or shaped differently; in every case no weight has been written.
    """
    manifest = artifact.manifest
    if not manifest.seed_bits <= bits <= manifest.parent_bits:
        raise PrecisionError(f"bits={bits} outside [{manifest.seed_bits}, {manifest.parent_bits}]")

    # Resolve every module before writing any weight, so a mismatch cannot leave the model
    # half-updated.

    modules = dict(_named_modules(model))
    targets = [_checked_linear(modules, entry) for entry in manifest.modules]

    for entry, linear in zip(manifest.modules, targets, strict=True):
        # Shift on the CPU, where uint8 shifts are exact, so the transfer stays one byte per
        # weight. [m, n] uint8 indices and a [m, 2**bits] codebook in the weight's dtype.

        weight = linear.weight
        idx = _indices_for(artifact, entry.name, bits).to(weight.device)
        lut = artifact.luts[bits][entry.name].to(weight.device, weight.dtype)

        # Gather one codebook entry per weight: [m, 2**bits] indexed by [m, n] -> [m, n].
        # The in-place write to a leaf parameter must not be recorded by autograd.

        with torch.no_grad():
            weight.copy_(lut.gather(1, idx.long()))


def snapshot_weights(model: nn.Module, names: Sequence[str]) -> dict[str, torch.Tensor]:
    """Copy the named linears' weights to the CPU so they can be restored later.

    :param model: The model.
    :param names: Qualified names of ``nn.Linear`` modules.
    :return: Name to a detached CPU copy of its weight.
    :raises PrecisionError: If a name is missing or is not an ``nn.Linear``.
    """
    modules = dict(_named_modules(model))
    return {name: _linear(modules, name).weight.detach().cpu().clone() for name in names}


def restore_weights(model: nn.Module, saved: Mapping[str, torch.Tensor]) -> None:
    """Copy saved weights back in place.

    :param model: The model the weights were taken from.
    :param saved: Output of ``snapshot_weights``.
    :raises PrecisionError: If a name is missing, not an ``nn.Linear``, or shaped differently.
    """
    modules = dict(_named_modules(model))
    linears = {name: _linear(modules, name) for name in saved}

    # Validate every shape before copying, so a mismatch leaves the model untouched.

    for name, linear in linears.items():
        if linear.weight.shape != saved[name].shape:
            raise PrecisionError(f"{name}: saved shape {tuple(saved[name].shape)} differs")
    with torch.no_grad():
        for name, linear in linears.items():
            linear.weight.copy_(saved[name])


def _named_modules(model: nn.Module) -> Iterator[tuple[str, nn.Module]]:
    """Typed view of ``named_modules()``, whose return type torch leaves partially unknown."""
    return cast("Iterator[tuple[str, nn.Module]]", model.named_modules())


def _linear(modules: Mapping[str, nn.Module], name: str) -> nn.Linear:
    """Look up a module and require it to be an ``nn.Linear``."""
    module = modules.get(name)
    if not isinstance(module, nn.Linear):
        raise PrecisionError(f"{name} is not an nn.Linear in this model")
    return module


def _checked_linear(modules: Mapping[str, nn.Module], entry: ModuleEntry) -> nn.Linear:
    """Look up a manifest module and require its weight shape to match the manifest."""
    linear = _linear(modules, entry.name)
    if tuple(linear.weight.shape) != entry.shape:
        raise PrecisionError(
            f"{entry.name}: weight {tuple(linear.weight.shape)} != manifest {entry.shape}"
        )
    return linear


def _indices_for(artifact: QuantizedArtifact, name: str, bits: int) -> torch.Tensor:
    """Indices at ``bits``: the shifted parent when nested, else that width's own indices."""
    manifest = artifact.manifest
    if manifest.mode == "incremental":
        return artifact.indices[manifest.parent_bits][name] >> (manifest.parent_bits - bits)
    return artifact.indices[bits][name]
