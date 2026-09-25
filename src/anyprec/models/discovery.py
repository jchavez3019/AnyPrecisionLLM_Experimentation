"""Find the quantizable linear layers of a model (ADR 0002, spec 0003)."""

import re
from collections.abc import Iterator
from typing import cast

from torch import nn

from anyprec.config.schemas import QuantizableModules


class QuantizableModuleError(RuntimeError):
    """The model's quantizable modules do not match the configuration."""


def find_quantizable_linears(model: nn.Module, cfg: QuantizableModules) -> dict[str, nn.Linear]:
    """Return the matching linear layers in ``named_modules()`` order, keyed by qualified name.

    That order is the canonical module order of the Fisher cache, manifests, and results.

    :param model: Any module tree.
    :param cfg: The name pattern and the exact number of modules it must match.
    :return: Qualified name to linear layer.
    :raises QuantizableModuleError: If the count differs from ``cfg.expected_count``, or two
        matched modules share one weight tensor.
    """
    pattern = re.compile(cfg.pattern)
    modules = cast("Iterator[tuple[str, nn.Module]]", model.named_modules())
    found = {
        name: module
        for name, module in modules
        if isinstance(module, nn.Linear) and pattern.match(name)
    }
    if len(found) != cfg.expected_count:
        raise QuantizableModuleError(
            f"expected {cfg.expected_count} quantizable linears, found {len(found)}"
        )

    # Tied weights would make one tensor appear under two names and be quantized twice.

    pointers = [module.weight.data_ptr() for module in found.values()]
    if len(set(pointers)) != len(pointers):
        raise QuantizableModuleError("two quantizable modules share one weight tensor")
    return found
