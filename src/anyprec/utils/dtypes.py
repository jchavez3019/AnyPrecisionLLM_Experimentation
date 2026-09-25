"""Mapping from configuration dtype names to ``torch.dtype`` values."""

from typing import Literal

import torch

type DTypeName = Literal["bfloat16", "float16", "float32"]

_DTYPES: dict[DTypeName, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def torch_dtype(name: DTypeName) -> torch.dtype:
    """Return the ``torch.dtype`` named in a validated configuration.

    :param name: A dtype name accepted by the config schemas.
    :return: The corresponding ``torch.dtype``.
    """
    return _DTYPES[name]
