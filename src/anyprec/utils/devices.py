"""Device resolution without silent fallbacks (spec 0009)."""

import torch


def resolve_device(name: str) -> torch.device:
    """Parse a configured device name.

    :param name: ``"cuda"``, ``"cuda:0"``, ``"cpu"``, or any other ``torch.device`` string.
    :return: The parsed device.
    :raises RuntimeError: If a CUDA device is requested but CUDA is not available.
    """
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"device={name!r} was requested but CUDA is not available; pass device=cpu explicitly"
        )
    return device
