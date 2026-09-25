"""Quantization of every target module of a model (spec 0005)."""

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch
from torch import nn

from anyprec.config.schemas import QuantizerConfig
from anyprec.quantization.layer import LayerQuantization, quantize_layer
from anyprec.utils.hashing import stable_seed


@dataclass(frozen=True)
class ModelQuantization:
    """Quantization results for every target module.

    :param layers: Module name to its quantization, in discovery order.
    :param seconds: Wall-clock time of the whole loop.
    """

    layers: dict[str, LayerQuantization]
    seconds: float


def quantize_model(
    targets: Mapping[str, nn.Linear],
    fisher: Mapping[str, torch.Tensor],
    cfg: QuantizerConfig,
    device: torch.device,
    progress: Callable[[str], None] | None = None,
) -> ModelQuantization:
    """Quantize every target module; the model's own weights are only read, never written.

    :param targets: Module name to linear layer, in discovery order.
    :param fisher: Module name to Fisher diagonal ``[m, n]``, on any device.
    :param cfg: Quantizer settings.
    :param device: Compute device for the kernels.
    :param progress: Optional callback receiving each module name after it is quantized.
    :return: Per-module results, on the CPU.
    :raises KeyError: If a target has no Fisher diagonal.
    """
    started = time.perf_counter()
    layers: dict[str, LayerQuantization] = {}
    with torch.no_grad():
        for name, linear in targets.items():
            # Per-module seeds make each codebook independent of processing order
            # (ADR 0003, Section 3). [m, n] weight and Fisher moved to the compute device.

            weight: torch.Tensor = linear.weight.detach().to(device, torch.float32)
            module_fisher: torch.Tensor = fisher[name].to(device)
            layers[name] = quantize_layer(weight, module_fisher, cfg, stable_seed(cfg.seed, name))
            del weight, module_fisher
            if progress is not None:
                progress(name)
    return ModelQuantization(layers, time.perf_counter() - started)
