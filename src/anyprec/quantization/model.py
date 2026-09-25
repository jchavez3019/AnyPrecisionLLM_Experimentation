"""Quantization of every target module of a model (spec 0005)."""

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch

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
    weights: Mapping[str, torch.Tensor],
    fisher: Mapping[str, torch.Tensor],
    cfg: QuantizerConfig,
    device: torch.device,
    progress: Callable[[str], None] | None = None,
) -> ModelQuantization:
    """Quantize every target weight matrix; the tensors passed in are only read, never written.

    The caller chooses what is clustered: the module weights as they are, or, once ADR 0006 is
    implemented, their rotated form. The kernels do not depend on ``torch.nn``.

    :param weights: Module name to weight matrix ``[m, n]``, in discovery order, on any device.
    :param fisher: Module name to Fisher diagonal ``[m, n]``, on any device.
    :param cfg: Quantizer settings.
    :param device: Compute device for the kernels.
    :param progress: Optional callback receiving each module name after it is quantized.
    :return: Per-module results, on the CPU.
    :raises KeyError: If a weight has no Fisher diagonal.
    :raises ValueError: If a Fisher diagonal's shape differs from its weight's shape.
    """
    started = time.perf_counter()
    layers: dict[str, LayerQuantization] = {}
    with torch.no_grad():
        for name, weight in weights.items():
            # Mismatched shapes would otherwise surface as an opaque gather error in the kernels.

            module_fisher: torch.Tensor = fisher[name]
            if module_fisher.shape != weight.shape:
                raise ValueError(
                    f"{name}: Fisher shape {tuple(module_fisher.shape)} differs from "
                    f"weight shape {tuple(weight.shape)}"
                )

            # Per-module seeds make each codebook independent of processing order
            # (ADR 0003, Section 3). [m, n] weight and Fisher moved to the compute device.

            w: torch.Tensor = weight.detach().to(device, torch.float32)
            f: torch.Tensor = module_fisher.to(device)
            layers[name] = quantize_layer(w, f, cfg, stable_seed(cfg.seed, name))
            del w, f
            if progress is not None:
                progress(name)
    return ModelQuantization(layers, time.perf_counter() - started)
