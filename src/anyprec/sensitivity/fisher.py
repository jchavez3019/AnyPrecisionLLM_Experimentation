"""Empirical Fisher diagonal of every target weight (ADR 0003, Section 1; spec 0004).

``Tensor.register_post_accumulate_grad_hook`` leaves its hook parameter untyped in torch, so this
module silences pyright's unknown-member check; the hook passed to it is fully typed here.
"""

# pyright: reportUnknownMemberType=false

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from anyprec.models.heads import causal_lm_loss
from anyprec.models.loading import CausalLM


@dataclass(frozen=True)
class FisherResult:
    """Empirical Fisher diagonals and calibration diagnostics.

    :param diagonals: Float32 CPU tensors keyed by module name in discovery order, each shaped
        like its weight ``[m, n]``.
    :param losses: Per-sequence mean token NLL, float32 ``[N]``.
    :param seconds: Wall-clock time of the accumulation loop.
    """

    diagonals: dict[str, torch.Tensor]
    losses: torch.Tensor
    seconds: float


def estimate_fisher(
    model: CausalLM,
    calibration: torch.Tensor,
    targets: Mapping[str, nn.Linear],
    progress: Callable[[int, int], None] | None = None,
) -> FisherResult:
    """Accumulate the empirical Fisher diagonal of every target weight (ADR 0003, Section 1).

    Each entry is the sum over sequences of the squared gradient of that sequence's mean token
    NLL. The model is returned with its ``requires_grad`` flags restored, every ``.grad`` cleared,
    and no hooks attached, even if a forward or backward pass raises.

    :param model: The causal LM; must not be called inside ``torch.no_grad()``.
    :param calibration: int64 token ids ``[N, T]``, on any device.
    :param targets: Name to linear layer, in discovery order.
    :param progress: Optional callback receiving ``(sequences done, N)``.
    :return: CPU diagonals keyed in ``targets`` order, and the per-sequence losses.
    :raises RuntimeError: If gradients are disabled, since every diagonal would be zero.
    """
    if not torch.is_grad_enabled():
        raise RuntimeError("estimate_fisher needs grad mode; do not call it inside torch.no_grad()")

    # Remember every parameter's requires_grad flag so the model is returned exactly as received.

    parameters = dict(model.named_parameters())
    saved_flags = {name: p.requires_grad for name, p in parameters.items()}
    device = next(iter(parameters.values())).device
    num_sequences = calibration.shape[0]
    accumulators: dict[str, torch.Tensor] = {}
    handles: list[RemovableHandle] = []
    losses = torch.empty(num_sequences, dtype=torch.float32)
    try:
        # Freeze everything, then enable gradients only on target weights, so no gradient of a
        # non-target parameter is ever materialized. Activations still carry gradients through
        # the frozen embedding and norms.

        for p in parameters.values():
            p.requires_grad_(False)
        for name, linear in targets.items():
            linear.weight.requires_grad_(True)
            accumulators[name] = torch.zeros_like(linear.weight, dtype=torch.float32)
            hook = _square_into(accumulators[name])
            handles.append(linear.weight.register_post_accumulate_grad_hook(hook))

        # One sequence per backward pass: the gradient must be squared before summing over
        # sequences, or cross terms between sequences would leak in.

        started = time.perf_counter()
        for i in range(num_sequences):
            tokens = calibration[i : i + 1].to(device)
            loss = causal_lm_loss(model, tokens)
            torch.autograd.backward(loss)
            losses[i] = loss.detach().float().cpu()
            if progress is not None:
                progress(i + 1, num_sequences)
        seconds = time.perf_counter() - started
    finally:
        # Remove hooks, drop any gradient left by a failed pass, and restore the original flags.

        for handle in handles:
            handle.remove()
        for name, p in parameters.items():
            p.grad = None
            p.requires_grad_(saved_flags[name])

    # Move to the CPU one module at a time, so peak GPU memory never holds two copies.

    diagonals = {name: accumulators.pop(name).cpu() for name in list(accumulators)}
    return FisherResult(diagonals=diagonals, losses=losses, seconds=seconds)


def _square_into(accumulator: torch.Tensor) -> Callable[[torch.Tensor], None]:
    """Build a post-accumulate-grad hook that adds ``grad ** 2`` to a float32 accumulator.

    :param accumulator: float32 ``[m, n]``, shaped like the weight.
    :return: The hook.
    """

    def hook(parameter: torch.Tensor) -> None:
        """Square the finished gradient into the accumulator, then free ``.grad``."""
        grad = parameter.grad
        if grad is None:
            raise RuntimeError("post-accumulate hook fired without a gradient")

        # [m, n] gradient in the model dtype -> [m, n] float32, so bfloat16 never accumulates.

        accumulator.add_(grad.float().square())
        parameter.grad = None

    return hook
