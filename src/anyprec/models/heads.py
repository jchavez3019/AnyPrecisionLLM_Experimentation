"""How a causal LM splits into a loss, a body, and an LM head (specs 0003, 0004, 0008).

These are the only functions that read untyped Hugging Face model outputs; each one narrows the
output to a ``Tensor`` before returning it.
"""

from collections.abc import Callable
from typing import cast

import torch
from torch import nn

from anyprec.models.loading import CausalLM


class SlicedLogitsError(RuntimeError):
    """Body plus sliced head does not reproduce ``model(x).logits`` for this model."""


def causal_lm_loss(model: CausalLM, input_ids: torch.Tensor) -> torch.Tensor:
    """Mean next-token NLL of ``[1, T]`` token ids, attached to the autograd graph.

    ``labels=input_ids`` makes Hugging Face shift the labels and average over ``T - 1`` targets.

    :param model: The causal LM.
    :param input_ids: int64 ``[1, T]`` on the model's device.
    :return: A 0-d tensor.
    :raises TypeError: If the model returns no loss for labelled input.
    """
    loss: object = model(input_ids=input_ids, labels=input_ids, use_cache=False).loss
    if not isinstance(loss, torch.Tensor):
        raise TypeError(f"{type(model).__name__} returned no loss for labelled input")
    return loss


def body_hidden_states(model: CausalLM, input_ids: torch.Tensor) -> torch.Tensor:
    """Run the decoder body only, up to and including its final norm.

    :param model: The causal LM.
    :param input_ids: int64 ``[1, T]``.
    :return: Hidden states ``[T, H]``.
    :raises SlicedLogitsError: If the body returns no hidden-state tensor.
    """
    body = cast(object, model.get_decoder())
    if not isinstance(body, nn.Module):
        raise SlicedLogitsError(f"get_decoder() returned {type(body).__name__}, not a module")
    hidden: object = body(input_ids=input_ids, use_cache=False).last_hidden_state
    if not isinstance(hidden, torch.Tensor):
        raise SlicedLogitsError("decoder body returned no last_hidden_state tensor")

    # [1, T, H] -> [T, H]: evaluation runs one chunk at a time.

    return hidden[0]


def logit_head(model: CausalLM) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return the LM head as a function, including Granite's division by ``logits_scaling``.

    :param model: The causal LM.
    :return: A function mapping hidden states ``[S, H]`` to logits ``[S, V]``.
    :raises SlicedLogitsError: If the output embeddings are not an ``nn.Linear``.
    """
    head = cast(object, model.get_output_embeddings())
    if not isinstance(head, nn.Linear):
        raise SlicedLogitsError(f"output embeddings are {type(head).__name__}, not nn.Linear")
    scaling: object = getattr(model.config, "logits_scaling", 1.0)
    if not isinstance(scaling, int | float):
        raise SlicedLogitsError(f"logits_scaling is {type(scaling).__name__}, not a number")
    scale = float(scaling)
    linear: nn.Linear = head

    def apply(hidden: torch.Tensor) -> torch.Tensor:
        """Project ``[S, H]`` hidden states to ``[S, V]`` scaled logits."""
        return linear(hidden) / scale

    return apply


def check_sliced_logits(
    model: CausalLM, input_ids: torch.Tensor, slice_len: int | None, atol: float = 1e-4
) -> None:
    """Require body plus sliced head to match ``model(input_ids).logits`` within ``atol``.

    :param model: The causal LM.
    :param input_ids: int64 ``[1, T]``.
    :param slice_len: Positions per head application; ``None`` applies it to all at once,
        which still checks the body/head split.
    :param atol: Largest allowed absolute logit difference.
    :raises SlicedLogitsError: If the difference exceeds ``atol``, or the split is unavailable.
    """
    with torch.inference_mode():
        full: object = model(input_ids=input_ids, use_cache=False).logits
        if not isinstance(full, torch.Tensor):
            raise SlicedLogitsError("forward() returned no logits tensor")

        # Rebuild the [T, V] logits from [T, H] hidden states, slice_len positions at a time.

        hidden = body_hidden_states(model, input_ids)
        head = logit_head(model)
        step = hidden.shape[0] if slice_len is None else slice_len
        sliced = torch.cat([head(hidden[s : s + step]) for s in range(0, hidden.shape[0], step)])
        error = float((sliced - full[0]).abs().max())
    if error > atol:
        raise SlicedLogitsError(f"sliced logits differ from forward() by {error:.3g} (atol {atol})")
