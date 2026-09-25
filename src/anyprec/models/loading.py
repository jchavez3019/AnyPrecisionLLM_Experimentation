"""Load Granite and its tokenizer from the Hugging Face Hub (ADR 0002, spec 0003).

``models/`` is the only subpackage that touches a Hugging Face model object. ``CausalLM`` lets
code outside this boundary name the model type without importing ``transformers``.

The ``Auto*.from_pretrained`` factories are untyped in ``transformers``, so this module alone
silences pyright's unknown-member check. Their results are cast to ``object`` and narrowed with
``isinstance``, which is the only type check that holds at runtime.
"""

# pyright: reportUnknownMemberType=false

from typing import cast

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from anyprec.config.schemas import ModelConfig

type CausalLM = PreTrainedModel


def load_model(cfg: ModelConfig, dtype: torch.dtype, device: torch.device) -> CausalLM:
    """Load the causal LM in inference mode on one device.

    :param cfg: Model identity and revision.
    :param dtype: Parameter dtype, ``model.dtype`` for quantization or ``model.eval_dtype``.
    :param device: The single device that holds every parameter.
    :return: The model, in ``eval()`` mode.
    :raises TypeError: If the checkpoint does not load as a ``PreTrainedModel``.
    """
    model = cast(
        object,
        AutoModelForCausalLM.from_pretrained(
            cfg.model_id, revision=cfg.revision, dtype=dtype, device_map=str(device)
        ),
    )

    # transformers is only partially typed; narrow once here so callers see PreTrainedModel.

    if not isinstance(model, PreTrainedModel):
        raise TypeError(f"{cfg.model_id} did not load as a PreTrainedModel")
    model.eval()
    return model


def load_tokenizer(cfg: ModelConfig) -> PreTrainedTokenizerBase:
    """Load the tokenizer pinned to the same revision as the weights.

    :param cfg: Model identity and revision.
    :return: The tokenizer.
    :raises TypeError: If the repository does not provide a Hugging Face tokenizer.
    """
    tokenizer = cast(object, AutoTokenizer.from_pretrained(cfg.model_id, revision=cfg.revision))
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise TypeError(f"{cfg.model_id} did not load a PreTrainedTokenizerBase")
    return tokenizer
