"""The Hugging Face model boundary: loading, module discovery, loss, body, and head (spec 0003)."""

from anyprec.models.discovery import QuantizableModuleError, find_quantizable_linears
from anyprec.models.heads import (
    SlicedLogitsError,
    body_hidden_states,
    causal_lm_loss,
    check_sliced_logits,
    logit_head,
)
from anyprec.models.loading import CausalLM, load_model, load_tokenizer

__all__ = [
    "CausalLM",
    "QuantizableModuleError",
    "SlicedLogitsError",
    "body_hidden_states",
    "causal_lm_loss",
    "check_sliced_logits",
    "find_quantizable_linears",
    "load_model",
    "load_tokenizer",
    "logit_head",
]
