"""Text loading at the Hugging Face boundary: calibration and evaluation tokens (spec 0003)."""

from anyprec.data.calibration import CalibrationError, Encoder, make_encoder, sample_calibration
from anyprec.data.evaluation_text import iter_chunks, load_eval_tokens
from anyprec.data.hub import load_texts

__all__ = [
    "CalibrationError",
    "Encoder",
    "iter_chunks",
    "load_eval_tokens",
    "load_texts",
    "make_encoder",
    "sample_calibration",
]
