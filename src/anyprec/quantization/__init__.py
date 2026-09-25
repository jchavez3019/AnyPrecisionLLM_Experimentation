"""Fisher-weighted k-means with incremental upscaling (ADR 0003, spec 0005).

Library modules import from the defining modules, never from this package (spec 0001).
"""

from anyprec.quantization.init import weighted_kmeanspp_init
from anyprec.quantization.layer import LayerQuantization, quantize_layer
from anyprec.quantization.lloyd import LloydResult, weighted_lloyd
from anyprec.quantization.model import ModelQuantization, quantize_model
from anyprec.quantization.rows import PreparedRows, prepare_rows, segment_stats
from anyprec.quantization.split import segment_ids, split_all_segments

__all__ = [
    "LayerQuantization",
    "LloydResult",
    "ModelQuantization",
    "PreparedRows",
    "prepare_rows",
    "quantize_layer",
    "quantize_model",
    "segment_ids",
    "segment_stats",
    "split_all_segments",
    "weighted_kmeanspp_init",
    "weighted_lloyd",
]
