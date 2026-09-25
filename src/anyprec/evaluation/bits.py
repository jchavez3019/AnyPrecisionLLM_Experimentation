"""Analytic bits per weight (ADR 0004, Metric 5; spec 0008).

The numbers depend only on module shapes and the bit-width range, so no model is needed.
"""

from collections.abc import Sequence

from anyprec.config.schemas import FrozenModel

_CODEBOOK_ENTRY_BITS: int = 16


def layer_bits_per_weight(bits: int, n: int) -> float:
    """Bits per weight of one ``b``-bit layer: indices plus one float16 codebook per row.

    :param bits: Bit-width ``b``.
    :param n: Row length, the layer's input features.
    :return: ``b + 16 * 2**b / n``.
    """
    return bits + _CODEBOOK_ENTRY_BITS * 2**bits / n


def parent_bits_per_weight(seed_bits: int, parent_bits: int, n: int) -> float:
    """Bits per weight of the resident any-precision parent: parent indices plus every codebook.

    :param seed_bits: Smallest bit-width ``b0``.
    :param parent_bits: Parent bit-width ``B``.
    :param n: Row length.
    :return: ``B + (16 / n) * sum_{b=b0}^{B} 2**b``.
    """
    entries = sum(2**b for b in range(seed_bits, parent_bits + 1))
    return parent_bits + _CODEBOOK_ENTRY_BITS * entries / n


class BitsReport(FrozenModel):
    """Bits per weight of every bit-width and of the parent, for one model.

    :param quantized_params: Parameters in the quantized layers.
    :param total_params: All parameters, tied weights counted once.
    :param per_bits: Bit-width to the parameter-weighted average over the quantized layers.
    :param per_bits_whole_model: The same, with every other parameter counted at 16 bits.
    :param parent: Parent bits per weight over the quantized layers.
    :param parent_whole_model: The same, over the whole model.
    """

    quantized_params: int
    total_params: int
    per_bits: dict[int, float]
    per_bits_whole_model: dict[int, float]
    parent: float
    parent_whole_model: float


def bits_report(
    shapes: Sequence[tuple[int, int]], total_params: int, seed_bits: int, parent_bits: int
) -> BitsReport:
    """Average the per-layer figures over quantized parameters, and over the whole model.

    :param shapes: ``(m, n)`` of every quantized layer.
    :param total_params: ``sum(p.numel() for p in model.parameters())``.
    :param seed_bits: Smallest bit-width.
    :param parent_bits: Largest bit-width.
    :return: The report.
    :raises ValueError: If there are no shapes, or they hold more parameters than the model.
    """
    quantized = sum(m * n for m, n in shapes)
    if quantized == 0 or quantized > total_params:
        raise ValueError(f"{quantized} quantized parameters for a {total_params}-parameter model")
    unquantized_bits = _CODEBOOK_ENTRY_BITS * (total_params - quantized)

    # Each layer contributes its bits per weight times its parameter count: total bits.

    per_bits: dict[int, float] = {}
    per_bits_whole: dict[int, float] = {}
    for bits in range(seed_bits, parent_bits + 1):
        layer_bits = sum(m * n * layer_bits_per_weight(bits, n) for m, n in shapes)
        per_bits[bits] = layer_bits / quantized
        per_bits_whole[bits] = (layer_bits + unquantized_bits) / total_params
    parent = sum(m * n * parent_bits_per_weight(seed_bits, parent_bits, n) for m, n in shapes)
    return BitsReport(
        quantized_params=quantized,
        total_params=total_params,
        per_bits=per_bits,
        per_bits_whole_model=per_bits_whole,
        parent=parent / quantized,
        parent_whole_model=(parent + unquantized_bits) / total_params,
    )
