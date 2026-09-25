"""Quantization of one weight matrix into storage-ready tensors (spec 0005)."""

from dataclasses import dataclass

import torch

from anyprec.config.schemas import QuantizerConfig
from anyprec.quantization.lloyd import weighted_lloyd
from anyprec.quantization.rows import PreparedRows, prepare_rows
from anyprec.quantization.split import segment_ids, split_all_segments
from anyprec.utils.hashing import stable_seed

type Codebook = tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class LayerQuantization:
    """Storage-ready quantization of one weight matrix (ADR 0005).

    :param indices: Bit-width to uint8 ``[m, n]`` indices on the CPU. Incremental mode stores
        only ``parent_bits``; standalone mode stores every bit-width.
    :param luts: Bit-width to float16 ``[m, 2**bits]`` codebooks on the CPU, one per row.
    :param relative_error: Bit-width to ``J / sum(f w^2)`` over the whole matrix, computed with
        the float16 codebooks that simulated inference uses; 0 for an all-zero matrix. The
        conditioned ``f`` is used, as in notebook 02, so a row with no Fisher signal (weighted
        by 1) can dominate a module whose other sensitivities are tiny.
    :param lloyd_iterations: Largest Lloyd iteration count over all chunks and bit-widths.
    """

    indices: dict[int, torch.Tensor]
    luts: dict[int, torch.Tensor]
    relative_error: dict[int, float]
    lloyd_iterations: int


def _fit_codebooks(
    rows: PreparedRows, cfg: QuantizerConfig, chunk_seed: int
) -> tuple[dict[int, Codebook], int]:
    """Fit ``(centroids, borders)`` for every bit-width of one chunk of rows.

    Every Lloyd fit starts a fresh generator from ``chunk_seed``, so the standalone fit at
    ``seed_bits`` is bitwise identical to the incremental seed (ADR 0003, Section 3).

    :param rows: Prepared rows of one chunk.
    :param cfg: Quantizer settings.
    :param chunk_seed: Seed of this chunk.
    :return: Codebooks keyed by bit-width, and the largest Lloyd iteration count.
    """
    device = rows.w_sorted.device

    def fit(bits: int) -> tuple[Codebook, int]:
        """Run one Lloyd fit from a fresh generator in the chunk's seed state.

        :param bits: Bit-width of the codebook.
        :return: ``(centroids, borders)`` and the Lloyd iteration count.
        """
        generator = torch.Generator(device=device).manual_seed(chunk_seed)
        result = weighted_lloyd(rows, 2**bits, generator, cfg.lloyd_max_iter, cfg.empty_eps)
        return (result.centroids, result.borders), result.iterations

    # Standalone: an independent Lloyd fit at every bit-width (ADR 0003, Section 5).

    if cfg.mode == "standalone":
        fits = {b: fit(b) for b in range(cfg.seed_bits, cfg.parent_bits + 1)}
        return {b: codebook for b, (codebook, _) in fits.items()}, max(i for _, i in fits.values())

    # Incremental: one Lloyd seed, then exact splits up to the parent (ADR 0003, Section 4).

    (centroids, borders), iterations = fit(cfg.seed_bits)
    codebooks: dict[int, Codebook] = {cfg.seed_bits: (centroids, borders)}
    for bits in range(cfg.seed_bits + 1, cfg.parent_bits + 1):
        centroids, borders = split_all_segments(rows, borders, centroids, cfg.empty_eps)
        codebooks[bits] = (centroids, borders)
    return codebooks, iterations


def quantize_layer(
    weight: torch.Tensor, fisher: torch.Tensor, cfg: QuantizerConfig, generator_seed: int
) -> LayerQuantization:
    """Quantize every row of one matrix, ``cfg.row_chunk`` rows at a time.

    :param weight: Weight matrix ``[m, n]`` on the compute device.
    :param fisher: Fisher diagonal ``[m, n]`` on the same device.
    :param cfg: Quantizer settings.
    :param generator_seed: This module's seed, from ``stable_seed(cfg.seed, module_name)``.
    :return: Indices, codebooks, and relative errors, all on the CPU.
    :raises ValueError: If a row is shorter than the parent codebook.
    """
    m, n = weight.shape
    if n < 2**cfg.parent_bits:
        raise ValueError(f"rows of length {n} cannot hold {2**cfg.parent_bits} distinct centroids")
    bit_widths = range(cfg.seed_bits, cfg.parent_bits + 1)
    stored_bits = [cfg.parent_bits] if cfg.mode == "incremental" else list(bit_widths)

    # Preallocate CPU outputs so each chunk's results can be written in place.

    indices = {b: torch.empty(m, n, dtype=torch.uint8) for b in stored_bits}
    luts = {b: torch.empty(m, 2**b, dtype=torch.float16) for b in bit_widths}
    error_sum: dict[int, float] = dict.fromkeys(bit_widths, 0.0)
    energy_sum: float = 0.0
    lloyd_iterations: int = 0

    for chunk_index, start in enumerate(range(0, m, cfg.row_chunk)):
        # One chunk of rows: [R, n] with R <= row_chunk. The chunk seed depends only on the
        # module seed and the chunk position.

        stop = min(start + cfg.row_chunk, m)
        rows = prepare_rows(weight[start:stop].float(), fisher[start:stop])
        energy_sum += (rows.f_sorted * rows.w_sorted.square()).sum().item()
        chunk_seed = stable_seed(generator_seed, f"chunk{chunk_index}")
        codebooks, iterations = _fit_codebooks(rows, cfg, chunk_seed)
        lloyd_iterations = max(lloyd_iterations, iterations)

        for bits, (centroids, borders) in codebooks.items():
            # Round the codebook to its storage dtype first, so the recorded error is the error
            # simulated inference will actually see. [R, 2**bits] gathered by [R, n] -> [R, n].

            lut16: torch.Tensor = centroids.to(torch.float16)
            ids_sorted: torch.Tensor = segment_ids(borders, n)
            reconstructed: torch.Tensor = lut16.double().gather(1, ids_sorted)
            residual: torch.Tensor = rows.w_sorted - reconstructed
            error_sum[bits] += (rows.f_sorted * residual.square()).sum().item()
            luts[bits][start:stop] = lut16.cpu()

            # Undo the sort: scatter each sorted position's id back to its original column.

            if bits in indices:
                ids: torch.Tensor = torch.empty_like(ids_sorted).scatter_(1, rows.order, ids_sorted)
                indices[bits][start:stop] = ids.to(torch.uint8).cpu()

    # An all-zero matrix has zero energy, and every codebook reconstructs it exactly.

    relative_error = {b: error_sum[b] / energy_sum if energy_sum > 0.0 else 0.0 for b in bit_widths}
    return LayerQuantization(indices, luts, relative_error, lloyd_iterations)
