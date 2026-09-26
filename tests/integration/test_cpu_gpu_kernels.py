"""The k-means kernels on the CPU and the GPU agree where they must (spec 0011)."""

import pytest
import torch

from anyprec.config.schemas import QuantizerConfig
from anyprec.quantization.layer import LayerQuantization, quantize_layer
from tests.integration.conftest import compose_quantize
from tests.quantization import oracles

pytestmark = pytest.mark.gpu

ROWS: int = 1024
COLUMNS: int = 1024
SEED_BITS_TOLERANCE: float = 0.02
HIGHER_BITS_TOLERANCE: float = 0.10


def _real_sized_layer() -> tuple[torch.Tensor, torch.Tensor]:
    """A random ``[1024, 1024]`` weight and a heavy-tailed positive Fisher, from a local generator.

    :return: ``(weight, fisher)``, float32 on the CPU.
    """
    generator = torch.Generator().manual_seed(0)
    weight = 0.02 * torch.randn(ROWS, COLUMNS, generator=generator)
    fisher = torch.exp(2.0 * torch.randn(ROWS, COLUMNS, generator=generator))
    return weight, fisher


def _assert_nested(
    result: LayerQuantization, weight: torch.Tensor, fisher: torch.Tensor, cfg: QuantizerConfig
) -> None:
    """Check that every width's recorded error is reproduced by the right-shifted parent indices.

    :param result: An incremental ``quantize_layer`` result.
    :param weight: Its ``[m, n]`` weights.
    :param fisher: Its ``[m, n]`` raw Fisher diagonal.
    :param cfg: Its quantizer settings.
    """
    f = oracles.conditioned_fisher(weight, fisher)
    energy = float((f * weight.double().square()).sum())
    parent = result.indices[cfg.parent_bits]
    for bits in range(cfg.seed_bits, cfg.parent_bits + 1):
        ids = parent >> (cfg.parent_bits - bits)
        cost = oracles.reconstruction_cost(weight, f, result.luts[bits], ids)
        assert cost / energy == pytest.approx(result.relative_error[bits], rel=1e-5), bits


def test_quantize_layer_on_cpu_and_gpu_agrees_in_structure_nesting_and_error(
    cuda_device: torch.device,
) -> None:
    """
    Given: one real-sized random matrix and the shipped incremental 3-to-8-bit quantizer.
    When: quantize_layer runs on the CPU and on the GPU with the same seed.
    Then: both results have the same stored widths, shapes, and dtypes; both are nested; and
        their relative errors agree within 2% at 3 bits and 10% above, since k-means++ draws
        differ between CPU and CUDA generators.
    """
    cfg = compose_quantize([]).quantizer
    weight, fisher = _real_sized_layer()

    cpu = quantize_layer(weight, fisher, cfg, generator_seed=0)
    gpu = quantize_layer(weight.to(cuda_device), fisher.to(cuda_device), cfg, generator_seed=0)

    assert sorted(cpu.indices) == sorted(gpu.indices) == [cfg.parent_bits]
    for bits in range(cfg.seed_bits, cfg.parent_bits + 1):
        assert cpu.luts[bits].shape == gpu.luts[bits].shape == (ROWS, 2**bits)
        assert cpu.luts[bits].dtype == gpu.luts[bits].dtype == torch.float16
    assert cpu.indices[cfg.parent_bits].dtype == gpu.indices[cfg.parent_bits].dtype == torch.uint8
    _assert_nested(cpu, weight, fisher, cfg)
    _assert_nested(gpu, weight, fisher, cfg)
    for bits in range(cfg.seed_bits, cfg.parent_bits + 1):
        tolerance = SEED_BITS_TOLERANCE if bits == cfg.seed_bits else HIGHER_BITS_TOLERANCE
        assert gpu.relative_error[bits] == pytest.approx(cpu.relative_error[bits], rel=tolerance)
