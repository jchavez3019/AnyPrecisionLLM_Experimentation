"""The empirical Fisher on the real Granite checkpoint and GPU (spec 0011)."""

import pytest
import torch

from anyprec.data.calibration import make_encoder, sample_calibration
from anyprec.data.hub import load_texts
from anyprec.models.discovery import find_quantizable_linears
from anyprec.models.loading import load_model, load_tokenizer
from anyprec.sensitivity.fisher import estimate_fisher
from anyprec.utils.dtypes import torch_dtype
from tests.integration.conftest import GIB, NUM_SEQUENCES, compose_quantize

pytestmark = [pytest.mark.gpu, pytest.mark.network]

PEAK_BUDGET_GIB: float = 3.5


def test_fisher_on_eight_granite_sequences_is_valid_and_fits_the_memory_budget(
    cuda_device: torch.device,
) -> None:
    """
    Given: Granite in bfloat16 on the GPU and 8 C4 calibration sequences.
    When: the empirical Fisher of all 168 linears is estimated.
    Then: every requires_grad flag is restored and no gradient is left, every diagonal is
        finite and non-negative, and peak GPU memory stays below 3.5 GiB.
    """
    cfg = compose_quantize([f"calibration.num_sequences={NUM_SEQUENCES}"])
    cal = cfg.calibration
    texts = load_texts(cal.path, cal.name, cal.data_files, cal.split, cal.text_field)
    calibration = sample_calibration(texts, make_encoder(load_tokenizer(cfg.model)), cal)
    model = load_model(cfg.model, torch_dtype(cfg.model.dtype), cuda_device)
    targets = find_quantizable_linears(model, cfg.model.quantizable_modules)
    flags = {name: p.requires_grad for name, p in model.named_parameters()}
    torch.cuda.reset_peak_memory_stats(cuda_device)

    result = estimate_fisher(model, calibration, targets)

    peak_gib = torch.cuda.max_memory_allocated(cuda_device) / GIB
    assert {name: p.requires_grad for name, p in model.named_parameters()} == flags
    assert all(p.grad is None for p in model.parameters())
    assert list(result.diagonals) == list(targets)
    for name, diagonal in result.diagonals.items():
        assert bool(torch.isfinite(diagonal).all()) and bool((diagonal >= 0).all()), name
    assert result.losses.shape == (NUM_SEQUENCES,)
    assert peak_gib < PEAK_BUDGET_GIB, f"peak {peak_gib:.2f} GiB"
