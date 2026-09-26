"""``run_quantization`` on the real Granite checkpoint, both modes (spec 0011)."""

import pytest

from anyprec.quantization.pipeline import run_quantization
from tests.integration.conftest import GraniteRun, compose_quantize, small_run_overrides

pytestmark = [pytest.mark.gpu, pytest.mark.network, pytest.mark.slow]


def test_both_granite_modes_share_one_fisher_and_a_second_run_reuses_everything(
    granite_run: GraniteRun,
) -> None:
    """
    Given: Granite quantized in incremental, then standalone mode, into one temporary cache.
    When: the incremental run is repeated with the identical config.
    Then: the standalone run reused the incremental run's Fisher, and the repeated run
        reports both artifacts reused under the same keys.
    """
    incremental = granite_run.outcomes["incremental"]
    standalone = granite_run.outcomes["standalone"]
    cfg = compose_quantize(small_run_overrides(granite_run.base_dir))

    again = run_quantization(cfg, granite_run.base_dir / "quantize-again")

    assert not incremental.fisher_reused and standalone.fisher_reused
    assert standalone.fisher_key == incremental.fisher_key
    assert again.fisher_reused and again.quantized_reused
    assert again.quantized_key == incremental.quantized_key
