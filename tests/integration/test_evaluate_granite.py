"""``run_evaluation`` on the real Granite artifacts, two chunks per dataset (spec 0011)."""

import pytest

from anyprec.evaluation.pipeline import RESULTS_FILE, run_evaluation
from anyprec.evaluation.results import Results
from tests.integration.conftest import GraniteRun, compose_evaluate, small_run_overrides

pytestmark = [pytest.mark.gpu, pytest.mark.network, pytest.mark.slow]


def test_granite_evaluation_passes_its_startup_checks_and_writes_valid_results(
    granite_run: GraniteRun,
) -> None:
    """
    Given: both Granite artifacts in the session's cache.
    When: run_evaluation runs the shipped protocol, capped at two chunks per dataset.
    Then: the sliced-head and same-weights checks pass (they raise otherwise), results.json
        re-validates with one entry per (mode, bits), and in each mode 8-bit KL is below 3-bit KL.
    """
    overrides = [*small_run_overrides(granite_run.base_dir), "eval.max_chunks=2"]
    cfg = compose_evaluate(overrides)
    run_dir = granite_run.base_dir / "evaluate"

    results = run_evaluation(cfg, run_dir)

    assert Results.model_validate_json((run_dir / RESULTS_FILE).read_text()) == results
    assert len(results.entries) == len(cfg.modes) * len(cfg.eval.bits)
    assert results.fisher_key == granite_run.outcomes["incremental"].fisher_key
    for mode in cfg.modes:
        by_bits = {e.bits: e for e in results.entries if e.mode == mode}
        assert by_bits[8].kl_mean < by_bits[3].kl_mean
