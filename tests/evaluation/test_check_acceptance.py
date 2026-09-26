"""Tests for the acceptance checker's criteria (spec 0011), on synthetic values and a tiny run."""

from pathlib import Path

from evaluation.check_acceptance import (
    BANDS,
    ModuleErrors,
    band_checks,
    error_criteria,
    metric_criteria,
)

from anyprec.config.schemas import QuantizerMode
from anyprec.evaluation.pipeline import run_evaluation
from anyprec.quantization.pipeline import run_quantization
from tests import factories
from tests.offline import OfflineLoaders


def _halving(start: float) -> dict[int, float]:
    """A 3-to-8-bit error curve that halves with every added bit."""
    return {bits: start / 2 ** (bits - 3) for bits in range(3, 9)}


def test_error_criteria_pass_nested_errors_and_name_each_offending_module() -> None:
    """
    Given: standalone errors that halve per bit, an incremental copy that is 3% worse above
        the seed, and a second incremental module whose 8-bit error does not improve.
    When: the error criteria run on the healthy pair, then on the broken one.
    Then: all three pass for the healthy pair; for the broken one, strict decrease fails and
        names the module, while seed equality and the 1% slack still pass.
    """
    standalone: ModuleErrors = {"a": _halving(1e-2), "b": _halving(2e-2)}
    healthy = {
        name: {b: e * (1.0 if b == 3 else 1.03) for b, e in errors.items()}
        for name, errors in standalone.items()
    }
    stalled = {**healthy, "b": {**healthy["b"], 8: healthy["b"][7]}}

    assert all(check.passed for check in error_criteria(healthy, standalone))
    decreasing, seed_equal, slack = error_criteria(stalled, standalone)
    assert not decreasing.passed and "b" in decreasing.detail
    assert seed_equal.passed and slack.passed


def test_error_criteria_reject_an_incremental_error_below_the_standalone_slack() -> None:
    """
    Given: an incremental 5-bit error 2% below the standalone one, beyond the 1% slack.
    When: the error criteria run.
    Then: only the incremental-at-least-standalone criterion fails, naming module and width.
    """
    standalone: ModuleErrors = {"a": _halving(1e-2)}
    incremental = {"a": {**standalone["a"], 5: standalone["a"][5] * 0.98}}

    decreasing, seed_equal, slack = error_criteria(incremental, standalone)

    assert decreasing.passed and seed_equal.passed
    assert not slack.passed and "a@5" in slack.detail


def test_band_checks_accept_the_notebook_values_and_reject_values_outside_each_band() -> None:
    """
    Given: every band's notebook value, and then each value moved just outside its band.
    When: the band checks run on both.
    Then: every band passes at the notebook value and fails outside it.
    """
    at_notebook: dict[QuantizerMode, dict[str, dict[int, float]]] = {
        "incremental": {},
        "standalone": {},
    }
    outside: dict[QuantizerMode, dict[str, dict[int, float]]] = {
        "incremental": {},
        "standalone": {},
    }
    for band in BANDS:
        shift = 1.0 + 1.01 * band.tolerance if band.kind == "relative" else 1.01 * band.tolerance
        at_notebook[band.mode].setdefault(band.module, {})[band.bits] = band.notebook
        outside[band.mode].setdefault(band.module, {})[band.bits] = band.notebook * shift

    assert all(check.passed for check in band_checks(at_notebook))
    assert not any(check.passed for check in band_checks(outside))


def test_metric_criteria_flag_a_kl_curve_that_rises_with_bits(
    tmp_path: Path, offline_loaders: OfflineLoaders
) -> None:
    """
    Given: real tiny-model results, with the incremental 4-bit KL raised above the 3-bit KL.
    When: the metric criteria run.
    Then: the incremental KL criterion fails, and the tiny model's bits per weight fail the
        Granite table, while the standalone KL criterion is unaffected.
    """
    for mode in ("incremental", "standalone"):
        run_quantization(factories.quantize_run_config(tmp_path, mode), tmp_path / mode)
    results = run_evaluation(factories.evaluate_run_config(tmp_path), tmp_path / "evaluate")
    entries = [
        e.model_copy(update={"kl_mean": 1.0}) if (e.mode, e.bits) == ("incremental", 4) else e
        for e in results.entries
    ]

    checks = {c.name: c for c in metric_criteria(results.model_copy(update={"entries": entries}))}

    assert not checks["incremental: mean KL on wikitext2 decreases with every added bit"].passed
    assert checks["standalone: mean KL on wikitext2 decreases with every added bit"].passed
    assert not checks["bits per weight equal spec 0008's table"].passed
