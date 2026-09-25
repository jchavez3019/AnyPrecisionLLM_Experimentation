"""Tests for the results schema and its entry builders (spec 0008)."""

import math
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from anyprec.config.schemas import EvalDatasetName, EvaluateRunConfig
from anyprec.evaluation.bits import BitsReport, bits_report
from anyprec.evaluation.metrics import MetricSummary
from anyprec.evaluation.results import (
    RESULTS_SCHEMA_VERSION,
    QuantizedEntry,
    Results,
    make_quantized_entry,
    make_reference_entry,
)


def _summary(mean_nll: float, with_kl: bool) -> MetricSummary:
    """A summary with plausible values; KL fields only when ``with_kl``."""
    return MetricSummary(
        perplexity=math.exp(mean_nll),
        mean_nll=mean_nll,
        num_chunks=2,
        num_tokens=64,
        kl_mean=0.05 if with_kl else None,
        kl_quantile=0.4 if with_kl else None,
        top1_agreement=0.9 if with_kl else None,
    )


def _summaries(mean_nll: float) -> dict[EvalDatasetName, MetricSummary]:
    """WikiText-2 with KL fields, the KL dataset of the tiny config, and C4 without."""
    return {"wikitext2": _summary(mean_nll, True), "c4": _summary(mean_nll + 0.5, False)}


def _report() -> BitsReport:
    """Bits of the tiny model's 12 layers at 2 to 4 bits."""
    return bits_report([(64, 64)] * 12, 90_432, 2, 4)


def _results(config: EvaluateRunConfig, entries: list[QuantizedEntry]) -> Results:
    """A complete results object around ``entries``."""
    return Results(
        schema_version=RESULTS_SCHEMA_VERSION,
        config=config,
        model_id=config.model.model_id,
        revision=config.model.revision,
        fisher_key="f" * 64,
        versions={"torch": "2.x"},
        created_at=datetime(2026, 9, 25, tzinfo=UTC),
        bits=_report(),
        reference=make_reference_entry(_summaries(3.0)),
        entries=entries,
        seconds=12.5,
    )


def test_quantized_entry_takes_kl_from_the_kl_dataset_and_bits_from_the_report() -> None:
    """
    Given: summaries where only WikiText-2 carries KL fields, and a bits report.
    When: the 3-bit incremental entry is built.
    Then: perplexity lists both datasets in order, KL fields come from WikiText-2, and bits
        per weight are the report's 3-bit figures.
    """
    report = _report()

    entry = make_quantized_entry("incremental", 3, "a" * 64, _summaries(3.5), "wikitext2", report)

    assert [p.dataset for p in entry.perplexity] == ["wikitext2", "c4"]
    assert (entry.kl_mean, entry.kl_quantile, entry.top1_agreement) == (0.05, 0.4, 0.9)
    assert entry.bits_per_weight == report.per_bits[3]
    assert entry.bits_per_weight_whole_model == report.per_bits_whole_model[3]


def test_quantized_entry_refuses_a_kl_dataset_without_kl_values() -> None:
    """
    Given: summaries where C4 has no KL fields.
    When: an entry is built with C4 as the KL dataset.
    Then: ValueError names the dataset, since the loop must have skipped the reference.
    """
    with pytest.raises(ValueError, match="c4 summary has no KL"):
        make_quantized_entry("incremental", 3, "a" * 64, _summaries(3.5), "c4", _report())


def test_results_round_trip_through_json_with_integer_bit_keys(
    evaluate_run_config: EvaluateRunConfig, tmp_path: Path
) -> None:
    """
    Given: results with entries for both modes at every configured bit-width.
    When: they are written to results.json and read back.
    Then: the parsed results equal the original, including the int-keyed bits report.
    """
    entries = [
        make_quantized_entry(
            mode, bits, "a" * 64, _summaries(3.0 + 1 / bits), "wikitext2", _report()
        )
        for mode in evaluate_run_config.modes
        for bits in evaluate_run_config.eval.bits
    ]
    results = _results(evaluate_run_config, entries)
    path = tmp_path / "results.json"

    path.write_text(results.model_dump_json(indent=2))
    loaded = Results.model_validate_json(path.read_text())

    assert loaded == results
    assert list(loaded.bits.per_bits) == [2, 3, 4]


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"bits": 5}, r"\(incremental, 5\) is not in the config"),
        ({"kl_dataset": "c4"}, "KL dataset"),
    ],
)
def test_results_reject_an_entry_the_config_did_not_ask_for(
    evaluate_run_config: EvaluateRunConfig, update: dict[str, object], message: str
) -> None:
    """
    Given: an entry at 5 bits when the config evaluates 2 to 4, or with the wrong KL dataset.
    When: results are built around it.
    Then: validation fails, so a results file always describes its own config.
    """
    entry = make_quantized_entry(
        "incremental", 3, "a" * 64, _summaries(3.0), "wikitext2", _report()
    )
    stray = entry.model_copy(update=update)

    with pytest.raises(ValidationError, match=message):
        _results(evaluate_run_config, [stray])
