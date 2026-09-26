"""Check a full run against the hard criteria and reference bands of spec 0011.

Usage::

    python evaluation/check_acceptance.py outputs/evaluate/<date>/<time>/results.json

The script only reads: ``results.json``, both quantized artifacts, and the Fisher manifest, all
through the pydantic schemas and ``ArtifactStore``. It prints one line per criterion and exits 1
if any fails, so later waves can rerun it as a regression check.
"""

import argparse
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Literal

from anyprec.artifacts.keys import quantized_snapshot
from anyprec.artifacts.manifest import ArtifactStats, ModuleEntry
from anyprec.artifacts.store import ArtifactMismatchError, ArtifactNotFoundError, ArtifactStore
from anyprec.config.schemas import QuantizerMode
from anyprec.evaluation.results import QuantizedEntry, Results

type ModuleErrors = Mapping[str, Mapping[int, float]]

GRANITE_LAYERS: int = 28

# [out_features, in_features] per block (ADR 0002), in the order module discovery visits them.

GRANITE_SHAPES: dict[str, tuple[int, int]] = {
    "shared_mlp.input_linear": (4096, 1024),
    "shared_mlp.output_linear": (1024, 2048),
    "self_attn.q_proj": (1024, 1024),
    "self_attn.k_proj": (256, 1024),
    "self_attn.v_proj": (256, 1024),
    "self_attn.o_proj": (1024, 1024),
}
GRANITE_MODULES: list[ModuleEntry] = [
    ModuleEntry(name=f"model.layers.{layer}.{suffix}", shape=shape)
    for layer in range(GRANITE_LAYERS)
    for suffix, shape in GRANITE_SHAPES.items()
]

NOTEBOOK_MEAN_LOSS: float = 3.22
MEAN_LOSS_TOLERANCE: float = 0.02
STANDALONE_SLACK: float = 0.01
AGREEMENT_FLOOR: float = 0.95

# Spec 0008's analytic bits per weight: bit-width to (quantized layers, whole model).

EXPECTED_BITS: dict[int, tuple[float, float]] = {
    3: (3.1103, 6.8713),
    4: (4.2206, 7.6576),
    5: (5.4412, 8.5221),
    6: (6.8824, 9.5427),
    7: (8.7647, 10.8758),
    8: (11.5294, 12.8339),
}
EXPECTED_PARENT: tuple[float, float] = (14.9485, 15.2553)


@dataclass(frozen=True)
class Band:
    """One notebook-02 measurement and the band a package value must fall in.

    :param module: Qualified module name.
    :param mode: Quantizer mode of the artifact holding the value.
    :param bits: Bit-width.
    :param notebook: The notebook's relative error.
    :param kind: ``relative`` accepts ``|x / notebook - 1| <= tolerance``; ``factor`` accepts
        ``notebook / tolerance <= x <= notebook * tolerance``.
    :param tolerance: The band's width.
    """

    module: str
    mode: QuantizerMode
    bits: int
    notebook: float
    kind: Literal["relative", "factor"]
    tolerance: float


BANDS: list[Band] = [
    Band("model.layers.14.self_attn.q_proj", "incremental", 3, 6.63e-3, "relative", 0.10),
    Band("model.layers.14.self_attn.q_proj", "incremental", 8, 3.18e-6, "factor", 2.0),
    Band("model.layers.14.self_attn.q_proj", "standalone", 8, 9.18e-7, "factor", 2.0),
    Band("model.layers.2.shared_mlp.output_linear", "incremental", 3, 1.48e-2, "relative", 0.10),
    Band("model.layers.2.shared_mlp.output_linear", "incremental", 8, 1.16e-5, "factor", 2.0),
    Band("model.layers.2.shared_mlp.output_linear", "standalone", 8, 7.22e-6, "factor", 2.0),
]


@dataclass(frozen=True)
class Check:
    """The outcome of one criterion.

    :param name: The criterion, as worded in spec 0011.
    :param passed: Whether it holds.
    :param detail: The measured values, or the offending modules.
    """

    name: str
    passed: bool
    detail: str


def error_criteria(incremental: ModuleErrors, standalone: ModuleErrors) -> list[Check]:
    """Check the three per-module relative-error criteria of ADR 0003, Section 4.

    :param incremental: Module to bit-width to relative error, incremental artifact.
    :param standalone: The same for the standalone artifact.
    :return: One check per criterion.
    """
    widths = sorted(next(iter(incremental.values())))
    seed = widths[0]

    # Property 1: every added bit splits at least one segment, so the error strictly falls.

    not_decreasing = [
        name
        for name, errors in incremental.items()
        if any(errors[b + 1] >= errors[b] for b in widths[:-1])
    ]

    # Property 4: the incremental seed is the standalone fit, and above the seed, nesting can
    # only cost accuracy, up to float16 rounding and Lloyd's local optimum.

    seed_differs = [n for n in incremental if incremental[n][seed] != standalone[n][seed]]
    inverted = [
        f"{n}@{b}"
        for n in incremental
        for b in widths[1:]
        if incremental[n][b] < standalone[n][b] * (1.0 - STANDALONE_SLACK)
    ]
    return [
        Check(
            f"incremental relative error strictly decreases from {seed} to {widths[-1]} bits",
            not not_decreasing,
            _offenders(not_decreasing, len(incremental)),
        ),
        Check(
            f"incremental and standalone {seed}-bit relative errors are equal",
            not seed_differs,
            _offenders(seed_differs, len(incremental)),
        ),
        Check(
            f"incremental error >= standalone error above {seed} bits (1% slack)",
            not inverted,
            _offenders(inverted, len(incremental) * (len(widths) - 1)),
        ),
    ]


def metric_criteria(results: Results) -> list[Check]:
    """Check the KL, agreement, perplexity, and bits-per-weight criteria on ``results.json``.

    :param results: The evaluation run's results.
    :return: One check per criterion and mode.
    """
    checks: list[Check] = []
    reference = {p.dataset: p.perplexity for p in results.reference.perplexity}
    for mode in results.config.modes:
        entries = sorted((e for e in results.entries if e.mode == mode), key=lambda e: e.bits)
        low, high = entries[0], entries[-1]

        # KL is the objective's target, so each added bit must lower it (ADR 0003, Section 1).

        kl = [e.kl_mean for e in entries]
        checks.append(
            Check(
                f"{mode}: mean KL on {low.kl_dataset} decreases with every added bit",
                all(later < earlier for earlier, later in pairwise(kl)),
                ", ".join(f"{e.bits}:{e.kl_mean:.3e}" for e in entries),
            )
        )

        # Agreement is a coarse 0/1 statistic, so only its endpoints are gated.

        checks.append(
            Check(
                f"{mode}: top-1 agreement at {high.bits} bits exceeds {low.bits}-bit and "
                f"{AGREEMENT_FLOOR}",
                high.top1_agreement > max(low.top1_agreement, AGREEMENT_FLOOR),
                f"{low.bits}:{low.top1_agreement:.4f}, {high.bits}:{high.top1_agreement:.4f}",
            )
        )
        checks.append(_perplexity_check(mode, low, reference))
    checks.append(_bits_check(results))
    return checks


def band_checks(errors: Mapping[QuantizerMode, ModuleErrors]) -> list[Check]:
    """Compare the package's errors with notebook 02's measurements.

    :param errors: Mode to module to bit-width to relative error.
    :return: One check per band.
    """
    checks: list[Check] = []
    for band in BANDS:
        value = errors[band.mode][band.module][band.bits]
        if band.kind == "relative":
            passed = abs(value / band.notebook - 1.0) <= band.tolerance
            accepted = f"±{band.tolerance:.0%}"
        else:
            passed = band.notebook / band.tolerance <= value <= band.notebook * band.tolerance
            accepted = f"within a factor of {band.tolerance:g}"
        checks.append(
            Check(
                f"band {band.module} {band.mode} {band.bits}-bit",
                passed,
                f"{value:.3e} vs notebook {band.notebook:.3e} ({accepted})",
            )
        )
    return checks


def _perplexity_check(
    mode: QuantizerMode, entry: QuantizedEntry, reference: Mapping[str, float]
) -> Check:
    """A model at the lowest evaluated width cannot beat its own reference on any dataset."""
    quantized = {p.dataset: p.perplexity for p in entry.perplexity}
    return Check(
        f"{mode}: {entry.bits}-bit perplexity is at or above the reference on every dataset",
        all(quantized[name] >= reference[name] for name in reference),
        ", ".join(f"{name}: {quantized[name]:.3f} vs {reference[name]:.3f}" for name in reference),
    )


def _bits_check(results: Results) -> Check:
    """Compare ``results.json``'s bits per weight with spec 0008's table to four decimals."""
    report = results.bits
    parent = (report.parent, report.parent_whole_model)
    wrong = [f"{b}: missing" for b in EXPECTED_BITS if b not in report.per_bits]
    wrong += [
        f"{b}: {report.per_bits[b]:.4f}/{report.per_bits_whole_model[b]:.4f}"
        for b, expected in EXPECTED_BITS.items()
        if b in report.per_bits
        and (round(report.per_bits[b], 4), round(report.per_bits_whole_model[b], 4)) != expected
    ]
    if tuple(round(v, 4) for v in parent) != EXPECTED_PARENT:
        wrong.append(f"parent: {parent[0]:.4f}/{parent[1]:.4f}")
    return Check("bits per weight equal spec 0008's table", not wrong, "; ".join(wrong) or "all")


def _offenders(names: list[str], total: int) -> str:
    """Summarize failing items: their count, and the first few by name."""
    if not names:
        return f"all {total} hold"
    return f"{len(names)} of {total} fail, e.g. {', '.join(names[:3])}"


def _load_stats(store: ArtifactStore, results: Results, mode: QuantizerMode) -> ArtifactStats:
    """Load one mode's artifact against the expected Granite modules and return its statistics.

    :param store: The run's artifact store.
    :param results: The run's results, which record the artifact keys.
    :param mode: The quantizer mode.
    :return: The artifact's per-module statistics.
    :raises ArtifactNotFoundError: If the artifact is missing.
    :raises ArtifactMismatchError: If it does not list the 168 Granite modules, in order.
    :raises ValueError: If ``results.json`` has no entry for the mode, or no ``stats.json``.
    """
    keys = {e.artifact_key for e in results.entries if e.mode == mode}
    if len(keys) != 1:
        raise ValueError(f"results.json should hold exactly one {mode} artifact key, got {keys}")
    quantizer = results.config.quantizer.with_mode(mode)
    snapshot = quantized_snapshot(results.fisher_key, quantizer)
    artifact = store.load_quantized(keys.pop(), snapshot, GRANITE_MODULES)
    if artifact.stats is None:
        raise ValueError(f"the {mode} artifact has no stats.json")
    return artifact.stats


def main(argv: list[str] | None = None) -> int:
    """Run every check on one ``results.json`` and print one line per criterion.

    :param argv: Command-line arguments, without the program name.
    :return: 0 if every criterion and band passes, else 1.
    """
    parser = argparse.ArgumentParser(description="Check a full run against spec 0011.")
    parser.add_argument("results", type=Path, help="path to the run's results.json")
    args = parser.parse_args(argv)
    results_path: Path = args.results
    results = Results.model_validate_json(results_path.read_text(encoding="utf-8"))
    store = ArtifactStore(results.config.output.cache_dir)

    # Loading through the store against the expected module list is itself the first criterion:
    # a missing artifact, a wrong module, or a wrong shape stops the check here.

    modules_check = Check("both artifacts list the 168 Granite modules, in order", True, "loaded")
    try:
        stats: dict[QuantizerMode, ArtifactStats] = {
            mode: _load_stats(store, results, mode) for mode in results.config.modes
        }
        fisher = store.load_fisher_manifest(results.fisher_key)
    except (ArtifactNotFoundError, ArtifactMismatchError, ValueError) as error:
        modules_check = Check(modules_check.name, False, str(error))
        print(f"FAIL  {modules_check.name}: {modules_check.detail}")
        return 1

    # Hard criteria first, then the notebook bands; any failure fails the run.

    errors: dict[QuantizerMode, ModuleErrors] = {m: s.relative_error for m, s in stats.items()}
    loss_gap = abs(fisher.mean_loss - NOTEBOOK_MEAN_LOSS)
    checks = [
        modules_check,
        Check(
            f"Fisher mean loss within {MEAN_LOSS_TOLERANCE} of {NOTEBOOK_MEAN_LOSS} nats/token",
            loss_gap <= MEAN_LOSS_TOLERANCE,
            f"{fisher.mean_loss:.4f} over {fisher.num_sequences} sequences",
        ),
        *error_criteria(errors["incremental"], errors["standalone"]),
        Check(
            "same-weights check: KL <= 1e-6 and agreement 1",
            True,
            "enforced by run_evaluation, which raises SameWeightsError before writing results",
        ),
        *metric_criteria(results),
        *band_checks(errors),
    ]
    for check in checks:
        print(f"{'PASS' if check.passed else 'FAIL'}  {check.name}: {check.detail}")
    failed = sum(not check.passed for check in checks)
    print(f"{len(checks) - failed} of {len(checks)} checks pass")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
