# Spec 0016: Comparison Run and Results

- Status: Proposed
- Wave: [1](index.md)
- Implements: [ADR 0008](../../adr/0008-batched-and-per-token-fisher.md) (Evaluation, Diagnostics), [ADR 0004](../../adr/0004-evaluation-protocol.md) (metrics)

This spec defines the wave's experiment. It reruns wave 0's acceptance as a regression gate and a noise floor, quantizes with two token-mode Fishers, and compares all three with a script whose output becomes `results.md`. The wave passes on a sound comparison, whichever estimator wins.

## Files

The comparison logic is a library module, so it is unit-tested offline. The script only parses arguments and prints.

| File | Contents |
| --- | --- |
| `src/anyprec/evaluation/comparison.py` | `RunInputs`, `regression_checks`, `quality_rows`, `fisher_diagnostics`, `estimator_agreement`, `cross_weighted_error`, `Comparison` |
| `src/anyprec/inference/precision.py` | Public `dequantized_weight(artifact, name, bits)`, which `set_precision` now calls |
| `evaluation/compare_fishers.py` | The argparse entry point |
| `tests/evaluation/test_comparison.py` | The tests below |
| `docs/waves/wave_1/results.md` | Written from the script's output after the run |

## Run matrix

Every run uses the default calibration set (100 C4 sequences of 512 tokens, `calibration.seed` unchanged), the default quantizer, and `fisher.split_half=true`. `split_half` is not in the key, so the `sequence` run still reproduces wave 0's estimator.

| Run | Fisher | Modes | Commands |
| --- | --- | --- | --- |
| `sequence` | ADR 0003 default | incremental and standalone | `quantize_any_precision.py fisher.split_half=true`, the same with `quantizer.mode=standalone`, then `evaluate_any_precision.py` |
| `token` | token, all positions, `batched_backward` | incremental | `quantize_any_precision.py fisher.granularity=token fisher.token_method=batched_backward fisher.batch_size=Q fisher.split_half=true`, then `evaluate_any_precision.py fisher.granularity=token modes=[incremental]` |
| `token_k64` | token, `positions_per_sequence: 64`, `batched_backward` | incremental | as `token`, plus `fisher.positions_per_sequence=64` in both commands |

The runs happen in this order:

1. **Timing probe.** Run the `token` quantize with `calibration.num_sequences=2`, at `fisher.batch_size` 2 and then 4, into a throwaway `output.cache_dir`. Choose the larger $Q$ whose `fisher_peak_gpu_gib` in `quantize_summary.json` is below 5.5. Scale `fisher_seconds` by 50 to estimate the full run. Record $Q$ and the estimate in `results.md`.
2. **The `sequence` run.** Then run `python evaluation/check_acceptance.py <results.json>`, which must pass unchanged, and the regression gate below. If either fails, stop and find the cause before running the token Fishers. Do not widen a tolerance to make the gate pass.
3. **The `token` and `token_k64` runs.**
4. **The comparison script,** then `results.md`.

The estimated time is about 6 to 7 hours of GPU time. That covers the standalone k-means (38 min in wave 0), four evaluation passes of about 1 hour each, and up to 1.5 hours for the token Fishers. The GPU throttles under sustained load (wave 0 [results](../wave_0/results.md)), so leave the laptop plugged in and cool.

## Regression gate

The `sequence` run uses wave 0's estimator under a new key, so its metrics differ from wave 0's only through GPU non-determinism in the Fisher pass (ADR 0003). The gate checks that the rewiring changed nothing else. The differences it measures are the noise floor for every comparison later on.

| Check | Tolerance |
| --- | --- |
| Reference WikiText-2 and C4 perplexity | relative difference at most `1e-4`; the reference does not depend on the Fisher |
| Fisher mean loss | within `1e-3` nats of wave 0's 3.2242 |
| Mean KL, each mode and bit-width | relative difference at most 3% |
| Top-1 agreement, each mode and bit-width | absolute difference at most 0.005 |
| WikiText-2 perplexity, each mode and bit-width | relative difference at most 0.5% |
| Per-module relative error, each mode and bit-width | median over modules of $\lvert r_{\text{new}} / r_{\text{wave 0}} - 1 \rvert$ at most 1% |

These tolerances are set well above the third-significant-digit differences ADR 0003 records for GPU Fisher reruns, and well below the differences that matter: the smallest effect wave 0 reports is the 5-bit nesting gap, a KL ratio of 1.27. `results.md` reports the measured differences, not only pass or fail.

Wave 0's `results.json` no longer validates as `Results` (spec 0012, Compatibility). `regression_checks` therefore validates only its `reference` and `entries` fields, with the existing `ReferenceEntry` and `QuantizedEntry` models. It reads wave 0's Fisher `mean_loss` from the raw manifest JSON, and wave 0's per-module errors from the `stats.json` of the artifacts named in `entries[].artifact_key`.

## Measurements

Every table below compares estimators on the same calibration set. Sequence and token Fishers differ by a constant scale (ADR 0008), so diagonals are only ever compared by rank or used as weights, never subtracted.

1. **Answer quality.** For each run, and each bit-width from 3 to 8 in incremental mode: mean KL, top-1 agreement, and WikiText-2 and C4 perplexity, plus each token run's difference from `sequence`. A difference counts as a change only if it exceeds twice the regression gate's measured difference at that bit-width. That is a single-rerun estimate of noise, and `results.md` says so.
2. **Cost.** Fisher seconds, Fisher-stage and k-means peak GPU memory, and the chosen `batch_size`, from the manifests and summaries.
3. **Split-half stability.** For each run, the median, 10th percentile and minimum over modules of `spearman_half` and `spearman_full`, overall and per module type (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `input_linear`, `output_linear`). ADR 0008's variance model predicts that token mode's halves agree much more closely than sequence mode's.
4. **Estimator agreement.** For each module, the Spearman correlation between each run's Fisher and the `token` Fisher, which is the least noisy; the table gives the median and minimum over modules. How far `sequence` falls short of 1 shows how much its ranking differs. How close `token_k64` comes to 1 shows what the subsample loses.
5. **Cross-weighted error.** Each incremental artifact's Fisher-weighted relative error $\sum f (w - \hat w)^2 / \sum f w^2$ at each bit-width, computed under both the `sequence` and the `token` Fisher, with the median taken over modules. Each artifact is expected to win under the Fisher it was fit to. Whether the `token` artifact also wins under the `sequence` weighting is the informative part. This isolates the Fisher's effect on k-means from the noise of the evaluation sets.

## Interface

`compare_fishers.py` names each run and points at its `results.json`. The Fisher and quantized artifacts are found from the keys inside it:

```bash
python evaluation/compare_fishers.py \
  --wave0 outputs/evaluate/2026-09-25/21-38-07/results.json \
  --run sequence=outputs/evaluate/<date>/<time>/results.json \
  --run token=outputs/evaluate/<date>/<time>/results.json \
  --run token_k64=outputs/evaluate/<date>/<time>/results.json \
  --baseline sequence --anchor token \
  --out outputs/compare/wave_1.json
```

- `--baseline` names the run checked against `--wave0` and used as the reference for differences.
- `--anchor` names the Fisher used for estimator agreement and as the second weighting in cross-weighted error.
- The script writes every measurement to `--out` as JSON, from the pydantic `Comparison` model. It prints each table as markdown, ready to copy into `results.md`.
- It exits non-zero if the regression gate fails, and prints the failing checks.

```python
def cross_weighted_error(
    weights: Mapping[str, torch.Tensor],
    artifact: QuantizedArtifact,
    fisher: Mapping[str, torch.Tensor],
    bit_widths: Sequence[int],
) -> dict[int, dict[str, float]]:
    """Fisher-weighted relative error of every module at every bit-width, under ``fisher``.

    ``weights`` are the original ``[m, n]`` weights. Each module is dequantized with
    ``dequantized_weight`` and reduced in float64, one module at a time on the CPU.
    """
```

The comparison runs on the CPU. It holds the model's weights (0.7 GB) and two Fishers (1 GB each) at once, and loads the remaining Fisher only for the agreement table.

## results.md

The page has the same shape as wave 0's [results](../wave_0/results.md), so the two can be read side by side. It has these sections, in this order:

1. **Header facts.** Run dates, versions, hardware, the three `results.json` paths, every Fisher and quantized key, and the `check_acceptance` outcome.
2. **Run matrix.** One row per run, with a `Status` column.
3. **Regression gate.** One row per check, with the measured difference and a `Result` column.
4. **Wall-clock time and memory.** One row per stage per run, next to ADR 0008's estimates.
5. **Answer quality.** Measurement 1, with wave 0's incremental numbers for reference.
6. **Fisher diagnostics.** Measurements 3 and 4.
7. **Cross-weighted error.** Measurement 5.
8. **Findings.** Whether the lower-variance Fisher changed quantization quality, by how much relative to the noise floor, whether `token_k64` keeps it, and what that suggests for the default. Say plainly if the answer is "no measurable change".
9. **Caveats.** For example: one calibration set, a single-rerun noise estimate, thermal throttling, and bfloat16 gradients.

## Verification

The comparison module is tested offline on fixtures built by `tests/factories.py`, with no GPU runs.

- **Regression checks.** Two results that differ by 2% in one KL pass the KL check. At 4% it fails, naming the mode and bit-width.
- **Wave 0 compatibility.** A `results.json` whose `config` lacks `fisher` is accepted for its `reference` and `entries`.
- **Cross-weighted error.** On the tiny artifact, weighting by the artifact's own Fisher reproduces the `relative_error` in its `stats.json` to `1e-5`. A uniform Fisher gives the plain relative squared error.
- **Scale invariance.** Multiplying a Fisher by a constant changes neither the agreement nor the cross-weighted error.
- **Dequantization.** `dequantized_weight` equals the weight `set_precision` installs, at every bit-width of the tiny artifact.

The wave's experiment is done when the run matrix is complete, the gate passes, and `results.md` contains every section above.
