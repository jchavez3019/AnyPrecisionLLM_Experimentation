# Spec 0015: Pipeline Wiring and Integration Tests

- Status: Proposed
- Wave: [1](index.md)
- Implements: [ADR 0008](../../adr/0008-batched-and-per-token-fisher.md) (Testing), [ADR 0001](../../adr/0001-code-maintainability.md) (verification loop)

This spec connects the configurable Fisher to both pipelines, records peak GPU memory per quantize stage, and adds the GPU tests on Granite that pin ADR 0008's equivalences at full scale. After it, one Hydra override switches the estimator end to end.

## Files

Only orchestration changes here. The estimators are specs 0013 and 0014, and the keys are spec 0012.

| File | Change |
| --- | --- |
| `src/anyprec/quantization/pipeline.py` | Pass `cfg.fisher` and `cfg.calibration.seed` to `estimate_fisher`; measure peak memory per stage; extend the summary |
| `src/anyprec/evaluation/pipeline.py` | Compute the Fisher key with `cfg.fisher` |
| `src/anyprec/utils/devices.py` | `peak_gpu_stage`, a context manager for measuring a stage's peak |
| `tests/quantization/test_pipeline.py`, `tests/evaluation/test_pipeline.py` | The offline tests below |
| `tests/integration/test_token_fisher_granite.py` | New GPU tests |
| `tests/integration/test_fisher_granite.py` | Extended to the batched sequence path |

## Interface

Peak memory is measured by one helper, used for both quantize stages:

```python
@dataclass
class StagePeak:
    """Peak allocated GPU memory over one stage; ``peak_bytes`` is ``None`` on the CPU."""

    peak_bytes: int | None = None


@contextmanager
def peak_gpu_stage(device: torch.device) -> Iterator[StagePeak]:
    """Reset CUDA's peak statistics on entry, and fill ``peak_bytes`` with ``max_memory_allocated``
    on exit. The figure includes everything already resident, such as the model weights.
    """
```

**`run_quantization`** changes in three places. Everything else, including the reuse paths, stays as in spec 0009.

1. The keys use `fisher_snapshot(cfg.model, cfg.calibration, cfg.fisher, cfg.rotation)`.
2. The Fisher phase runs inside `peak_gpu_stage(device) as fisher_peak`, and calls `estimate_fisher(model, calibration, targets, cfg.fisher, cfg.calibration.seed, progress)`. Then `FisherMeta.from_config(cfg, device, fisher_peak.peak_bytes)` is saved with the result.
3. The k-means phase runs inside `peak_gpu_stage(device) as kmeans_peak`, after the existing `empty_cache`.

**`quantize_summary.json`** gains four fields. Nothing reads it back (spec 0009), so no schema version is involved.

| Field | Value |
| --- | --- |
| `fisher_estimator` | `FisherEstimatorRecord.from_config(cfg.fisher)`, as JSON |
| `fisher_seconds` | The Fisher result's `seconds`, or `null` when the Fisher was reused |
| `fisher_peak_gpu_gib` | `fisher_peak.peak_bytes / 2**30`, or `null` when reused or on the CPU |
| `kmeans_peak_gpu_gib` | `kmeans_peak.peak_bytes / 2**30`, or `null` on the CPU |

**`run_evaluation`** computes `fisher_key(cfg.model, cfg.calibration, cfg.fisher, cfg.rotation)`. An evaluation run must use the same `fisher.granularity` and `fisher.positions_per_sequence` as the quantize run. Otherwise it looks for artifacts that do not exist, and fails with the existing missing-artifact error, which names the key.

## Offline tests

These run on the tiny model through the existing `compose_quantize` fixtures, without a GPU.

- **The override reaches the estimator.** `fisher.granularity=token` gives a Fisher manifest whose `estimator.granularity` is `token`, and a Fisher key different from the default run's.
- **Reuse ignores non-key fields.** A second run that differs only in `fisher.batch_size` reuses the Fisher and the quantized artifact.
- **Summary on the CPU.** `quantize_summary.json` has `fisher_estimator` and `fisher_seconds`, and `null` for both peak fields.
- **Split halves are saved.** `fisher.split_half=true` gives a manifest with one `split_half` entry per module, each in $[-1, 1]$.
- **Evaluation finds the token artifacts.** An evaluation with the same `fisher` overrides as the quantize run finds its artifacts. One with the default `fisher` group raises the missing-artifact error.

## GPU integration tests

These repeat ADR 0008's probe on Granite, at full size where memory allows. They are marked `gpu` and `network`, and each runs in about a minute of GPU time. Every test compares two Fisher diagonals accumulated over the same inputs. The error is the largest absolute difference divided by the reference's largest entry, which avoids dividing by position 0's zero gradient.

Squaring roughly doubles a gradient's relative error, so each Fisher tolerance is twice the gradient tolerance ADR 0008 sets for the capability check. The two dtypes play different roles:

- **Float32 is the correctness pin.** The model is loaded with `model.dtype=float32`, which takes 1.4 GB for the weights and about twice bfloat16's graph. To fit, these tests use two 256-token C4 sequences and the 18 linears of layers 0, 13 and 27, passed as `targets`. The tolerance is `2e-4`.
- **Bfloat16 is the smoke and memory test.** It uses the full 168 linears and 512-token sequences, as the real runs do. The tolerance is `2e-1`. That covers ADR 0008's measured 1.7% for batched against looped gradients, and 5.4–5.8% for prefix against full-sequence gradients, each doubled by the square.

| Test | Compares | Positions |
| --- | --- | --- |
| `test_batched_sequence_fisher_on_granite_matches_the_unbatched_loop` | Sequence mode, `batch_size: 2` against `batch_size: 1` | all, over 4 sequences in bfloat16 and 2 in float32 |
| `test_batched_backward_matches_looped_autograd_on_granite` | `batched_backward` against one `torch.autograd.grad` per position with respect to the target weights | 16 per sequence, including 0 and $T - 2$, drawn by `sample_positions` |
| `test_prefix_batch_matches_batched_backward_on_granite` | `prefix_batch` against `batched_backward` | the same 16 |
| `test_capability_check_passes_on_granite` | `check_batched_backward` raises nothing | its own three, bfloat16 only |

The bfloat16 sequence-batching test also asserts that the Fisher stage's peak memory, from `peak_gpu_stage`, is below 5.5 GiB. The memory of the token runs is measured by spec 0016's timing probe instead, at the batch size those runs will use.

## Verification

This spec is complete when the full ADR 0001 loop passes:

- ruff, strict pyright, and the offline pytest suite with coverage of at least 70%;
- `pytest -m "gpu or network or slow"` on the laptop GPU, including the four tests above and wave 0's existing integration tests with the new keys.
