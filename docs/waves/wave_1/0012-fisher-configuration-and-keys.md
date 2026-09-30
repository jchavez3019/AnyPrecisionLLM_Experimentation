# Spec 0012: Fisher Configuration, Keys and Manifest

- Status: Proposed
- Wave: [1](index.md)
- Implements: [ADR 0008](../../adr/0008-batched-and-per-token-fisher.md) (Configuration, Stored scale and diagnostics), [ADR 0005](../../adr/0005-artifact-format-and-simulated-inference.md) (manifest fields)

This spec adds the `fisher` config group, puts the estimator's key fields into the Fisher cache key, and extends the Fisher manifest with the estimator, split-half correlations and peak GPU memory. It changes no numerics: after this spec the default run still computes wave 0's Fisher, under a new key.

## Files

Every file below exists except the new YAML group. The callers of `fisher_key` and `fisher_snapshot` are listed in full, because the signature change breaks each one.

| File | Change |
| --- | --- |
| `configs/fisher/default.yaml` | New group, the five fields below |
| `configs/quantize.yaml`, `configs/evaluate.yaml` | `fisher: default` added to the defaults list, after `calibration` |
| `src/anyprec/config/schemas.py` | `FisherGranularity`, `TokenMethod`, `FisherConfig`; `fisher` field and validation on both run configs |
| `src/anyprec/artifacts/keys.py` | `FISHER_SCHEMA_VERSION = 2`; `fisher` argument on `fisher_snapshot` and `fisher_key` |
| `src/anyprec/artifacts/manifest.py` | `FisherEstimatorRecord`, `SplitHalfEntry`; three new `FisherManifest` fields |
| `src/anyprec/artifacts/store.py` | `FisherMeta` gains the estimator and peak memory; `QuantizedMeta.from_config` passes `cfg.fisher` |
| `src/anyprec/evaluation/results.py` | `RESULTS_SCHEMA_VERSION = 2`, because `EvaluateRunConfig` gains a field |
| `src/anyprec/quantization/pipeline.py`, `src/anyprec/evaluation/pipeline.py` | Pass `cfg.fisher` to the key functions (spec 0015 does the rest) |
| `tests/factories.py`, `tests/artifacts/test_keys.py`, `tests/artifacts/test_store.py`, `tests/quantization/test_pipeline.py`, `tests/config/` | Updated callers and the tests below |

## Interface

The YAML group mirrors ADR 0008's block exactly:

```yaml
# configs/fisher/default.yaml
granularity: sequence
token_method: prefix_batch
positions_per_sequence: null
batch_size: 1
split_half: false
```

The pydantic model follows the existing `FrozenModel` conventions:

```python
FisherGranularity = Literal["sequence", "token"]
TokenMethod = Literal["prefix_batch", "batched_backward"]


class FisherConfig(FrozenModel):
    """Fisher estimator settings (ADR 0008, Configuration)."""

    granularity: FisherGranularity
    token_method: TokenMethod
    positions_per_sequence: PositiveInt | None
    batch_size: PositiveInt
    split_half: bool

    @model_validator(mode="after")
    def _positions_only_in_token_mode(self) -> Self:
        """Reject a position subsample in sequence mode, where it has no meaning."""
```

`QuantizeRunConfig` and `EvaluateRunConfig` each gain `fisher: FisherConfig`, with no default, because Hydra always composes the group. Both run configs call one shared helper from their `after` validators:

```python
def check_fisher_fits_calibration(fisher: FisherConfig, calibration: CalibrationConfig) -> None:
    """Require ``positions_per_sequence`` to be at most ``calibration.seq_len - 1``.

    :raises ValueError: If the subsample is larger than the positions available.
    """
```

**Keys.** `fisher_snapshot` gains a `fisher` block that holds only the two fields that change the estimate:

```python
FISHER_SCHEMA_VERSION: int = 2


def fisher_snapshot(
    model: ModelConfig,
    calibration: CalibrationConfig,
    fisher: FisherConfig,
    rotation: RotationConfig,
) -> dict[str, JsonValue]:
    """Return the Fisher key's snapshot (ADR 0002, caching table; ADR 0008, Cache key).

    The ``fisher`` block holds ``granularity`` and ``positions_per_sequence``. ``token_method``,
    ``batch_size`` and ``split_half`` are left out, because they do not change the estimate.
    """


def fisher_key(
    model: ModelConfig,
    calibration: CalibrationConfig,
    fisher: FisherConfig,
    rotation: RotationConfig,
) -> str:
    """Return ``sha256_key(fisher_snapshot(...))``."""
```

`quantized_snapshot` and `QUANTIZED_SCHEMA_VERSION` are unchanged. Every quantized key changes anyway, because it embeds the Fisher key.

**Manifest.** Three fields are added to `FisherManifest`. They are pydantic models in `artifacts/manifest.py`, so `artifacts` never imports `sensitivity`:

```python
class FisherEstimatorRecord(FrozenModel):
    """How the Fisher was computed, including the fields that are not in the key."""

    granularity: FisherGranularity
    token_method: TokenMethod | None
    positions_per_sequence: PositiveInt | None
    batch_size: PositiveInt
    split_half: bool

    @classmethod
    def from_config(cls, fisher: FisherConfig) -> Self:
        """Record the estimator; ``token_method`` is ``None`` in sequence mode, where it is ignored."""


class SplitHalfEntry(FrozenModel):
    """Split-half stability of one module's diagonal (ADR 0008, Diagnostics)."""

    spearman_half: float = Field(ge=-1.0, le=1.0)
    spearman_full: float | None = Field(ge=0.0, le=1.0)


class FisherManifest(ManifestBase):
    # Existing: kind, num_sequences, seq_len, mean_loss, seconds.
    estimator: FisherEstimatorRecord
    split_half: dict[str, SplitHalfEntry] | None
    peak_gpu_bytes: NonNegativeInt | None
```

- `split_half` is `None` unless `fisher.split_half` was set. When set, its keys equal `modules` in order, and a validator enforces that.
- `peak_gpu_bytes` is `torch.cuda.max_memory_allocated()` over the Fisher stage alone, and `None` on the CPU.
- `mean_loss` keeps its name. In token mode it is the mean NLL over the evaluated positions (ADR 0008, Stored scale and diagnostics).

`FisherMeta` gains `estimator: FisherEstimatorRecord` and `peak_gpu_bytes: int | None`. `FisherMeta.from_config(cfg, device, peak_gpu_bytes)` fills both, and `save_fisher` copies them into the manifest along with the split-half entries from the result (spec 0013).

## Compatibility

The schema version and the new `fisher` block both enter the Fisher snapshot, so every Fisher key changes, and every quantized key with it. Wave 1's first run therefore finds no cached artifact and recomputes everything; wave 0's directories are never looked up. They stay on disk under their old keys. Spec 0016 reads their `stats.json` and wave 0's `results.json`, so the cache directory must not be cleared during this wave.

`Results` from wave 0 no longer validate, because their `config` lacks `fisher`. `check_acceptance.py` only reads new results, so it is unaffected; spec 0016 reads wave 0's metrics field by field instead.

## Verification

The unit tests are offline, and they build every config through `tests/factories.py`.

- **Key membership.** Changing `granularity` or `positions_per_sequence` changes `fisher_key`. Changing `token_method`, `batch_size` or `split_half` does not. One parametrized test covers all five fields.
- **Quantized keys follow.** Two configs that differ only in `fisher.granularity` give different `quantized_key`s, with identical quantizer settings.
- **Validation.** `positions_per_sequence` is rejected in sequence mode, and when it exceeds `seq_len - 1` in either run config. A `positions_per_sequence` of exactly `seq_len - 1` is accepted.
- **Manifest round trip.** A manifest with split-half entries saves and reloads. A manifest whose `split_half` keys differ from `modules` is rejected.
- **Estimator record.** `from_config` records `token_method=None` in sequence mode and the configured method in token mode.
- **Snapshot contents.** The default config's snapshot has `schema_version: 2` and a `fisher` block with exactly the keys `granularity` and `positions_per_sequence`.
- **Hydra composition.** `load_quantize_config` and `load_evaluate_config` compose the default group and accept `fisher.granularity=token` overrides.
