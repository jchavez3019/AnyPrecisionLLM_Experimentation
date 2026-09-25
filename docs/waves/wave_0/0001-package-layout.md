# Spec 0001: Package Layout and Module Boundaries

- Status: Proposed
- Wave: [0](index.md)
- Implements: [ADR 0002](../../adr/0002-project-layout-and-architecture.md) (repository layout)

This spec lists every file wave 0 creates, the one job each module has, and the import rules between them. Later specs define the contents; this one defines where things live.

## File tree

Every path below is new in wave 0, except `src/anyprec/__init__.py`, `src/anyprec/py.typed`, and `pyproject.toml`, which already exist. Directories that ADR 0002 reserves for later waves (`judge/`, `configs/judge/`) are not created.

```
AnyPrecisionLLM/
├── configs/
│   ├── quantize.yaml                    # spec 0002
│   ├── evaluate.yaml                    # spec 0002
│   ├── model/granite_4_0_350m.yaml
│   ├── calibration/c4.yaml
│   ├── quantizer/kmeans_iu.yaml
│   ├── rotation/{none,hadamard}.yaml
│   └── eval/default.yaml
├── quantization/
│   └── quantize_any_precision.py        # spec 0009
├── evaluation/
│   ├── evaluate_any_precision.py        # spec 0009
│   └── check_acceptance.py              # spec 0011: reads results and artifacts, prints pass/fail
├── src/anyprec/
│   ├── __init__.py                      # existing: __version__
│   ├── config/
│   │   ├── __init__.py
│   │   ├── schemas.py                   # pydantic models for every Hydra group
│   │   └── loading.py                   # DictConfig -> validated run config
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── hashing.py                   # canonical JSON, SHA-256 keys, stable per-name seeds
│   │   ├── seeding.py                   # seed_everything
│   │   ├── dtypes.py                    # "bfloat16" -> torch.bfloat16
│   │   ├── devices.py                   # resolve_device (no silent CPU fallback)
│   │   └── versions.py                  # library versions for manifests and results
│   ├── rotation/
│   │   ├── __init__.py
│   │   └── resolve.py                   # resolve_rotation (raises for hadamard)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── loading.py                   # CausalLM, load_model, load_tokenizer
│   │   ├── discovery.py                 # find_quantizable_linears
│   │   └── heads.py                     # causal_lm_loss, body_hidden_states, logit_head, check_sliced_logits
│   ├── data/
│   │   ├── __init__.py
│   │   ├── hub.py                       # load_texts: the only call to datasets.load_dataset
│   │   ├── calibration.py               # make_encoder, sample_calibration
│   │   └── evaluation_text.py           # load_eval_tokens, iter_chunks
│   ├── sensitivity/
│   │   ├── __init__.py
│   │   └── fisher.py                    # estimate_fisher
│   ├── quantization/
│   │   ├── __init__.py
│   │   ├── rows.py                      # PreparedRows, prepare_rows, segment_stats
│   │   ├── init.py                      # weighted_kmeanspp_init
│   │   ├── lloyd.py                     # weighted_lloyd
│   │   ├── split.py                     # split_all_segments, segment_ids
│   │   ├── layer.py                     # quantize_layer, LayerQuantization
│   │   ├── model.py                     # quantize_model (all target modules)
│   │   └── pipeline.py                  # run_quantization (entry-script body)
│   ├── artifacts/
│   │   ├── __init__.py
│   │   ├── keys.py                      # fisher_key, quantized_key
│   │   ├── manifest.py                  # FisherManifest, QuantizedManifest, ArtifactStats
│   │   └── store.py                     # ArtifactStore, QuantizedArtifact, atomic writes
│   ├── inference/
│   │   ├── __init__.py
│   │   └── precision.py                 # set_precision, snapshot_weights, restore_weights
│   └── evaluation/
│       ├── __init__.py
│       ├── bits.py                      # bits-per-weight accounting
│       ├── metrics.py                   # streaming KL, agreement, NLL
│       ├── results.py                   # Results schema
│       └── pipeline.py                  # run_evaluation (entry-script body)
└── tests/                               # spec 0010
```

## Layering

Modules import only from their own layer or the layers below them. This keeps the numerical kernels testable without a model, and it keeps Hydra out of the library.

```
 layer 5   quantization.pipeline, evaluation.pipeline      (orchestration; only these import artifacts + models + data together)
 layer 4   evaluation.{metrics, bits, results}, inference
 layer 3   sensitivity, quantization.{layer, model}, artifacts
 layer 2   models, data, rotation, quantization.{rows, init, lloyd, split}
 layer 1   config
 layer 0   utils
```

*Each layer may import from any layer with a smaller number, never from a larger one. `anyprec/__init__.py` holds only `__version__` and imports nothing from the package.*

Five rules apply on top of the layering:

- **Library modules import from the defining module, never from a subpackage `__init__`.** For example, `artifacts/store.py` writes `from anyprec.quantization.model import ModelQuantization`, not `from anyprec.quantization import ModelQuantization`. The subpackage `__init__` files re-export their pipelines (layer 5), so importing through them from a lower layer would create an import cycle. The re-exports exist for entry scripts, tests, and notebooks.

- **Hydra stops at the entry scripts.** Only `quantization/quantize_any_precision.py` and `evaluation/evaluate_any_precision.py` import `hydra` or `omegaconf`, plus `anyprec.config.loading`, which receives a `DictConfig`. Everything else receives frozen pydantic objects.
- **No module reads the environment or the working directory.** Paths come from `OutputConfig` (spec 0002).
- **Kernels are device-agnostic.** `quantization.{rows, init, lloyd, split, layer}` never call `.cuda()`. They work on whatever device their inputs are on.
- **Heavy dependencies are loaded at the boundary.** `transformers` and `datasets` are imported only in `models/`, `data/`, and the two `pipeline.py` modules. The kernels import only `torch`. Code outside `models/` that runs a model, such as `sensitivity/`, names it as `anyprec.models.loading.CausalLM` and calls helpers from `models/heads.py`, so untyped model outputs are narrowed in one place.

## Public API

Each subpackage's `__init__.py` re-exports exactly the names below and defines `__all__`. Everything else is private to its module, and names with a leading underscore are never imported across subpackages.

| Subpackage | Public names | Spec |
| --- | --- | --- |
| `anyprec.config` | `QuantizeRunConfig`, `EvaluateRunConfig`, `ModelConfig`, `QuantizableModules`, `CalibrationConfig`, `QuantizerConfig`, `RotationConfig`, `RotationNone`, `RotationHadamard`, `EvalConfig`, `EvalDatasetConfig`, `KLConfig`, `OutputConfig`, `FrozenModel`, `load_quantize_config`, `load_evaluate_config` | 0002 |
| `anyprec.utils` | `JsonValue`, `DTypeName`, `canonical_json`, `sha256_key`, `stable_seed`, `seed_everything`, `torch_dtype`, `resolve_device`, `library_versions` | 0002, 0009 |
| `anyprec.rotation` | `resolve_rotation` | 0002 |
| `anyprec.models` | `CausalLM`, `load_model`, `load_tokenizer`, `find_quantizable_linears`, `QuantizableModuleError`, `causal_lm_loss`, `body_hidden_states`, `logit_head`, `check_sliced_logits`, `SlicedLogitsError` | 0003 |
| `anyprec.data` | `load_texts`, `Encoder`, `make_encoder`, `sample_calibration`, `CalibrationError`, `load_eval_tokens`, `iter_chunks` | 0003, 0009 |
| `anyprec.sensitivity` | `estimate_fisher`, `FisherResult` | 0004 |
| `anyprec.quantization` | `PreparedRows`, `prepare_rows`, `segment_stats`, `weighted_kmeanspp_init`, `weighted_lloyd`, `LloydResult`, `split_all_segments`, `segment_ids`, `LayerQuantization`, `quantize_layer`, `ModelQuantization`, `quantize_model`, `run_quantization` | 0005, 0009 |
| `anyprec.artifacts` | `FISHER_SCHEMA_VERSION`, `QUANTIZED_SCHEMA_VERSION`, `fisher_snapshot`, `quantized_snapshot`, `fisher_key`, `quantized_key`, `ModuleEntry`, `FisherManifest`, `QuantizedManifest`, `ArtifactStats`, `ArtifactStore`, `QuantizedArtifact`, `FisherMeta`, `QuantizedMeta`, `ArtifactNotFoundError`, `ArtifactMismatchError` | 0002, 0006 |
| `anyprec.inference` | `set_precision`, `snapshot_weights`, `restore_weights`, `PrecisionError` | 0007 |
| `anyprec.evaluation` | `layer_bits_per_weight`, `parent_bits_per_weight`, `bits_report`, `BitsReport`, `ChunkOutputs`, `chunk_metrics`, `StreamingMetrics`, `MetricSummary`, `Results`, `run_evaluation` | 0008, 0009 |

## Verification

The layout itself is checked mechanically, so the rules above do not rely on review alone.

- `tests/test_layering.py` (spec 0010) parses every module under `src/anyprec` with `ast`. It fails if a module imports from a higher layer, imports through a subpackage `__init__`, or, other than `config/loading.py`, imports `hydra` or `omegaconf`.
- `pyright` strict and `ruff` run over `src`, `quantization`, `evaluation`, and `tests`, as configured in `pyproject.toml`.
