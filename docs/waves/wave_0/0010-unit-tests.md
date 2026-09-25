# Spec 0010: Unit Tests

- Status: Proposed
- Wave: [0](index.md)
- Implements: [ADR 0001](../../adr/0001-code-maintainability.md) (verification loop), [ADR 0002](../../adr/0002-project-layout-and-architecture.md) (pydantic-built fixtures)

This spec defines the offline test suite: the test tree, the shared fixtures, the conventions every test follows, and the property-based tests for the k-means kernels. Every test here runs on the CPU, without network access, in the default `pytest` invocation. GPU and network tests are in spec 0011.

## Conventions

These rules apply to every test file. They make a failing test explain itself, and they keep the suite deterministic.

- **Names state the scenario and the outcome.** For example, `test_split_all_segments_matches_brute_force_optimum` or `test_load_quantized_raises_mismatch_when_snapshot_differs`.
- **Every test has a Given / When / Then docstring.**

  ```python
  def test_set_precision_leaves_unquantized_parameters_untouched(tiny_artifact: TinyArtifact) -> None:
      """
      Given: the tiny Granite model and an incremental artifact covering its 12 linears.
      When: set_precision is called at 3 bits.
      Then: the embedding, norms, and every non-target parameter are bitwise unchanged.
      """
  ```

- **No shared mutable state.** Fixtures that build models or artifacts are function-scoped, or session-scoped and returned as read-only copies (`copy.deepcopy`, or `clone()` for tensors). Tests never write outside `tmp_path`.
- **Determinism.** Every random tensor comes from a local `torch.Generator` seeded in the test. Hypothesis runs with `derandomize=True` in a `ci` profile, which is registered in `tests/conftest.py` and selected by default.
- **No pass-through tests.** A test must check a transformation, an invariant, or an error path, never "a mock returns X, so the wrapper returns X".
- **Fully typed.** Test code is checked by `pyright` strict, like the library (`pyproject.toml` already includes `tests`).

## Test tree

The tree mirrors `src/anyprec`, so each spec's verification list maps onto one directory.

```
tests/
├── conftest.py                      # shared fixtures, hypothesis profile
├── strategies.py                    # hypothesis strategies for rows and codebooks
├── test_layering.py                 # spec 0001
├── config/
│   ├── test_schemas.py              # spec 0002
│   └── test_hydra_compose.py        # spec 0002: every YAML composes and validates
├── utils/
│   └── test_utils.py                # hashing, stable_seed, dtypes, resolve_device
├── rotation/
│   └── test_resolve.py
├── models/
│   ├── test_discovery.py            # spec 0003
│   └── test_heads.py                # spec 0003: sliced head equals forward()
├── data/
│   ├── test_calibration.py
│   └── test_evaluation_text.py
├── sensitivity/
│   └── test_fisher.py               # spec 0004
├── quantization/
│   ├── test_rows.py                 # spec 0005
│   ├── test_init.py
│   ├── test_lloyd.py
│   ├── test_split.py
│   ├── test_layer.py
│   ├── test_model.py
│   └── test_pipeline.py             # spec 0009
├── artifacts/
│   ├── test_keys.py                 # spec 0002
│   ├── test_manifest.py             # spec 0006
│   └── test_store.py
├── inference/
│   └── test_precision.py            # spec 0007
├── evaluation/
│   ├── test_metrics.py              # spec 0008
│   ├── test_bits.py
│   ├── test_results.py
│   └── test_pipeline.py             # spec 0009
└── integration/                     # spec 0011; every test marked gpu, network, or slow
```

## Shared fixtures

Fixtures build real objects of the library's own types. Configs are pydantic instances, and the model is a genuine `GraniteMoeHybridForCausalLM`, so no fixture can drift away from the schemas it stands in for.

| Fixture | Scope | Returns |
| --- | --- | --- |
| `tiny_granite_config` | session | The verified `GraniteMoeHybridConfig` below |
| `tiny_model` | function | A fresh float32 model from `tiny_granite_config`, weights initialized under `torch.manual_seed(0)`, in `eval()` mode |
| `tiny_model_config` | session | `ModelConfig` whose `quantizable_modules` matches the tiny model, with `expected_count=12` |
| `char_encoder` | session | `Encoder` mapping each byte of the UTF-8 text to one token id in `[0, 256)` |
| `quantizer_config` | function, parametrized | `QuantizerConfig(mode=..., seed_bits=2, parent_bits=4, seed=0, lloyd_max_iter=50, empty_eps=1e-12, row_chunk=16)` |
| `quantize_run_config` | function | `QuantizeRunConfig` with `device="cpu"`, `output.cache_dir = tmp_path / "cache"` |
| `evaluate_run_config` | function | `EvaluateRunConfig` with `chunk_len=32`, `max_chunks=2`, `bits=[2, 3, 4]` |
| `tiny_artifact` | function | A `QuantizedArtifact` from `quantize_model` on `tiny_model` with a random positive Fisher |
| `offline_loaders` | function | Monkeypatches `load_model`, `load_tokenizer`/`make_encoder`, and `load_texts` in both pipeline modules |

```python
TINY_GRANITE = GraniteMoeHybridConfig(
    num_hidden_layers=2, hidden_size=64, intermediate_size=128, shared_intermediate_size=128,
    num_attention_heads=4, num_key_value_heads=2, vocab_size=256, num_local_experts=0,
    layer_types=["attention", "attention"], max_position_embeddings=128, tie_word_embeddings=True,
)
```

This configuration was verified in the development environment. It has 12 quantizable linears; `input_linear` is `[256, 64]`, `output_linear` is `[64, 128]`, `q_proj` and `o_proj` are `[64, 64]`, and `k_proj` and `v_proj` are `[32, 64]`. The embedding and LM head are tied, the model has 90,432 parameters, and one forward plus backward pass takes about 0.05 s on the CPU.

Using `seed_bits=2, parent_bits=4` keeps $2^{b} \le n$ for every tiny module, so every bit-width has more weights per row than centroids.

## Property-based kernel tests

The k-means kernels have exact mathematical properties (ADR 0003, Section 4), and hypothesis is the cheapest way to check them over many row shapes. The strategies live in `tests/strategies.py`.

```python
@st.composite
def weighted_rows(draw, max_rows: int = 4, max_n: int = 48) -> tuple[Tensor, Tensor]:
    """Draw (weight [R, n], fisher [R, n]) float32 tensors, including ties, zeros, and all-zero Fisher rows."""
```

The strategy mixes in the hard cases on purpose: repeated weight values, exact zeros, a single dominant sensitivity (like the notebook's row 58, column 148), and rows whose sensitivities sum to zero.

| Property | Test | Brute-force oracle |
| --- | --- | --- |
| Prefix sums equal cumulative sums | `test_rows.py` | Python loop in float64 |
| Segment mass, mean, and cost | `test_rows.py` | Direct sums over each segment |
| Lloyd never increases $J$ over its initialization | `test_lloyd.py` | $J$ computed from the returned borders |
| Lloyd is a fixed point at convergence | `test_lloyd.py` | One extra assign and update step changes nothing |
| Lloyd is no better than the optimum | `test_lloyd.py` | Enumerate every contiguous partition for $n \le 10$, $K \le 3$ |
| Each split is the best 2-way split | `test_split.py` | Try every split point of every segment |
| Splits never increase $J$ | `test_split.py` | $J$ before and after |
| Stored indices are nested | `test_layer.py` | `idx_B >> (B - b)` equals the segment id at $b$ |
| The incremental seed equals the standalone fit at $b_0$ | `test_layer.py` | Bitwise equality of LUTs and indices |

Hypothesis settings: `max_examples=200` for the pure kernels, and 25 for `quantize_layer`. Floating-point comparisons use `torch.testing.assert_close` in float64 with `rtol=1e-9`, except where float16 LUT rounding is involved, which uses the float16 tolerance.

## Layering test

`test_layering.py` enforces the rules of spec 0001 mechanically, so they do not depend on review alone.

```python
LAYERS: dict[str, int] = {
    "anyprec.utils": 0, "anyprec.config": 1,
    "anyprec.models": 2, "anyprec.data": 2, "anyprec.rotation": 2,
    "anyprec.quantization.rows": 2, "anyprec.quantization.init": 2,
    "anyprec.quantization.lloyd": 2, "anyprec.quantization.split": 2,
    "anyprec.sensitivity": 3, "anyprec.quantization.layer": 3, "anyprec.quantization.model": 3,
    "anyprec.artifacts": 3,
    "anyprec.evaluation.metrics": 4, "anyprec.evaluation.bits": 4, "anyprec.evaluation.results": 4,
    "anyprec.inference": 4,
    "anyprec.quantization.pipeline": 5, "anyprec.evaluation.pipeline": 5,
}
```

For every module file, the test parses the imports with `ast`, maps each `anyprec` import to its longest matching prefix in `LAYERS`, and fails on any import from a higher layer. A module missing from `LAYERS` fails the test as well, so a new module cannot skip classification. The test also fails on:

- an import of `hydra` or `omegaconf` outside `anyprec.config.loading`;
- an import of `transformers` or `datasets` outside `models`, `data`, and the two pipelines;
- an import that names a package rather than a module (for example `from anyprec.quantization import ...`), since that runs the subpackage `__init__` and its layer-5 re-exports (spec 0001).

`__init__.py` files are exempt from the layer check, but they may import only from modules inside their own subpackage. `anyprec/__init__.py` must import nothing from the package.

## Coverage and runtime

The suite gates on coverage and stays fast enough to run after every change, which is how ADR 0001's verification loop is meant to be used.

- `pytest --cov=anyprec --cov-fail-under=70` passes. The kernel, artifact, inference, and metric modules are expected to exceed 90%; the pipelines lower the average.
- The default offline suite finishes in under 60 s on the laptop. A test that needs longer gets the `slow` marker and moves to spec 0011.
- `pyproject.toml` already deselects `gpu`, `slow`, and `network` by default; nothing needs to change there. `pytest-cov` and `hypothesis` are already in `environment.yaml`.

## Per-spec test inventory

Each spec lists its verification cases in its own final section, and this table maps them to files. A spec's tests land in the same implementation step as its code (see the [wave index](index.md)).

| Spec | Test files |
| --- | --- |
| [0001](0001-package-layout.md) | `test_layering.py` |
| [0002](0002-configuration-and-cache-keys.md) | `config/*`, `utils/test_utils.py`, `rotation/test_resolve.py`, `artifacts/test_keys.py` |
| [0003](0003-model-and-data-loading.md) | `models/test_discovery.py`, `models/test_heads.py`, `data/*` |
| [0004](0004-fisher-estimation.md) | `sensitivity/test_fisher.py` |
| [0005](0005-kmeans-and-upscaling.md) | `quantization/test_{rows,init,lloyd,split,layer,model}.py` |
| [0006](0006-artifact-store.md) | `artifacts/test_manifest.py`, `artifacts/test_store.py` |
| [0007](0007-simulated-inference.md) | `inference/test_precision.py` |
| [0008](0008-evaluation-metrics.md) | `evaluation/test_{metrics,bits,results}.py` |
| [0009](0009-entry-scripts-and-pipelines.md) | `quantization/test_pipeline.py`, `evaluation/test_pipeline.py` |
