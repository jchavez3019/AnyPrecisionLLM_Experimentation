# Spec 0011: Integration and Acceptance

- Status: Proposed
- Wave: [0](index.md)
- Implements: the last three gates of the wave's definition of done

This spec defines the tests that touch the real Granite checkpoint, the GPU, or the Hugging Face Hub, and the full quantize-and-evaluate run that closes the wave. Hard criteria must hold for the wave to be accepted. Observations are recorded for the next wave's planning, but they do not gate this one.

## Integration tests

These tests exercise the real model and datasets, at a size that finishes in minutes. They live in `tests/integration/`, and they only run when selected explicitly, because the default `addopts` in `pyproject.toml` deselects all three markers.

| Test | Markers | What it checks |
| --- | --- | --- |
| `test_discovery_granite.py` | `network` | `load_model` on the CPU in bfloat16: exactly 168 modules, in layer order, with the Granite shapes of ADR 0002, and parameter count 352,379,904 |
| `test_calibration_c4.py` | `network` | `load_texts`, `load_tokenizer`, and eight sequences from the real C4 shard: shape `[8, 512]`, reproducible across two calls, and all token ids below the vocabulary size |
| `test_fisher_granite.py` | `gpu`, `network` | `estimate_fisher` on 8 sequences: the model's flags are restored, every diagonal is finite and non-negative, and peak memory stays below 3.5 GiB |
| `test_quantize_granite.py` | `gpu`, `network`, `slow` | `run_quantization` with 8 sequences, in both modes, into a `tmp_path` cache; the second run reuses both artifacts |
| `test_evaluate_granite.py` | `gpu`, `network`, `slow` | `run_evaluation` on those artifacts with `max_chunks=2`: the same-weights check passes and `results.json` validates |
| `test_cpu_gpu_kernels.py` | `gpu` | `quantize_layer` on one real-sized random `[1024, 1024]` matrix, run on the CPU and on the GPU: identical shapes, dtypes, and nesting; relative errors within 2% at 3 bits and 10% at higher bit-widths |

Commands:

```bash
pytest -m network                  # CPU-only integration tests, downloads allowed
pytest -m "gpu or network or slow" # every integration test; the definition-of-done command
```

The GPU tests call `pytest.skip` with a clear reason when `torch.cuda.is_available()` is false, so `-m gpu` on a CPU machine reports skips, not failures. The cross-device test does not assert bitwise equality. k-means++ draws differ between CPU and CUDA generators (spec 0002), so the codebooks can differ; only the properties that must not depend on the device are checked. The error tolerance is wider above 3 bits for the same reason as the reference bands below: with few weights per cluster, the error depends more on the exact draws.

## The full run

The acceptance run uses the default configs with no overrides, apart from `quantizer.mode` for the standalone artifact. It is the run whose `results.json` becomes the project's first baseline.

```bash
python quantization/quantize_any_precision.py
python quantization/quantize_any_precision.py quantizer.mode=standalone
python evaluation/evaluate_any_precision.py
```

A short script, `evaluation/check_acceptance.py`, takes the path of `results.json`, reads it, and then loads the two manifests and both `stats.json` files through `ArtifactStore`, using the keys recorded in the results. It prints one line per criterion below with its pass or fail status, and exits nonzero if any hard criterion or reference band fails. The script only reads artifacts; it is a thin checker over the pydantic schemas of specs 0006 and 0008, and it stays reusable as a regression check in later waves.

## Hard criteria

Every row must pass. Each one follows from a property proven in ADR 0003, a measurement in notebook 02, or an invariant of this wave's code, so a failure means a bug, not bad luck.

| Criterion | Source |
| --- | --- |
| Both artifacts list 168 modules, in the same order, with the Granite shapes | Spec 0003 |
| The Fisher manifest's `mean_loss` is within 0.02 of 3.22 nats/token | Notebook 02, section 4.4; spec 0003 uses the same sampling rule |
| For every module in the incremental artifact, `stats.relative_error` is strictly decreasing from 3 to 8 bits | ADR 0003, Section 4, property 1 (strict in practice, since each split separates distinct values) |
| For every module, the incremental and standalone 3-bit relative errors are equal | ADR 0003, Section 4, property 4 |
| For every module and every $b \ge 4$, the incremental relative error is at least the standalone error, allowing a relative tolerance of 1% | ADR 0003, Section 4, property 4 (float16 rounding and Lloyd's local optimum make a small inversion possible) |
| The same-weights check reports mean KL at most $10^{-6}$ and agreement 1 | Spec 0008 |
| For both modes, mean KL on WikiText-2 decreases with every added bit | The KL divergence is the objective's target (ADR 0003, Section 1) |
| For both modes, top-1 agreement at 8 bits is above both its 3-bit value and 0.95 | Agreement is a coarse 0/1 statistic, so only the endpoints are gated; the threshold is conservative at 8-bit errors of about $10^{-5}$ |
| Quantized perplexity is at or above the reference on both datasets at 3 bits | A 3-bit model cannot beat its own reference by a measurable margin |
| `results.json` bits per weight equal spec 0008's table to four decimals | Spec 0008 |

## Reference bands

Notebook 02 measured whole-module relative errors for two modules. The package's values differ slightly, because it rounds LUTs to float16 and uses per-chunk k-means++ seeds. The bands allow for that; a value outside its band points to a real difference in the algorithm, and must be explained before the wave is accepted.

| Module | Quantity | Notebook 02 | Accepted band |
| --- | --- | --- | --- |
| `model.layers.14.self_attn.q_proj` | IU relative error, 3 bits | 6.63e-3 | ±10% |
| `model.layers.14.self_attn.q_proj` | IU relative error, 8 bits | 3.18e-6 | within a factor of 2 |
| `model.layers.14.self_attn.q_proj` | SA relative error, 8 bits | 9.18e-7 | within a factor of 2 |
| `model.layers.2.shared_mlp.output_linear` | IU relative error, 3 bits | 1.48e-2 | ±10% |
| `model.layers.2.shared_mlp.output_linear` | IU relative error, 8 bits | 1.16e-5 | within a factor of 2 |
| `model.layers.2.shared_mlp.output_linear` | SA relative error, 8 bits | 7.22e-6 | within a factor of 2 |

*The 3-bit bands are tighter because the seed is well conditioned: 8 centroids per 1,024 weights. At 8 bits, a single cluster holds about 4 weights, so errors are sensitive to the exact random draws.*

## Observations to record

These are the numbers the next wave will plan around. They are written into a short `docs/waves/wave_0/results.md` after the run, along with the run date, the library versions, and the `results.json` path. None of them gates acceptance.

- Wall-clock time of each stage: Fisher (the notebook measured 14 s), k-means for each mode (expected 3 to 5 minutes incremental), and evaluation.
- Peak GPU memory of each stage, from `torch.cuda.max_memory_allocated`.
- The KL, agreement, and perplexity curves for both modes, and the incremental-to-standalone gap at 4 and 5 bits. Notebook 02 predicts this is where nesting costs the most in answer quality.
- The distribution of the 8-bit incremental-to-standalone error ratio over all 168 modules. Notebook 02 saw 1.6 to 3.5 on two modules.
- The modules with the largest 3-bit relative error, and whether they coincide with the extreme-crest rows found in the first notebook.
- The WikiText-2 perplexity of the float32 reference, as the anchor for every later comparison.

## Exit

Wave 0 is accepted when the definition of done in the [wave index](index.md) holds: the offline suite, the integration tests, every hard criterion, and every reference band. `results.md` records the observations. After acceptance, the status of the wave index and of every spec changes from Proposed to Accepted.
