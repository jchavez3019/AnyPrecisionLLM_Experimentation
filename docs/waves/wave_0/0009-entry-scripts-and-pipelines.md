# Spec 0009: Entry Scripts and Pipelines

- Status: Proposed
- Wave: [0](index.md)
- Implements: [ADR 0002](../../adr/0002-project-layout-and-architecture.md) (entry scripts, Hydra, caching), and the orchestration that ties specs 0002–0008 together

This spec defines the two library functions that run a whole stage, `run_quantization` and `run_evaluation`, and the two thin Hydra scripts that call them. The pipelines are the only modules that combine models, data, and artifacts (spec 0001, layer 5).

## Files

The entry scripts do three things: compose the config, validate it, and call a pipeline. Everything testable lives in the pipelines.

| File | Contents |
| --- | --- |
| `quantization/quantize_any_precision.py` | Hydra `main` for quantization |
| `evaluation/evaluate_any_precision.py` | Hydra `main` for evaluation |
| `src/anyprec/quantization/pipeline.py` | `run_quantization`, `QuantizationOutcome` |
| `src/anyprec/evaluation/pipeline.py` | `run_evaluation` |
| `src/anyprec/data/hub.py` | `load_texts` |
| `src/anyprec/utils/devices.py` | `resolve_device` |

## Shared boundary helpers

These small helpers keep third-party types at the edge, so the functions in specs 0003–0008 only ever see plain Python and `torch` types. `load_texts` is the only call to `datasets.load_dataset` in the package. It lives in `data/` (layer 2), so both pipelines import it without importing each other.

```python
def load_texts(                                          # anyprec/data/hub.pypath: str, name: str | None, data_files: dict[str, str] | None, split: str, text_field: str) -> list[str]:
    """Load one split with `datasets` and return its text column as a list of strings."""
    dataset = load_dataset(path, name=name, data_files=data_files, split=split)
    if not isinstance(dataset, Dataset):
        raise TypeError(f"expected a Dataset for {path}:{split}, got {type(dataset).__name__}")
    column = dataset[text_field]
    if not isinstance(column, list) or not all(isinstance(t, str) for t in column):
        raise TypeError(f"column {text_field!r} of {path} is not a list of strings")
    return column

def resolve_device(name: str) -> torch.device:          # anyprec/utils/devices.py
    """Parse cfg.device; a CUDA request without CUDA is an error, never a silent CPU fallback."""
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"device={name!r} requested but CUDA is not available; pass device=cpu explicitly")
    return device
```

`resolve_device` lives in `utils/` (layer 0), because both pipelines and the tests use it. `load_texts` is a pass-through wrapper, and like `load_model` it is covered only by the `network` integration tests (spec 0011).

## Quantization pipeline

`run_quantization` produces one quantized artifact for the configured `quantizer.mode`, reusing the cached Fisher if one exists. Running it again with `quantizer.mode=standalone` reuses the same Fisher and adds the standalone artifact.

```python
@dataclass(frozen=True)
class QuantizationOutcome:
    fisher_key: str
    quantized_key: str
    fisher_dir: Path
    quantized_dir: Path
    fisher_reused: bool
    quantized_reused: bool

def run_quantization(cfg: QuantizeRunConfig, run_dir: Path) -> QuantizationOutcome:
    # Fail fast on configuration the wave cannot run, before any download or GPU allocation.

    resolve_rotation(cfg.rotation)
    device = resolve_device(cfg.device)
    seed_everything(cfg.seed)
    store = ArtifactStore(cfg.output.cache_dir)

    # Keys depend only on configuration, so both are known before any model is loaded.

    f_snapshot = fisher_snapshot(cfg.model, cfg.calibration, cfg.rotation)
    f_key = sha256_key(f_snapshot)
    q_snapshot = quantized_snapshot(f_key, cfg.quantizer)
    q_key = sha256_key(q_snapshot)

    # The model is needed even when both artifacts exist, because discovery validates the module
    # list that every load checks against.

    model = load_model(cfg.model, torch_dtype(cfg.model.dtype), device)
    targets = find_quantizable_linears(model, cfg.model.quantizable_modules)
    modules = [ModuleEntry(name=n, shape=tuple(l.weight.shape)) for n, l in targets.items()]

    if store.has_quantized(q_key):
        store.load_quantized(q_key, q_snapshot, modules)           # validates, then discards
        return _outcome(..., fisher_reused=store.has_fisher(f_key), quantized_reused=True)

    # Phase 1: the Fisher diagonal, from the cache or computed on the calibration set.

    fisher_reused = store.has_fisher(f_key)
    if fisher_reused:
        fisher = store.load_fisher(f_key, f_snapshot, modules)
    else:
        texts = load_texts(cfg.calibration.path, cfg.calibration.name, cfg.calibration.data_files,
                           cfg.calibration.split, cfg.calibration.text_field)
        calibration = sample_calibration(texts, make_encoder(load_tokenizer(cfg.model)), cfg.calibration)  # [N, T]
        result = estimate_fisher(model, calibration, targets, progress=_tqdm_counter("fisher"))
        store.save_fisher(f_key, f_snapshot, result, FisherMeta.from_config(cfg, device))
        fisher = result.diagonals
        del result, calibration

    # Phase 2: k-means over every module. Free the Fisher pass's cached blocks first, because
    # the k-means++ trial tensors are the next largest allocation.

    if device.type == "cuda":
        torch.cuda.empty_cache()
    weights = {name: linear.weight for name, linear in targets.items()}   # [m, n] each; rotated here once ADR 0006 lands
    quantization = quantize_model(weights, fisher, cfg.quantizer, device, progress=_tqdm_names(len(targets)))

    # Phase 3: persist atomically, and record a human-readable summary in the Hydra run directory.

    q_dir = store.save_quantized(q_key, q_snapshot, quantization, QuantizedMeta.from_config(cfg, device))
    _write_summary(run_dir / "quantize_summary.json", f_key, q_key, quantization)
    return _outcome(..., fisher_reused=fisher_reused, quantized_reused=False)
```

`quantize_summary.json` lists both keys, the directories, the timings, and the per-bit median relative error across modules. It is a convenience for people reading the run; nothing reads it back.

The model is loaded in `model.dtype` (bfloat16), which is the dtype the Fisher is defined on (ADR 0003). `quantize_model` casts each weight to float32 on its own. The pipeline is the one place that decides which tensors are clustered, so ADR 0006's rotation becomes a change to the `weights` mapping above, not to `quantize_model`.

## Evaluation pipeline

`run_evaluation` never quantizes. A missing artifact is an error that names the command that would create it, because silently quantizing inside an evaluation run would hide an expensive, cache-populating step.

```python
def run_evaluation(cfg: EvaluateRunConfig, run_dir: Path) -> Results:
    resolve_rotation(cfg.rotation)
    device = resolve_device(cfg.device)
    seed_everything(cfg.seed)
    store = ArtifactStore(cfg.output.cache_dir)

    # Recompute each requested mode's artifact key; only quantizer.mode differs between them.

    f_key = sha256_key(fisher_snapshot(cfg.model, cfg.calibration, cfg.rotation))
    requested = {
        mode: quantized_snapshot(f_key, cfg.quantizer.model_copy(update={"mode": mode}))
        for mode in cfg.modes
    }

    # Two independent copies in eval_dtype: the reference is never modified (ADR 0004).

    dtype = torch_dtype(cfg.model.eval_dtype)
    reference = load_model(cfg.model, dtype, device).eval()
    quantized = load_model(cfg.model, dtype, device).eval()
    targets = find_quantizable_linears(quantized, cfg.model.quantizable_modules)
    modules = [ModuleEntry(name=n, shape=tuple(l.weight.shape)) for n, l in targets.items()]

    # Load every artifact up front, so a missing one fails before hours of evaluation.

    artifacts = {mode: _load_or_explain(store, snapshot, modules, mode) for mode, snapshot in requested.items()}

    # Tokenize each evaluation dataset once; the streams are reused for every (mode, bits).

    encode = make_encoder(load_tokenizer(cfg.model))
    eval_tokens = {
        name: load_eval_tokens(load_texts(d.path, d.name, d.data_files, d.split, d.text_field), encode, d)
        for name, d in cfg.eval.datasets.items()
    }

    # Prove the sliced-head metric path reproduces forward(), then that both instances agree,
    # before any quantized weight is written (spec 0008).

    probe = eval_tokens[cfg.eval.kl.dataset][None, :512].to(device)               # [1, 512]
    check_sliced_logits(reference, probe, cfg.eval.lm_head_chunk_tokens)
    check_sliced_logits(quantized, probe, cfg.eval.lm_head_chunk_tokens)
    _assert_identical_models(reference, quantized, eval_tokens[cfg.eval.kl.dataset], cfg.eval, device)
    report = bits_report([m.shape for m in modules], _count_params(reference), cfg.quantizer.seed_bits, cfg.quantizer.parent_bits)

    # The metric loop of spec 0008: reference metrics once, then every (mode, bits).

    reference_entry, entries = _sweep(reference, quantized, artifacts, eval_tokens, cfg, report, device)

    results = Results(schema_version=RESULTS_SCHEMA_VERSION, config=cfg, fisher_key=f_key, bits=..., reference=reference_entry, entries=entries, ...)
    (run_dir / "results.json").write_text(results.model_dump_json(indent=2))
    return results
```

`_load_or_explain` turns `ArtifactNotFoundError` into an error message that includes the exact command, for example `python quantization/quantize_any_precision.py quantizer.mode=standalone`. `ArtifactMismatchError` propagates unchanged.

`results.json` is written with a temporary file and `os.replace`, the same pattern as the artifact store, so an interrupted run never leaves a truncated results file.

## Entry scripts

Each script is under 30 lines. It reads Hydra's run directory once, and passes it on, so no library module reads the working directory (spec 0001).

```python
# quantization/quantize_any_precision.py
@hydra.main(version_base=None, config_path="../configs", config_name="quantize")
def main(cfg: DictConfig) -> None:
    """Quantize the configured model into an any-precision artifact."""
    run_config = load_quantize_config(cfg)
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    outcome = run_quantization(run_config, run_dir)
    logger.info("fisher %s (%s)", outcome.fisher_key[:16], "reused" if outcome.fisher_reused else "computed")
    logger.info("quantized %s at %s", outcome.quantized_key[:16], outcome.quantized_dir)

if __name__ == "__main__":
    main()
```

`evaluation/evaluate_any_precision.py` is identical in shape. It calls `load_evaluate_config` and `run_evaluation`, and logs one line per `QuantizedEntry`: mode, bits, KL mean, agreement, and WikiText-2 perplexity. Logging goes through the standard `logging` module, which Hydra configures; progress bars use `tqdm` inside the pipelines.

## Typical commands

These are the commands the acceptance run uses (spec 0011). They are listed here so the config overrides are documented next to the code that reads them.

```bash
# Incremental artifact; computes and caches the Fisher on the first run.
python quantization/quantize_any_precision.py

# Standalone baseline; reuses the cached Fisher.
python quantization/quantize_any_precision.py quantizer.mode=standalone

# Both modes, all bit-widths, WikiText-2 and C4.
python evaluation/evaluate_any_precision.py

# Smoke run: two chunks per dataset, incremental only.
python evaluation/evaluate_any_precision.py eval.max_chunks=2 'modes=[incremental]'

# Rejected before any download: Hadamard is not implemented.
python quantization/quantize_any_precision.py rotation=hadamard
```

## Verification

Tests for this spec are listed in spec 0010 under `tests/quantization/test_pipeline.py` and `tests/evaluation/test_pipeline.py`. They run offline on the CPU, with `load_model`, `load_tokenizer`, and `load_texts` monkeypatched to return the tiny Granite fixture, the character encoder, and a fixed list of strings.

- `run_quantization` on an empty cache computes and saves the Fisher and the artifact, and writes `quantize_summary.json`.
- A second identical call reports `fisher_reused` and `quantized_reused`, and does not call `estimate_fisher` or `quantize_model`; the tests patch both with spies that fail if called.
- Changing only `quantizer.mode` reuses the Fisher and creates a second artifact.
- `rotation=hadamard` raises `NotImplementedError` before any loader is called, and `device="cuda"` on a CUDA-less machine raises `RuntimeError`.
- `run_evaluation` with a missing standalone artifact raises before tokenizing any dataset, and the message contains `quantizer.mode=standalone`.
- `run_evaluation` on the tiny model writes a `results.json` that validates as `Results`, with one entry per (mode, bits), in order.
- The entry scripts compose through Hydra's `compose` API, and a subprocess run of `quantize_any_precision.py --cfg job` exits 0. This proves the `config_path` is correct without running a pipeline.
