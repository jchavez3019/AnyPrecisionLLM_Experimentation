# Spec 0003: Model and Data Loading

- Status: Proposed
- Wave: [0](index.md)
- Implements: [ADR 0002](../../adr/0002-project-layout-and-architecture.md) (Hugging Face directly), [ADR 0004](../../adr/0004-evaluation-protocol.md) (evaluation text)

This spec covers the boundary with Hugging Face: loading Granite and its tokenizer, finding the 168 quantizable linears, sampling calibration sequences, and building the evaluation token streams. Everything past this boundary sees plain tensors.

## Files

Six modules own the Hugging Face boundary. Each keeps untyped library objects local and returns explicitly typed values.

| File | Contents |
| --- | --- |
| `src/anyprec/models/loading.py` | `CausalLM`, `load_model`, `load_tokenizer` |
| `src/anyprec/models/discovery.py` | `find_quantizable_linears`, `QuantizableModuleError` |
| `src/anyprec/models/heads.py` | `causal_lm_loss` (spec 0004), `body_hidden_states`, `logit_head`, `check_sliced_logits`, `SlicedLogitsError` (spec 0008) |

`models/` is the only subpackage that touches a Hugging Face model object. `CausalLM` is an alias of `transformers.PreTrainedModel`, so modules outside the boundary (`sensitivity/`, the pipelines) can name the type without importing `transformers`, and every untyped model output, such as `ModelOutput.loss` or `last_hidden_state`, is narrowed to a `Tensor` inside `models/`.
| `src/anyprec/data/hub.py` | `load_texts` (spec 0009) |
| `src/anyprec/data/calibration.py` | `Encoder`, `make_encoder`, `sample_calibration`, `CalibrationError` |
| `src/anyprec/data/evaluation_text.py` | `load_eval_tokens`, `iter_chunks` |

## Loading the model

Models come from `from_pretrained`, as ADR 0002 requires. The dtype is a parameter because quantization uses `model.dtype` (bfloat16) while evaluation uses `model.eval_dtype` (float32).

```python
type CausalLM = PreTrainedModel

def load_model(cfg: ModelConfig, dtype: torch.dtype, device: torch.device) -> CausalLM:
    """Load the causal LM in inference mode on one device."""
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, revision=cfg.revision, dtype=dtype, device_map=str(device)
    )

    # transformers is only partially typed; narrow once here so callers see PreTrainedModel.

    if not isinstance(model, PreTrainedModel):
        raise TypeError(f"{cfg.model_id} did not load as a PreTrainedModel")
    model.eval()
    return model

def load_tokenizer(cfg: ModelConfig) -> PreTrainedTokenizerBase:
    """Load the tokenizer pinned to the same revision as the weights."""
```

## Discovering quantizable modules

Discovery walks `named_modules()`, keeps the `nn.Linear` layers whose qualified names match the configured pattern, and treats a count mismatch as a hard error (ADR 0002).

```python
class QuantizableModuleError(RuntimeError):
    """The model's quantizable modules do not match the configuration."""

def find_quantizable_linears(model: nn.Module, cfg: QuantizableModules) -> dict[str, nn.Linear]:
    """Return matching linear layers in named_modules() order, keyed by qualified name."""
    pattern = re.compile(cfg.pattern)
    found = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and pattern.match(name)
    }
    if len(found) != cfg.expected_count:
        raise QuantizableModuleError(f"expected {cfg.expected_count} quantizable linears, found {len(found)}")

    # Tied weights would make one tensor appear under two names and be quantized twice.

    pointers = [module.weight.data_ptr() for module in found.values()]
    if len(set(pointers)) != len(pointers):
        raise QuantizableModuleError("two quantizable modules share one weight tensor")
    return found
```

The order of the returned dictionary is the canonical module order everywhere else: the Fisher cache, artifact manifests, `stats.json`, and results.

## Body and LM head

Evaluation applies the LM head in slices of positions to bound GPU memory (spec 0008). These three functions are the only code that knows how a causal LM splits into a body and a head. The split was verified on the tiny Granite model: `get_decoder()` returns `GraniteMoeHybridModel`, `get_output_embeddings()` returns the tied `nn.Linear`, and sliced logits match `forward` to within $10^{-7}$.

```python
class SlicedLogitsError(RuntimeError):
    """Body plus sliced head does not reproduce model(x).logits for this model."""

def causal_lm_loss(model: CausalLM, input_ids: torch.Tensor) -> torch.Tensor:
    """Mean next-token NLL of [1, T] token ids, as a 0-d tensor attached to the autograd graph.

    Uses labels=input_ids, so Hugging Face shifts the labels and averages over T - 1 targets.
    """
    loss = model(input_ids=input_ids, labels=input_ids, use_cache=False).loss
    if not isinstance(loss, torch.Tensor):
        raise TypeError(f"{type(model).__name__} returned no loss for labelled input")
    return loss

def body_hidden_states(model: PreTrainedModel, input_ids: torch.Tensor) -> torch.Tensor:
    """Run the decoder body only: [1, T] token ids -> [T, H] final-norm hidden states."""
    body = model.get_decoder()
    return body(input_ids=input_ids, use_cache=False).last_hidden_state[0]

def logit_head(model: PreTrainedModel) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return h [S, H] -> logits [S, V], including Granite's division by config.logits_scaling."""
    head = model.get_output_embeddings()
    if not isinstance(head, nn.Linear):
        raise SlicedLogitsError(f"output embeddings are {type(head).__name__}, not nn.Linear")
    scale = float(getattr(model.config, "logits_scaling", 1.0))
    return lambda hidden: head(hidden) / scale

@torch.inference_mode()
def check_sliced_logits(model: PreTrainedModel, input_ids: torch.Tensor, slice_len: int | None, atol: float = 1e-4) -> None:
    """Raise SlicedLogitsError unless body plus head matches model(input_ids).logits within atol.

    slice_len None applies the head to every position at once, which still checks the body/head split.
    """
    full = model(input_ids=input_ids, use_cache=False).logits[0]                 # [T, V]
    hidden = body_hidden_states(model, input_ids)                                # [T, H]
    head = logit_head(model)
    step = hidden.shape[0] if slice_len is None else slice_len
    sliced = torch.cat([head(hidden[s : s + step]) for s in range(0, hidden.shape[0], step)])
    error = (sliced - full).abs().max().item()
    if error > atol:
        raise SlicedLogitsError(f"sliced logits differ from forward() by {error:.3g} (atol {atol})")
```

The pipeline calls `check_sliced_logits` on a 512-token prefix of the KL dataset, with `slice_len = eval.lm_head_chunk_tokens`, the same setting the metrics use. At that length the full logits are 0.2 GB. The `atol` of $10^{-4}$ is loose enough for float32 summation-order differences, and far below any real change to the head, such as soft-capping.

## Tokenization boundary

The data functions accept an `Encoder` instead of a tokenizer. This keeps them testable with a trivial in-code encoder and avoids the untyped `__call__` of Hugging Face tokenizers.

```python
Encoder = Callable[[str], list[int]]

def make_encoder(tokenizer: PreTrainedTokenizerBase) -> Encoder:
    """Wrap a tokenizer as raw-text encoding with no special tokens (ADR 0002)."""
    def encode(text: str) -> list[int]:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        return cast(list[int], ids)
    return encode
```

`make_encoder` lives in `data/calibration.py` and is exported. The `cast` is the single place where the tokenizer's return type is narrowed (ADR 0001, Section 4).

## Calibration sampling

The rule is ADR 0002's, and it is identical to the prototype in notebook 02: a seeded permutation of documents, raw text, documents shorter than `seq_len` skipped, and the first `seq_len` tokens of each accepted document kept.

```python
class CalibrationError(ValueError):
    """Too few documents are long enough to fill the calibration set."""

def sample_calibration(texts: Sequence[str], encode: Encoder, cfg: CalibrationConfig) -> torch.Tensor:
    """Sample calibration sequences.

    :return: int64 token ids of shape [cfg.num_sequences, cfg.seq_len], on the CPU.
    """
    # A CPU generator makes the document order identical on every machine.

    order = torch.randperm(len(texts), generator=torch.Generator().manual_seed(cfg.seed)).tolist()
    accepted: list[torch.Tensor] = []
    for index in order:
        ids = encode(texts[index])
        if len(ids) < cfg.seq_len:
            continue
        accepted.append(torch.tensor(ids[: cfg.seq_len], dtype=torch.long))
        if len(accepted) == cfg.num_sequences:
            break
    if len(accepted) < cfg.num_sequences:
        raise CalibrationError(...)

    # num_sequences x [seq_len] -> [num_sequences, seq_len]

    return torch.stack(accepted)
```

The pipeline (spec 0009) obtains `texts` from `load_texts` in `data/hub.py`, which calls `load_dataset(cfg.path, name=cfg.name, data_files=cfg.data_files, split=cfg.split)` and narrows `dataset[cfg.text_field]` to `list[str]` with `isinstance` checks.

## Evaluation token streams

ADR 0004 defines both evaluation texts as single long token streams, cut into non-overlapping chunks with the remainder dropped.

```python
def load_eval_tokens(texts: Sequence[str], encode: Encoder, cfg: EvalDatasetConfig) -> torch.Tensor:
    """Join documents with cfg.joiner, tokenize once, and truncate to cfg.max_tokens.

    :return: int64 token ids of shape [num_tokens], on the CPU.
    """
    if cfg.max_tokens is None:
        return torch.tensor(encode(cfg.joiner.join(texts)), dtype=torch.long)

    # Tokenizing the whole C4 validation shard is wasteful when only 2**19 tokens are needed.
    # Grow the joined prefix in blocks of 1000 documents until it is long enough, then truncate.

    count = 0
    while True:
        count = min(count + 1000, len(texts))
        ids = encode(cfg.joiner.join(texts[:count]))
        if len(ids) >= cfg.max_tokens or count == len(texts):
            return torch.tensor(ids[: cfg.max_tokens], dtype=torch.long)

def iter_chunks(tokens: torch.Tensor, chunk_len: int, max_chunks: int | None) -> Iterator[torch.Tensor]:
    """Yield non-overlapping [1, chunk_len] chunks; the trailing remainder is dropped."""
    num_chunks = tokens.numel() // chunk_len
    if max_chunks is not None:
        num_chunks = min(num_chunks, max_chunks)
    for i in range(num_chunks):
        yield tokens[i * chunk_len : (i + 1) * chunk_len].unsqueeze(0)
```

Joining the first $k$ documents and tokenizing once, with $k$ a multiple of 1000, is deterministic, and it tokenizes the same text for every model under evaluation. That is all ADR 0004 needs. The block size is a module constant, not configuration, because it does not change which tokens are produced once `max_tokens` is reached.

## Verification

Tests for this spec are listed in spec 0010 under `tests/models/` and `tests/data/`. None of them downloads anything.

- On the tiny Granite fixture, with `logits_scaling=4`, `check_sliced_logits` passes for `None` and for several `slice_len` values, including one that does not divide $T$. It raises `SlicedLogitsError` when a forward hook soft-caps the model's logits after the head, which body plus head does not reproduce. Patching the head itself would not be detected, since both paths call it. `logit_head` and `body_hidden_states` raise `SlicedLogitsError` for a null `logits_scaling`, a non-linear output embedding, and a missing decoder body.
- `find_quantizable_linears` on the tiny Granite fixture (spec 0010) returns the 12 expected names in `named_modules()` order. It raises `QuantizableModuleError` when `expected_count` is wrong, and when two matched modules are made to share one weight.
- `sample_calibration` with a character-level encoder and an in-memory list of strings:
  - it skips documents that are too short;
  - it returns exactly `[num_sequences, seq_len]`;
  - it is identical across calls with the same seed and differs with a different seed;
  - it raises `CalibrationError` when too few documents qualify.
- `load_eval_tokens` equals `encode(joiner.join(texts))` when `max_tokens` is `None`, and returns exactly `max_tokens` tokens otherwise.
- `iter_chunks` drops the remainder, respects `max_chunks`, and yields `[1, chunk_len]` views of the original tensor.
- `load_model` and `load_tokenizer` are covered only by the `network` integration test in spec 0011, since they are pass-through wrappers (ADR 0001, Section 2).
