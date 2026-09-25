# Spec 0008: Evaluation Metrics

- Status: Proposed
- Wave: [0](index.md)
- Implements: [ADR 0004](../../adr/0004-evaluation-protocol.md), Metric 1 (KL and top-1 agreement), Metric 2 (perplexity), Metric 5 (bits per weight), and the results format

This spec defines how wave 0 measures quality: streaming KL divergence and top-1 agreement against the reference model, perplexity on WikiText-2 and C4, analytic bits per weight, and the `results.json` schema. `lm-eval` and generations (ADR 0004, Metrics 3 and 4) are out of scope for this wave.

## Files

Metric arithmetic and bits accounting are pure `torch` and plain Python, so they are unit-tested without a model. The model-specific pieces (running the body, and applying the LM head) live in `models/heads.py` (spec 0003). The loop that drives the models is part of the evaluation pipeline (spec 0009).

| File | Contents |
| --- | --- |
| `evaluation/metrics.py` | `ChunkOutputs`, `ChunkMetrics`, `chunk_metrics`, `StreamingMetrics`, `MetricSummary` |
| `evaluation/bits.py` | `layer_bits_per_weight`, `parent_bits_per_weight`, `bits_report`, `BitsReport` |
| `evaluation/results.py` | `Results`, `ReferenceEntry`, `QuantizedEntry`, `PerplexityResult` |

## Per-chunk metrics

Full logits for a 2048-token chunk are `[2048, 100352]`, 0.82 GB in float32, and Granite's `forward` briefly holds two copies while it applies `logits_scaling`. The metrics are all per position, so they never need every position's logits at once. Each model runs its body once per chunk, producing hidden states `[T, H]` (8 MB). The LM head is then applied to `eval.lm_head_chunk_tokens` positions at a time (default 256). The result is mathematically identical to `model(x).logits`. On the tiny Granite model the difference was below $10^{-7}$, from floating-point summation order.

`eval.lm_head_chunk_tokens` is configured in Hydra (spec 0002):

| Value | Behaviour |
| --- | --- |
| A power of two, such as `256` | The head and the metrics run on slices of that many positions; only one slice of logits per model exists at a time |
| `null` | The head runs on all $T$ positions at once, and the metrics are computed on the full `[T, V]` logits in a single step |

`null` suits a GPU with more memory, or CPU tests. For a 2048-token chunk on Granite, it needs about 3.3 GB of logit-sized float32 tensors on top of the two models, which does not fit the 6 GB laptop GPU. Either value computes the same numbers, up to floating-point summation order, so the setting is not a protocol change. It is recorded in the config stored in `results.json`, but results with different values remain comparable.

```python
LogitHead = Callable[[Tensor], Tensor]            # [S, H] -> [S, V], including any logit scaling

@dataclass(frozen=True)
class ChunkOutputs:
    """One model's output for one chunk, before the LM head."""
    hidden: Tensor                                # [T, H], final-norm hidden states
    head: LogitHead

@torch.no_grad()
def chunk_metrics(
    q: ChunkOutputs,
    ref: ChunkOutputs | None,       # None for perplexity-only datasets
    tokens: Tensor,                 # [T] int64, same device
    slice_len: int | None,          # eval.lm_head_chunk_tokens; None processes all T positions at once
) -> ChunkMetrics:
    """Per-position KL, top-1 agreement, and the chunk's mean next-token NLL for model q.

    Position t predicts token t + 1, so the last position has no target for NLL, while KL and
    agreement use all T positions.
    """
```

```python
def chunk_metrics(q, ref, tokens, slice_len) -> ChunkMetrics:
    T = tokens.shape[0]
    kl_parts, agree_parts = [], []
    nll_sum = 0.0

    # A null setting is one slice spanning the whole chunk, so both modes share a single code path.

    step = T if slice_len is None else slice_len
    for s in range(0, T, step):
        # One slice of positions: apply the LM head to [S, H] -> [S, V], then work in float32.
        # Only one slice of logits per model exists at any time.

        e = min(s + step, T)
        q_logp = q.head(q.hidden[s:e]).float().log_softmax(-1)                   # [S, V]

        # Next-token NLL for positions whose target lies inside the chunk.

        targets = tokens[s + 1 : e + 1]                                           # [S'] with S' = S, or S - 1 on the last slice
        nll_sum -= q_logp[: targets.shape[0]].gather(1, targets[:, None]).sum().item()

        if ref is not None:
            # KL(p_ref || p_q) per position, reduced over the vocabulary: [S, V] -> [S].

            ref_logp = ref.head(ref.hidden[s:e]).float().log_softmax(-1)         # [S, V]
            kl_parts.append((ref_logp.exp() * (ref_logp - q_logp)).sum(-1).cpu())
            agree_parts.append((ref_logp.argmax(-1) == q_logp.argmax(-1)).cpu())

    return ChunkMetrics(
        kl=torch.cat(kl_parts) if kl_parts else None,                             # [T] float32
        agree=torch.cat(agree_parts) if agree_parts else None,                    # [T] bool
        mean_nll=nll_sum / (T - 1),
    )
```

The KL formula follows ADR 0004 exactly. Tiny negative values from float32 cancellation are kept as they are, not clamped, and the unit tests bound them at $-10^{-6}$.

The per-chunk mean NLL matches Hugging Face's `ForCausalLMLoss` with `labels=input_ids`: the labels are shifted, and the mean is taken over $T - 1$ targets. A unit test checks that the two agree.

Slicing relies on three facts about the model: its body is `get_decoder()`, its head is `get_output_embeddings()`, and the only post-processing is division by `config.logits_scaling`. A model that adds another step, such as logit soft-capping, would silently diverge. Two guards cover this. A unit test compares sliced and full logits on the tiny Granite model. At startup, `run_evaluation` runs `check_sliced_logits` (spec 0003) on both models and raises before the sweep if the two paths disagree.

## Streaming accumulation

`StreamingMetrics` collects chunk results for one (model, dataset) pair. Per-position KL values are kept on the CPU so the 99th percentile is exact. WikiText-2 is about 0.3M tokens, so this is about 1.2 MB.

```python
class StreamingMetrics:
    def __init__(self, quantile: float) -> None: ...
    def update(self, chunk: ChunkMetrics) -> None: ...
    def summary(self) -> MetricSummary: ...

@dataclass(frozen=True)
class MetricSummary:
    perplexity: float                  # exp(mean of per-chunk mean NLL), ADR 0004
    mean_nll: float
    num_chunks: int
    num_tokens: int
    kl_mean: float | None
    kl_quantile: float | None          # torch.quantile over all positions, at `quantile`
    top1_agreement: float | None
```

Perplexity is $\exp$ of the mean of the per-chunk mean NLLs, as in ADR 0004 and the reference `evaluate_ppl`. It is not $\exp$ of the per-token mean, although with equal-length chunks the two coincide. `summary()` raises `ValueError` if no chunk was added, and returns `None` for the KL fields if any chunk lacked reference logits.

`torch.quantile` rejects inputs above $2^{24}$ elements. The stored KL tensor stays well below that for WikiText-2, and `summary()` raises if the limit is ever exceeded, rather than subsampling silently.

## Evaluation loop

This is the orchestration inside `run_evaluation` (spec 0009). It is shown here because it fixes what each metric is computed on.

```python
# The heads never change: set_precision touches only the decoder linears, never the tied LM head.

ref_head, q_head = logit_head(reference), logit_head(quantized)

# Reference metrics are computed once per dataset; they do not depend on mode or bit-width.

for dataset_name, tokens in eval_tokens.items():                      # tokens: [L] int64 on the CPU
    ref_stats = StreamingMetrics(cfg.eval.kl.quantile)
    for chunk in iter_chunks(tokens, cfg.eval.chunk_len, cfg.eval.max_chunks):   # [1, T]
        x = chunk.to(device)                                              # [1, T]
        ref_out = ChunkOutputs(body_hidden_states(reference, x), ref_head)  # hidden [T, H]
        ref_stats.update(chunk_metrics(ref_out, None, x[0], cfg.eval.lm_head_chunk_tokens))
    reference_results[dataset_name] = ref_stats.summary()

for mode in cfg.modes:
    artifact = store.load_quantized(...)                               # CPU tensors
    for bits in cfg.eval.bits:
        set_precision(quantized, artifact, bits)
        summaries: dict[str, MetricSummary] = {}
        for dataset_name, tokens in eval_tokens.items():
            # KL needs the reference distribution for the same chunk, so the reference body is run
            # again side by side; caching 100,352-wide logits is ruled out by ADR 0004.

            with_kl = dataset_name == cfg.eval.kl.dataset
            stats = StreamingMetrics(cfg.eval.kl.quantile)
            for chunk in iter_chunks(tokens, cfg.eval.chunk_len, cfg.eval.max_chunks):   # [1, T]
                x = chunk.to(device)                                      # [1, T]
                q_out = ChunkOutputs(body_hidden_states(quantized, x), q_head)
                ref_out = ChunkOutputs(body_hidden_states(reference, x), ref_head) if with_kl else None
                stats.update(chunk_metrics(q_out, ref_out, x[0], cfg.eval.lm_head_chunk_tokens))
            summaries[dataset_name] = stats.summary()

        # One entry per (mode, bits): perplexity for every dataset, KL fields from the KL dataset.

        entries.append(make_quantized_entry(mode, bits, artifact.manifest.key, summaries, cfg.eval.kl.dataset, bits_report))
```

`make_quantized_entry` raises `ValueError` if the KL dataset's summary has no KL values, which can only happen through a programming error in the loop.

Every forward pass runs under `torch.inference_mode()` with `use_cache=False`. Both models are loaded once, in `eval_dtype`, and put in `eval()` mode. The reference model is never passed to `set_precision`.

Before the sweep, the pipeline runs two checks. The first is `check_sliced_logits` on both models (spec 0003), which proves that body plus sliced head reproduces `model(x).logits`.

The second is a same-weights check. Before the sweep, the pipeline runs the first KL chunk through both models, while `quantized` still holds its original weights, and asserts that the mean KL is at most $10^{-6}$ and agreement is exactly 1. The KL bound is a tolerance rather than exact zero because CUDA kernels need not be bitwise reproducible across two model instances. This proves the two instances and the metric code are identical before any quantization effect is measured.

## GPU memory budget

Evaluation is the most memory-hungry stage, because two float32 models are resident at once. Slicing the LM head keeps the logits small. The budget below is for a 2048-token chunk on the 6 GB laptop GPU, with the default `lm_head_chunk_tokens: 256`. With `null`, the logit rows below grow by a factor of eight, to about 3.3 GB.

| Item | Size |
| --- | --- |
| Reference model, float32 | 1.41 GB |
| Quantized model, float32 | 1.41 GB |
| Hidden states `[2048, 1024]`, float32, one per model | 16 MB |
| Body activations during one forward pass (freed before the head runs) | about 0.1 GB |
| Logit slices `[256, 100352]` float32: head output, scaled copy, and log-softmax, per model | about 0.6 GB |
| One module's artifact data during `set_precision` | at most 8 MB |
| Total | about 3.5 GB, plus the CUDA context |

`check_sliced_logits` runs the full `forward` once on a short sequence (spec 0003), so its logits stay far below the budget above.

## Bits per weight

Bits per weight is analytic (ADR 0004, Metric 5). It depends only on module shapes and the bit-width range, so `bits.py` takes a list of shapes, not a model.

```python
def layer_bits_per_weight(bits: int, n: int) -> float:
    """b + 16 * 2**b / n: indices plus one float16 codebook per row."""

def parent_bits_per_weight(seed_bits: int, parent_bits: int, n: int) -> float:
    """B + (16 / n) * sum_{b=b0}^{B} 2**b: parent indices plus every codebook."""

@dataclass(frozen=True)
class BitsReport:
    quantized_params: int
    total_params: int
    per_bits: dict[int, float]            # parameter-weighted average over quantized layers
    per_bits_whole_model: dict[int, float] # all other parameters counted at 16 bits
    parent: float
    parent_whole_model: float

def bits_report(shapes: Sequence[tuple[int, int]], total_params: int, seed_bits: int, parent_bits: int) -> BitsReport: ...
```

`total_params` is `sum(p.numel() for p in model.parameters())`. It counts the tied embedding and LM head once, which matches how they are stored. For Granite 4.0 350M these are the expected values; the full run's `results.json` must match them to four decimals (spec 0011).

| Bits | Quantized layers | Whole model |
| --- | --- | --- |
| 3 | 3.1103 | 6.8713 |
| 4 | 4.2206 | 7.6576 |
| 5 | 5.4412 | 8.5221 |
| 6 | 6.8824 | 9.5427 |
| 7 | 8.7647 | 10.8758 |
| 8 | 11.5294 | 12.8339 |
| Resident parent, 3 to 8 bits | 14.9485 | 15.2553 |

*168 quantized modules hold 249,561,088 of 352,379,904 parameters. The 2048-wide `output_linear` rows halve the codebook overhead, which is why the parent figure sits below ADR 0004's 15.875 for 1024-wide rows.*

## Results schema

`results.json` is validated by the pydantic models below and written into the Hydra run directory. Each file is self-contained: it repeats the reference metrics and the full resolved config.

```python
RESULTS_SCHEMA_VERSION: int = 1

class PerplexityResult(_Frozen):
    dataset: Literal["wikitext2", "c4"]
    perplexity: float
    mean_nll: float
    num_chunks: int
    num_tokens: int

class ReferenceEntry(_Frozen):
    perplexity: list[PerplexityResult]

class QuantizedEntry(_Frozen):
    mode: Literal["incremental", "standalone"]
    bits: int
    artifact_key: str
    perplexity: list[PerplexityResult]
    kl_dataset: Literal["wikitext2", "c4"]
    kl_mean: float
    kl_quantile: float
    top1_agreement: float
    bits_per_weight: float
    bits_per_weight_whole_model: float

class Results(_Frozen):
    schema_version: int
    config: EvaluateRunConfig
    model_id: str
    revision: str
    fisher_key: str
    versions: dict[str, str]
    created_at: datetime
    bits: BitsReportModel                 # pydantic mirror of BitsReport
    reference: ReferenceEntry
    entries: list[QuantizedEntry]         # ordered by (mode, bits)
    seconds: float
```

`Results` stores the resolved `EvaluateRunConfig` itself, so a results file can be re-validated and diffed against another without parsing the Hydra logs.

## Verification

Tests for this spec are listed in spec 0010 under `tests/evaluation/`. They run on the CPU with random hidden states and a seeded random `nn.Linear` head, so no model is loaded. The expected values come from the full logits `head(hidden)`.

- `chunk_metrics` with identical reference and quantized outputs gives KL 0 (within $10^{-6}$) and agreement 1.
- KL matches `torch.nn.functional.kl_div(q_logp, ref_logp, log_target=True, reduction="none").sum(-1)` computed on the full logits, and is the same for `slice_len=None` and for every integer `slice_len`, including values that do not divide $T$.
- The mean NLL equals `ForCausalLMLoss(logits[None], tokens[None], vocab_size)` from `transformers` on the full logits, and does not depend on `slice_len`.
- `StreamingMetrics` perplexity is $\exp$ of the mean of the per-chunk means. A two-chunk example is constructed where this differs from the token-level mean, and the test checks the documented choice.
- The KL quantile equals `torch.quantile` over the concatenated positions; `summary()` without updates raises.
- `layer_bits_per_weight(3, 1024) == 3.125`, `layer_bits_per_weight(8, 1024) == 12.0`, and `parent_bits_per_weight(3, 8, 1024) == 15.875`, matching the ADR 0004 table. `bits_report` on Granite's 168 shapes, with `total_params = 352_379_904`, reproduces the table above.
- `Results` round-trips through JSON, and rejects an entry whose `bits` lies outside the config's range.
