# ADR 0004: Evaluation Protocol for Answer Quality

- Status: Proposed
- Date: 2026-09-24
- Deciders: Project maintainer

This ADR fixes how quantized models are judged, so that results across bit-widths, quantizer modes, and future milestones (for example Hadamard rotation) are directly comparable. The project's goal is answer quality, not latency; no timing is measured.

## Context

The paper evaluates with perplexity on WikiText-2, C4, and PTB, and zero-shot accuracy on five `lm-eval` tasks (its Section 6.1 and Appendix A). Those are the community conventions and are kept for comparability. They are, however, blunt instruments for a 350M-parameter model: zero-shot accuracy on hard tasks sits near chance, and perplexity averages away where the model changes its mind.

[ADR 0003](0003-fisher-weighted-kmeans-methodology.md) derives the quantization objective as a second-order approximation of the KL divergence between the original and quantized next-token distributions. Measuring that KL divergence directly is therefore the most faithful quality metric, and it is added as the primary metric.

Every configuration is evaluated against the same reference: the unquantized model, run through the same code path.

## Decision

### Models under evaluation

For each quantized artifact, evaluate every bit-width $b \in \lbrace 3,\dots,8\rbrace$, plus the reference model. With `mode: incremental` and `mode: standalone` artifacts both available, each bit-width is reported for both, reproducing the paper's IU-versus-standalone comparison.

Both the reference and the quantized models run in `eval_dtype` (float32 by default; see [ADR 0005](0005-artifact-format-and-simulated-inference.md)), so differences come only from the weights.

### Metric 1 (primary): KL divergence and top-1 agreement

On the WikiText-2 test text, in the same chunks as the perplexity metric, run the reference and quantized models side by side and compute per token position:

$$
\begin{aligned}
\mathrm{KL}_t &= \sum_{v} p_{\text{ref}}(v \mid x_{\lt t}) \left(\log p_{\text{ref}}(v \mid x_{\lt t}) - \log p_{q}(v \mid x_{\lt t})\right), \\
\mathrm{agree}_t &= \mathbb{1}\left[\arg\max_v p_{\text{ref}} = \arg\max_v p_q\right].
\end{aligned}
$$

Report the mean $\mathrm{KL}_t$, its 99th percentile (rare large disagreements matter for generation), and mean top-1 agreement. Logits over the 100,352-token vocabulary are never cached; both models are resident at once (about 1.4 GB each in float32) and compared chunk by chunk. Within a chunk, each model's decoder body runs once, and the LM head is applied to a few hundred positions at a time. A full `[2048, 100352]` float32 logit tensor is 0.82 GB, and Granite's `forward` briefly holds two of them while it applies `logits_scaling`, which would not fit next to two models on a 6 GB GPU. The sliced path is mathematically identical to `forward`, and each run verifies this at startup before measuring anything.

### Metric 2: perplexity

Perplexity follows the paper's Appendix A and the reference `evaluate_ppl`:

- **WikiText-2**: the `test` split of `Salesforce/wikitext` / `wikitext-2-raw-v1`, joined with `"\n\n"`, tokenized once.
- **C4**: the first validation shard (`en/c4-validation.00000-of-00008.json.gz`), documents joined with `" "`, truncated to the first $256 \times 2048$ tokens.
- The token stream is split into non-overlapping 2048-token chunks (the trailing remainder is dropped). Perplexity is $\exp$ of the mean of the per-chunk mean negative log-likelihoods.

PTB is omitted: its dataset script requires `trust_remote_code`, and it adds little beyond WikiText-2 for this project.

### Metric 3: zero-shot tasks

Run `lm-eval` zero-shot on `piqa`, `hellaswag`, and `arc_easy`, reporting byte-length-normalized accuracy (`acc_norm`) where available and `acc` otherwise. `arc_challenge` and `winogrande` are optional (disabled by default) because a 350M model scores near chance on them. An eval-config `limit` caps examples per task so a full sweep fits on the laptop GPU; the limit is recorded in results, and results with different limits are not compared.

If `lm-eval` turns out to be incompatible with `transformers` 5.x (a risk noted in [ADR 0002](0002-project-layout-and-architecture.md)), the fallback is a minimal in-house multiple-choice log-likelihood scorer for the same tasks, recorded as such in results.

### Metric 4: qualitative generations

A fixed prompt set lives in `configs/eval/prompts.yaml` (about 20 prompts covering factual recall, short reasoning, instruction following, and code). Each prompt is rendered with the tokenizer's chat template, decoded greedily (`do_sample=False`) for at most 128 new tokens, and saved to JSON per model and bit-width for side-by-side reading. Greedy decoding makes the outputs deterministic, so differences reflect the weights only.

### Metric 5: honest bits per weight

Memory claims must count everything. For a quantized layer with $m$ rows and $n$ columns, at bit-width $b$ with per-row float16 codebooks:

$$
\text{bpw}_{\text{layer}}(b) = b + \frac{16 \cdot 2^{b}}{n}.
$$

The resident any-precision parent (indices at $B$ bits plus every codebook from $b_0$ to $B$) costs

$$
\text{bpw}_{\text{parent}} = B + \frac{16}{n}\sum_{b=b_0}^{B} 2^{b}.
$$

For Granite's 1024-wide rows (five of the six linear layers per block):

| Configuration | Bits per weight |
| --- | --- |
| 3-bit only | 3.125 |
| 4-bit only | 4.25 |
| 8-bit only | 12.0 |
| Resident parent, 3 to 8 bits | 15.875 |
| Original bfloat16 | 16.0 |

*Per-row codebooks are cheap at low bit-widths but dominate at high ones on narrow rows; the paper's 4096-wide rows hide this.*

Results report the average over quantized layers weighted by parameter count, and separately a whole-model figure that counts the unquantized tied embedding / LM head at 16 bits.

### Results format

Each evaluation run writes `results.json` into its Hydra run directory, validated by a pydantic schema. It records: the resolved config, the artifact manifest hash, library versions, the model revision, and one entry per (mode, bit-width) with every metric above. A companion `generations.json` holds the qualitative outputs. Reference-model metrics are included in every file so that each file is self-contained.

### Evaluation configuration

```yaml
# configs/eval/default.yaml
chunk_len: 2048
max_chunks: null          # cap on chunks per dataset for smoke runs; null evaluates every chunk
datasets:
  wikitext2: {path: Salesforce/wikitext, name: wikitext-2-raw-v1, split: test, text_field: text, joiner: "\n\n"}
  c4: {path: allenai/c4, data_files: {validation: en/c4-validation.00000-of-00008.json.gz},
       split: validation, text_field: text, joiner: " ", max_tokens: 524288}
kl:
  dataset: wikitext2
  quantile: 0.99
lm_eval:                  # optional; when absent, Metric 3 is skipped
  tasks: [piqa, hellaswag, arc_easy]
  limit: null
generations:              # optional; when absent, Metric 4 is skipped
  prompts_file: configs/eval/prompts.yaml
  max_new_tokens: 128
bits: [3, 4, 5, 6, 7, 8]
```

A run with a non-null `max_chunks` is a smoke test. The value is recorded in `results.json`, and its numbers are not compared with full runs.

## Consequences

Results become comparable across milestones and carry their own provenance, and the primary metric is the quantity the method actually optimizes.

- Positive: KL and top-1 agreement are sensitive enough to separate adjacent bit-widths on a small model; perplexity and `lm-eval` keep comparability with the paper; bits-per-weight accounting prevents overstated memory savings.
- Negative: a full sweep (6 bit-widths × 2 modes × all metrics) is slow on a laptop GPU; `limit` and metric toggles mitigate this at the cost of noisier `lm-eval` numbers.
- Follow-up: pin the prompt set once the first baseline is recorded, since changing it invalidates qualitative comparisons; judged answer quality is specified separately in [ADR 0007](0007-llm-as-judge-evaluation.md), which uses its own larger prompt set.
