# ADR 0008: Batched and Per-Token Fisher Estimation

- Status: Accepted
- Date: 2026-09-29
- Deciders: Project maintainer

This ADR adds two opt-in features to the Fisher estimation of [ADR 0003](0003-fisher-weighted-kmeans-methodology.md). The first batches calibration sequences without mixing their gradients. The second replaces the per-sequence estimator with the per-token empirical Fisher, computed either from batches of prefixes or from batched backward passes. Both are selected in a new Hydra `fisher` config group, and the defaults reproduce ADR 0003's estimator exactly.

## Context

ADR 0003 estimates the Fisher diagonal by squaring the gradient of each sequence's mean token loss:

$$
\hat F^{\text{seq}}_{jj} = \sum_{i=1}^{N} \big(\partial_{\theta_j} \ell_i\big)^2,
\qquad \ell_i = \frac{1}{T-1}\sum_{t} -\log p_\theta\big(x^{(i)}_{t+1} \mid x^{(i)}_{\le t}\big).
$$

Two properties of that estimator motivate this record.

- **It is noisier than it needs to be.** Averaging token gradients before squaring adds within-sequence cross terms (ADR 0003, Section 1). Each entry then rests on $N = 100$ samples rather than about $N(T-1) = 51{,}100$. The Variance analysis section below quantifies the difference.
- **It leaves the GPU mostly idle.** One 512-token sequence per backward pass took 0.143 s, and the whole quantize stage peaked at 2.63 GiB of the laptop GPU's 6 GiB (wave 0 [results](../waves/wave_0/results.md)). A naive batched backward cannot use the spare memory, because the weight gradient $\Delta^\top A$ sums over sequences before anything is squared (ADR 0003, Section 1).

Two facts make both features possible:

- **Sequences are independent in the forward pass.** Granite-350M runs in eval mode with per-token RMSNorm, attention within each sequence and no mixture-of-experts routing. So a sequence's gradient can be separated from the others inside one batched pass, by contracting over tokens but not over the batch.
- **Attention is causal.** Position $t$'s loss depends only on tokens $\le t + 1$. So the gradient of one token's loss can be computed from a truncated prefix, or selected from a shared forward pass with a one-hot backward.

A probe on Granite (64 random tokens; gradients of layer 2 `shared_mlp.output_linear` and layer 14 `self_attn.q_proj`; errors measured against the largest gradient entry over all probed positions) established what the two token-mode methods can promise:

| Check | Device and dtype | Attention | Result | Detail |
| --- | --- | --- | --- | --- |
| Prefix batches equal full-sequence per-token gradients | CPU, float32 | sdpa, eager | pass | error at most 3.8e-6 |
| `is_grads_batched` equals one backward per token | CPU, float32 | sdpa, eager | pass | error at most 1.3e-6; 4 tokens in 0.52 s against 0.81–0.84 s |
| Both checks | CUDA, float32 | sdpa, eager | pass | error at most 4.6e-6; 4 tokens in 0.06 s against 0.19–0.20 s |
| Both checks | CUDA, bfloat16 | sdpa, eager | pass | batched against looped differs by 1.7%, prefix against full by 5.4–5.8%; no vmap fallback warnings |

Two lessons follow from the probe.

- **Tolerances depend on the dtype.** In bfloat16, two correct computations that reduce in a different order, or over different sequence lengths, differ by a few percent after 28 layers. That is still well below the sampling noise of the per-sequence estimator.
- **Some gradients are exactly zero.** The gradient of the first position's loss with respect to any `q_proj` is exactly zero, because position 0 attends to a single key and its attention weight is 1 whatever the query. `sdpa` returns about 2e-10 of rounding noise there. Errors must therefore be measured against the gradient's overall scale, never against a single position's magnitude.

## Decision

### Terminology

In this project a *row* always means a row of a weight matrix $W \in \mathbb{R}^{m \times n}$: one output channel, which gets its own codebook and its own weighted k-means problem (ADR 0003). This ADR never uses "row" for anything else. The entries along the leading dimension of an input batch are called **batch items**. A batch holds $Q$ items, indexed by $q = 1, \dots, Q$, with $Q \le$ `batch_size`. What an item contains depends on the mode: a distinct calibration sequence in sequence mode, or a copy of one sequence's prefix, whose loss is taken at a single position, in token mode.

| Symbol | Meaning |
| --- | --- |
| $Q$, $q$ | Batch items in one backward pass, and the index of one item |
| $T$ | Tokens per calibration sequence (512) |
| $m$, $n$ | Out and in features of one target linear, as in ADR 0003 |
| $A$ | Input activations of one target linear, $[Q, T, n]$ |
| $\Delta$ | Gradients reaching that linear's output, $[Q, T, m]$ |
| $G_q$ | Weight gradient of batch item $q$ alone, $[m, n]$ |

### Configuration

A new Hydra group, `configs/fisher/default.yaml`, is added to the defaults lists of both `quantize.yaml` and `evaluate.yaml`, and validated by a pydantic `FisherConfig`. Evaluation needs it because the Fisher key feeds every quantized key it recomputes.

```yaml
granularity: sequence          # sequence (ADR 0003) | token
token_method: prefix_batch     # token mode only: prefix_batch | batched_backward
positions_per_sequence: null   # token mode only: sample k of the T - 1 positions; null = all
batch_size: 1                  # sequences (sequence mode) or positions (token mode) per backward pass
split_half: false              # also measure split-half stability (see Diagnostics)
```

The pydantic model enforces the following:

- **`granularity`** is `Literal["sequence", "token"]`, and **`token_method`** is `Literal["prefix_batch", "batched_backward"]`.
- **`batch_size`** is a `PositiveInt`.
- **`positions_per_sequence`** is `PositiveInt | None`, at most `calibration.seq_len - 1`. It must be `None` in sequence mode, which the combined run config checks.
- **`token_method`** is ignored in sequence mode, and the manifest records it only in token mode.

**Cache key.** Only settings that change the estimator's mathematical value go into the Fisher key ([ADR 0002](0002-project-layout-and-architecture.md), caching table):

| Field | In the Fisher key | Reason |
| --- | --- | --- |
| `granularity` | yes | A different estimator |
| `positions_per_sequence` | yes | A different estimator (a random subsample) |
| `token_method` | no; recorded in the manifest | The same gradients up to rounding (probe above) |
| `batch_size` | no; recorded in the manifest | The same estimator; only speed and memory change |
| `split_half` | no; its results are recorded in the manifest | A diagnostic; the stored Fisher is the same sum, added in a different order |

`fisher_snapshot` gains a `fisher` block holding the two key fields, and the Fisher schema version is bumped, so every cached Fisher is recomputed once. Position sampling reuses `calibration.seed`, so no new seed field is needed.

**Default path.** `granularity: sequence` with `batch_size: 1` runs ADR 0003's existing loop (`register_post_accumulate_grad_hook`) unchanged. The default Fisher therefore keeps wave 0's numbers, up to the GPU non-determinism ADR 0003 already records.

### Shared machinery: per-item gradients from layer hooks

Every non-default setting computes one weight gradient per batch item from each target linear's input activations and output gradients, rather than reading `.grad`. For one linear layer $y = Wa$ and a batch of $Q$ items:

$$
G_q = \sum_{t} \delta_{q,t}\, a_{q,t}^\top
\quad\Longleftrightarrow\quad
G = \texttt{einsum("qtm,qtn->qmn", Δ, A)},
\qquad [Q, T, m] \times [Q, T, n] \to [Q, m, n].
$$

This is the same contraction autograd performs for $\nabla_W L$ (ADR 0003, Section 1), except that the item index $q$ is kept rather than summed. Each $G_q$ covers the whole matrix $W$, all $m$ of its rows; it is squared elementwise and then summed over items into the accumulator, which gives the estimators below.

1. **Capture activations.** A forward hook on each target linear saves its input $A$, shaped $[Q, T, n]$ and detached. It also registers a tensor hook on the layer's output. `q_proj`, `k_proj` and `v_proj` share one input tensor, so it is stored once.
2. **Receive output gradients.** During backward, the tensor hook receives $\Delta = \partial L / \partial y$, shaped $[Q, T, m]$. It computes `G` in float32, adds `G.square().sum(0)` into the module's float32 accumulator, and frees `G` before returning.
3. **Keep the reduced weight gradient from being built.** Target weights keep `requires_grad=False`, so autograd never forms the reduced $\Delta^\top A$. Gradients still flow because a forward hook on the input embedding marks its output `requires_grad_()`.
4. **Keep each item's loss separate.** The loss is the *sum* of the per-item losses, not Hugging Face's batch mean. Items are independent, so the gradient reaching item $q$'s output is exactly $\partial \ell_q / \partial y_q$.
5. **Clean up.** Hooks are removed, and `requires_grad` flags restored, in a `finally` block, as in ADR 0003's `estimate_fisher`.

The largest per-item tensor is `input_linear`'s $G_q$ ($4096 \times 1024$ float32, 16 MB). Saved activations add about 150 MB per 512-token item across the 168 modules in bfloat16.

### Sequence mode with `batch_size > 1`

Each batch item is a distinct calibration sequence, and its loss is that sequence's mean token NLL $\ell_i$. The estimator is therefore exactly ADR 0003's $\hat F^{\text{seq}}$, with `batch_size` sequences per backward pass.

```python
def estimate_fisher_batched(model, calibration, targets, batch_size):
    # calibration: [N, T] int64; one backward per block of Q <= batch_size sequences.
    with item_gradient_hooks(model, targets) as fisher:          # name -> float32 [m, n]
        for block in calibration.split(batch_size):              # [Q, T]
            logits = model(input_ids=block).logits               # [Q, T, V]
            nll = token_nll(logits, block)                       # [Q, T - 1]
            # Sum of per-item means: item q's output gradient is d(l_q)/dy_q.
            nll.mean(dim=1).sum().backward()
    return fisher
```

Memory grows mainly with the full-vocabulary logits. The float32 loss needs $[Q, T, V]$ logits (205 MB per 512-token item) plus the softmax saved for backward. With the rest of the graph, one 512-token item needs about 0.9 GB, so `batch_size` of about 4 fits in 6 GiB, or 2 with `split_half`. Applying the head in slices under activation checkpointing would lift that limit, but it is out of scope here.

### Token mode

Token mode estimates the per-token empirical Fisher:

$$
\hat F^{\text{tok}}_{jj} = \sum_{i=1}^{N} \sum_{t \in P_i} \big(\partial_{\theta_j} \ell_{i,t}\big)^2,
\qquad \ell_{i,t} = -\log p_\theta\big(x^{(i)}_{t+1} \mid x^{(i)}_{\le t}\big),
$$

where $P_i$ is the set of evaluated positions of sequence $i$: all $T - 1$, or a random subsample. Both methods run the same two loops. The outer loop walks over calibration sequences. The inner loop walks over blocks of `batch_size` positions, squares each position's gradient, and adds it into the accumulator. They differ in how they obtain one output gradient per position.

**`prefix_batch`.**

- A block of positions $t_1 \lt \dots \lt t_Q$ becomes a batch of $Q$ items. Every item is a copy of the same tokens $x_{\le t_Q}$, cut to the block's longest prefix. Item $q$'s loss is $\ell_{t_q}$ alone, read from the hidden state at $t_q$ against the target $x_{t_q + 1}$.
- Causal attention makes the tokens after $t_q$ irrelevant to item $q$'s loss. They receive zero gradient, so items need no padding or attention mask.
- The batch dimension exists only so the shared hooks can keep each position's gradient separate. The forward pass is recomputed once per item.
- The LM head is applied only at each item's own position, $[Q, H] \to [Q, V]$. That removes the $[Q, T, V]$ logits tensor, which is what limits batching in sequence mode. Each item's graph still grows with its prefix, to about 0.5 GB at the full 512 tokens, so blocks of long prefixes fit about 4 to 6 items beside the accumulators.
- Blocks are formed from consecutive positions, so early blocks run on short prefixes. The total work per sequence is about $\sum_t (t + 1) \approx T^2 / 2 \approx 131{,}000$ token positions, about 256 times the sequence mode.

```python
def token_fisher_prefix_batch(model, sequence, positions, batch_size):
    # sequence: [T] int64; positions: sorted target positions, each in [0, T - 2].
    for block in positions.split(batch_size):                    # [Q] positions
        length = int(block[-1]) + 1
        items = sequence[:length].expand(len(block), length)     # [Q, L], identical items
        hidden = body_hidden_states_batched(model, items)        # [Q, L, H]
        picked = hidden[torch.arange(len(block)), block]         # [Q, H], one position per item
        logits = logit_head(model)(picked).float()               # [Q, V]
        next_tokens = sequence[block + 1]                        # [Q]
        # Sum over items: item q's output gradient is the gradient of token t_q's loss alone.
        nll(logits, next_tokens).sum().backward()                # hooks square and sum [Q, m, n]
```

**`batched_backward`.**

- One forward pass per sequence keeps each target's output $Y$ (shape $[1, T, m]$) and input $A$ (shape $[1, T, n]$) in the graph, and computes the per-position losses $\ell \in \mathbb{R}^{T-1}$. As in step 3 of the shared machinery, target weights stay frozen, and the embedding hook gives the outputs a gradient path.
- For each block of $Q$ positions, a one-hot matrix $E \in \lbrace 0, 1\rbrace^{Q \times (T-1)}$ selects one loss per batch item. `torch.autograd.grad(ℓ, Ys, grad_outputs=E, is_grads_batched=True, retain_graph=True)` then returns $\Delta$ for every module at once, shaped $[Q, 1, T, m]$. Here the batch items exist only in the backward pass: they share one forward pass and differ only in which loss they select.
- Gradients are taken with respect to layer *outputs*, not weights. $\Delta$ for all 168 modules is about 220 MB per position in bfloat16 ($28 \times 7{,}680$ output features $\times\, 512$ tokens), against about 0.5 GB per position for weight gradients. The per-module $G$ is formed from $\Delta$ and the shared $A$ as below.
- The forward pass is shared by every position. The work per sequence is one forward pass plus $T - 1$ backward passes, vectorized $Q$ at a time. On the GPU, batching 4 positions ran about 3.2 times faster than looping.

```python
def token_fisher_batched_backward(model, sequence, positions, batch_size, targets, fisher):
    # sequence: [T] int64. Forward hooks keep each target's output Y [1, T, m] and input A [1, T, n].
    with output_capture(model, targets) as captured:            # name -> (Y, A)
        logits = model(input_ids=sequence[None]).logits[0].float()   # [T, V]
        nll = token_nll(logits, sequence)                            # [T - 1]
        for block in positions.split(batch_size):                    # [Q] positions
            onehots = torch.zeros(len(block), nll.shape[0])          # [Q, T - 1]
            onehots[torch.arange(len(block)), block] = 1.0
            deltas = torch.autograd.grad(
                nll, [y for y, _ in captured.values()], grad_outputs=onehots,
                is_grads_batched=True, retain_graph=True,
            )                                                        # each [Q, 1, T, m]
            for (name, (_, a)), delta in zip(captured.items(), deltas):
                # [Q, T, m] x [T, n] -> [Q, m, n]: one gradient per position, then square and sum.
                g = torch.einsum("qtm,tn->qmn", delta[:, 0].float(), a[0].float())
                fisher[name] += g.square().sum(0)
```

**Start-up capability check for `batched_backward`.**

1. On the first calibration sequence, compute per-token gradients at three positions both ways: batched, and one `torch.autograd.grad` per position.
2. Measure the largest difference against the largest gradient entry over those positions. The tolerance is `1e-4` in float32 and `1e-1` in bfloat16; the probe measured 5e-6 and 1.7%.
3. Run the batched call under `warnings.catch_warnings()`, with PyTorch's vmap slow-path warning escalated to an error.
4. On a mismatch or a fallback, raise a `FisherCapabilityError` that names `fisher.token_method=prefix_batch`. There is no silent fallback: the user chooses the slower method knowingly.

**`positions_per_sequence = k`.**

- Each sequence draws $k$ of its $T - 1$ positions uniformly without replacement. The generator is seeded from `stable_seed(calibration.seed, f"fisher_positions/{i}")`, so the draw is reproducible and determined by fields already in the key.
- Every position is included with probability $k / (T - 1)$, so multiplying the sequence's sum by $(T - 1) / k$ keeps it an unbiased estimate of the all-positions sum.

**Stored scale and diagnostics.** The two granularities store diagonals of different magnitude. Sequence mode stores squared gradients of the *mean* loss; in expectation that is $1/(T-1)^2$ of token mode's sum when the cross terms average to zero. Neither the k-means minimizer nor the Fisher-weighted relative error depends on a constant factor, so no rescaling is applied, and the manifest records `granularity` so readers know which one they have. The per-sequence `losses` diagnostic in token mode is the mean NLL over the evaluated positions. With all positions it equals $\ell_i$, and the acceptance check on the Fisher mean loss (spec 0011) applies only then.

### Diagnostics: split-half stability

The variance analysis below rests on idealized assumptions. Split-half stability measures the estimator's noise on the model's real gradients instead, at no extra compute.

1. **Two accumulators.** With `split_half: true`, every estimator adds odd-numbered calibration sequences into one float32 accumulator and even-numbered ones into another. In batched paths, each batch item is routed by the parity of the sequence it came from. The stored Fisher is the sum of the two halves, so it is the same estimate as without the flag, up to float32 rounding in the order of addition.
2. **Compare the halves.** For each module, the two flattened $[m, n]$ half-diagonals are compared with Spearman's rank correlation $r_{50}$, with tied values given their average rank, as `scipy.stats.spearmanr` does. Ranks are used because weighted k-means depends only on the relative sizes of the sensitivities, and because the entries span orders of magnitude, so a Pearson correlation would be dominated by the few largest.
3. **Project to the full size.** Each half holds 50 sequences. The Spearman–Brown formula, $r_{100} = 2 r_{50} / (1 + r_{50})$, estimates the reliability of the full 100-sequence estimate. The projection is meaningful only for $r_{50} > 0$, and is left empty otherwise.
4. **Record, then discard.** The manifest stores $r_{50}$ and $r_{100}$ per module. The half-diagonals themselves are not saved.

The cost is one extra float32 accumulator, 1 GB of GPU memory for Granite's 249.6M target entries. That is why the flag is off by default.

### Variance analysis

This section derives the coefficients of variation quoted above and in ADR 0003. The model is deliberately simple, a best case, and the experiment below measures the real gap.

**Assumptions.** Fix one weight $j$ and write $g_t = \partial_{\theta_j} \ell_{i,t}$. Assume the $g_t$ are independent across tokens and sequences, with $g_t \sim \mathcal{N}(0, \sigma^2)$.

**One chi-square fact.** If $Z \sim \mathcal{N}(0, 1)$, then $Z^2 \sim \chi^2_1$, with $\mathbb{E}[Z^2] = 1$ and $\mathbb{E}[Z^4] = 3$, so $\operatorname{Var}(Z^2) = 3 - 1 = 2$. A $\chi^2_1$ variable therefore has a coefficient of variation (standard deviation over mean) of $\sqrt{2} \approx 1.41$.

**Sequence mode.** Up to the dropped constant, each sequence contributes the square of the sum $S = \sum_{t=1}^{T-1} g_t$.

1. The sum is Gaussian: $S \sim \mathcal{N}\big(0, (T-1)\sigma^2\big)$.
2. So $S^2 / \big((T-1)\sigma^2\big) \sim \chi^2_1$, and one sequence's contribution has mean $(T-1)\sigma^2$ and a coefficient of variation of $\sqrt{2}$, about 141%.
3. Averaging $N$ independent contributions divides the standard deviation by $\sqrt{N}$:

$$
\mathrm{CV}_{\text{seq}} = \sqrt{2 / N} = 14.1\% \quad (N = 100).
$$

**Token mode.** The estimator sums $N(T-1)$ independent terms $g_t^2 = \sigma^2 Z_t^2$, each with mean $\sigma^2$ and variance $2\sigma^4$:

$$
\mathrm{CV}_{\text{tok}} = \sqrt{\frac{2}{N(T-1)}} = 0.63\% \quad (N(T-1) = 51{,}100),
\qquad
\mathrm{CV}_{\text{tok}, k} = \sqrt{\frac{2}{N k}} = 1.8\% \quad (k = 64).
$$

**Beyond the Gaussian assumption.**

- **Heavy tails.** For any zero-mean $g$ with variance $\sigma^2$ and kurtosis $\kappa = \mathbb{E}[g^4]/\sigma^4$, $\operatorname{Var}(g^2) = (\kappa - 1)\sigma^4$. The Gaussian case is $\kappa = 3$, which gives the factor 2 above. Token mode's coefficient of variation becomes $\sqrt{(\kappa - 1)/(N(T-1))}$. Sequence mode is less affected: $S$ sums 511 terms and is close to Gaussian by the central limit theorem, so its $\sqrt{2/N}$ stays roughly right. Even at $\kappa = 30$, token mode's figure is $\sqrt{29/51{,}100} \approx 2.4\%$.
- **Correlated tokens.** Positive correlation between the squared terms of nearby tokens lowers token mode's effective count below $N(T-1)$. Correlation between $g_t$ and $g_s$ adds the bias to sequence mode described in ADR 0003.
- **Therefore** the figures are bounds on precision under ideal conditions, not predictions.

### Cost and memory

The estimates below are scaled from the wave 0 Fisher pass: 14.3 s for 100 sequences on an RTX 3060 Laptop GPU, with the whole quantize stage peaking at 2.63 GiB.

| Setting | Work per sequence | Estimated time, 100 sequences | Main extra memory |
| --- | --- | --- | --- |
| `sequence`, `batch_size: 1` | 1 forward and 1 backward | 14 s | none |
| `sequence`, `batch_size: 4` | one batched pass per 4 sequences | under 14 s | $[Q, T, V]$ logits and saved activations, about 4× |
| `token`, `prefix_batch` | about $T^2/2$ token positions (256× the sequence mode) | about 1 h | activations of `batch_size` prefixes |
| `token`, `batched_backward` | 1 forward and $T - 1$ backward passes, batched, plus one per-position weight-gradient contraction over the position's prefix | about 30–75 min, mostly the contraction (spec 0014) | about 220 MB of $\Delta$ and 205 MB of logit gradients per position in the block |
| `token`, either method, `positions_per_sequence: 64` | about 8× less | under 10 min | as above |

The GPU in this project throttles under sustained load (wave 0 [results](../waves/wave_0/results.md)), so long token-mode runs may take longer. Their spec should confirm the times with a short timing run before relying on them.

### Testing

The offline suite ([ADR 0001](0001-code-maintainability.md)) uses the tiny test model and pins every equivalence:

- **Batching sequences.** Sequence mode with `batch_size` 1 and 4 gives the same Fisher to float32 tolerance.
- **Token-mode methods.** `prefix_batch`, `batched_backward` and one `torch.autograd.grad` per token agree, with errors measured against the overall gradient scale.
- **An exact oracle.** For every attention layer, the `q_proj` and `k_proj` gradients of $\ell_{i,0}$ are zero, because position 0 attends to itself alone.
- **Unbiased subsampling.** Averaging the subsampled estimate over many seeds converges to the all-positions sum within a stated tolerance.
- **Cache key.** The Fisher key changes with `granularity` and `positions_per_sequence`, and does not change with `batch_size`, `token_method` or `split_half`.
- **Split halves.** The two halves sum to the Fisher computed without the flag, and two identical halves give a correlation of 1.
- **The capability check.** It raises `FisherCapabilityError` when the batched gradients are corrupted by a monkeypatched `torch.autograd.grad`.
- **Cleanup.** Hooks are removed and `requires_grad` flags restored after a forward or backward pass raises.

A GPU integration test repeats the probe above on Granite. It checks both token methods, in float32 and bfloat16, with the dtype-dependent tolerances.

### Evaluation

The features exist to answer one question: does a less noisy Fisher produce a better quantized model? The spec implementing this ADR runs a comparison with everything else fixed (calibration set, quantizer, seeds):

| Fisher | Why it is included |
| --- | --- |
| `sequence` (ADR 0003 default) | Baseline |
| `token`, all positions | The low-variance estimator |
| `token`, `positions_per_sequence: 64` | Whether a cheap subsample keeps most of the gain |

**What each Fisher is measured by:**
- mean KL and top-1 agreement at 3, 4 and 5 bits, where quantization error dominates (ADR 0004);
- the Spearman rank correlation of each module's diagonal against the token-mode Fisher;
- split-half stability (the Diagnostics section above). The chi-square model predicts that token mode's halves agree far more closely than sequence mode's.

## Consequences

The Fisher estimator becomes configurable, and the lower-variance per-token estimator becomes available at a known cost, while the default stays ADR 0003's.

- **Positive:**
  - The per-token empirical Fisher is available, and it reduces the estimator's sampling noise from about 14% to under 1% per entry under the model above.
  - Batching lets both modes use the GPU memory that the one-sequence loop leaves idle.
  - Both token methods are pinned against a one-backward-per-token reference.
  - An unsupported environment fails loudly, rather than producing a subtly wrong Fisher.
- **Negative:**
  - Token mode costs about 30 to 250 times the sequence mode's compute, depending on method, batching and subsampling.
  - The layer-hook path is more code to verify than the post-accumulate hook, and it depends on hook ordering in the backward pass.
  - Bumping the Fisher schema version invalidates the cached Fisher once.
  - Estimates differ between granularities by a constant scale, so diagonals from different granularities must not be mixed.
- **Out of scope:**
  - Sampled labels (the Monte Carlo estimate of the true Fisher). This is a separate, orthogonal flag, listed as future work in ADR 0003.
  - Checkpointed, sliced-head backward passes for larger sequence-mode batches.
  - Per-position approximations that split the sequence gradient by where $W$ was applied rather than by loss term.
