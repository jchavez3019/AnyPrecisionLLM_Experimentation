# ADR 0003: Methodology — Fisher-Weighted k-Means with Incremental Upscaling

- Status: Proposed
- Date: 2026-09-24
- Deciders: Project maintainer

This ADR specifies the quantization method: the objective (a diagonal-Fisher-weighted k-means per weight row), why that objective is the right local proxy for answer quality, how the seed model and each incremental upscaling step are solved exactly or near-exactly in 1D, and how this is implemented in batched PyTorch. It follows Park et al., "Any-Precision LLM" (ICML 2024), whose backbone is SqueezeLLM (Kim et al., 2023).

## Context

Any-precision quantization stores one $B$-bit *parent* model from which every lower bit-width $b \in \lbrace b_0, \dots, B\rbrace$ is obtained by keeping only the top $b$ bits of each weight's index. The paper builds the parent in two stages: quantize to the lowest bit-width $b_0$ (the *seed*), then *upscale* one bit at a time. For this to work, the quantizer must admit a refinement step in which every bin splits into two children without moving any existing boundary.

The paper shows (its Appendix E) that uniform methods do not refine well: GPTQ's column-wise error compensation diverges between bit-widths and forces runaway clamping, and AWQ's scales are tuned for a single bit-width. Clustering-based non-uniform quantization refines naturally, because each cluster can be split at an arbitrary point.

The reference implementation (`any-precision-llm/any_precision/quantization/`) uses Numba and `flash1dkmeans` on the CPU. This project reimplements the same method in PyTorch, batched across rows, on the GPU.

## Decision

### Notation

| Symbol                                      | Meaning |
|---------------------------------------------| --- |
| $\theta$                                  | All model parameters |
| $W \in \mathbb{R}^{m \times n}$           | Weight of one quantizable linear layer; $y = Wx$, $m$ = out features, $n$ = in features |
| $w \in \mathbb{R}^{n}$                    | One row of $W$ (one output channel); rows are quantized independently |
| $(b_0, B)$                                  | Seed and parent bit-widths (defaults 3 and 8) |
| $K_b = 2^b$                               | Number of centroids at bit-width $b$ |
| $C^{(b)} \in \mathbb{R}^{K_b}$            | Codebook (lookup table) of one row at bit-width $b$ |
| $a^{(b)}(j) \in \lbrace 0,\dots,K_b-1\rbrace$        | Index assigned to weight $j$ at bit-width $b$ |
| $f_j \ge 0$                               | Sensitivity weight of weight $j$ (a diagonal Fisher entry) |
| $\mathcal{D} = \lbrace x^{(1)},\dots,x^{(N)}\rbrace$ | Calibration sequences (defaults $N=100$, length $T=512$) |

### 1. The objective

**What we want.** A quantized model should predict the same next-token distributions as the original. Write the quantized parameters as $\theta + \delta$. The natural loss is the expected KL divergence


$$
\mathcal{L}(\delta) = \mathbb{E}_{x}\,\sum_t \mathrm{KL}\left(p_\theta(\cdot \mid x_{\lt t}) \,\big\Vert \, p_{\theta+\delta}(\cdot \mid x_{\lt t})\right).
$$

**Second-order expansion.** $\mathcal{L}(0) = 0$ is a minimum, so the gradient vanishes there and

$$
\mathcal{L}(\delta) \approx \tfrac{1}{2}\,\delta^\top F\,\delta,
\qquad
F = \mathbb{E}_{x}\,\sum_t\,\mathbb{E}_{y \sim p_\theta(\cdot\mid x_{\lt t})}\left[\nabla_\theta \log p_\theta(y \mid x_{\lt t})\,\nabla_\theta \log p_\theta(y \mid x_{\lt t})^\top\right],
$$

where $F$ is the (true) Fisher information matrix. This is the precise sense in which the method optimizes answer quality: it minimizes a local approximation of the output-distribution shift, which is also the KL metric evaluated in [ADR 0004](0004-evaluation-protocol.md).

**Approximation 1: empirical Fisher.** Instead of sampling $y$ from the model, use the observed next tokens and per-sequence gradients. With $\ell_i(\theta) = \frac{1}{T-1}\sum_{t} -\log p_\theta(x^{(i)}_t \mid x^{(i)}_{\lt t})$, the mean token negative log-likelihood of sequence $i$ (what `model(input_ids=x, labels=x).loss` returns),

$$
\hat F = \sum_{i=1}^{N} \nabla_\theta \ell_i\, \nabla_\theta \ell_i^\top.
$$

This matches the reference implementation. It differs from $F$ by using data labels rather than model samples and by averaging token gradients within a sequence before squaring. Positive constant factors ($\tfrac12$, $1/N$, $1/(T-1)^2$) do not change any minimizer below and are dropped.

**Approximation 2: diagonal.** Keep only the diagonal, $\delta^\top \hat F \delta \approx \sum_i \hat F_{ii}\,\delta_i^2$. This discards all interactions between weights; methods that keep them (GPTQ, via $XX^\top$) compensate errors across columns, which is exactly what makes them incompatible with upscaling.

**Approximation 3: row-wise codebooks.** Each row of each layer has its own codebook, so the objective separates into one independent problem per row. For a row $w$ with sensitivities $f_j = \hat F_{jj}$:

$$
\boxed{\;J\left(C, a\right) = \sum_{j=1}^{n} f_j \left(w_j - C_{a(j)}\right)^2\;}
$$

This is **weighted k-means in one dimension** with $K_b$ clusters.

**Sensitivity conditioning.** Following the reference implementation, weights that are exactly zero receive $f_j = 0$. If a row's sensitivities sum to zero, the row falls back to $f_j = 1$ for all $j$ (unweighted k-means). Fisher entries are accumulated in float32; the reference stores them in bfloat16, where small squared gradients can underflow.

### 2. Structure of the 1D problem

**Lemma (contiguity).** For fixed centroids, each term of $J$ is minimized independently by assigning $w_j$ to its nearest centroid, since $f_j \ge 0$. In $\mathbb{R}$, the set of points nearest to a given centroid is an interval with endpoints at midpoints between adjacent centroids. Therefore an optimal clustering, after sorting $w$ ascending, is a partition into $K$ *contiguous* segments, and an optimal centroid for a segment is its weighted mean.

**Prefix-sum identities.** Sort the row: $w_{(1)} \le \dots \le w_{(n)}$, carrying $f$ along. Define, for $k = 0,\dots,n$,

$$
P^0_k = \sum_{i \le k} f_{(i)}, \qquad P^1_k = \sum_{i \le k} f_{(i)} w_{(i)}, \qquad P^2_k = \sum_{i \le k} f_{(i)} w_{(i)}^2,
$$

with $P_0 = 0$. For the half-open segment $[s, e)$ of sorted positions, let $M = P^0_e - P^0_s$, $S = P^1_e - P^1_s$, and $Q = P^2_e - P^2_s$. Then

$$
\mu(s,e) = \frac{S}{M}, \qquad \mathrm{cost}(s,e) = \sum_{i=s+1}^{e} f_{(i)}\big(w_{(i)} - \mu\big)^2 = Q - \frac{S^2}{M}.
$$

Any segment's centroid and cost are available in $O(1)$. The subtraction $Q - S^2/M$ cancels catastrophically in float32, so prefix sums and costs are computed in **float64**. If $M \le \varepsilon$, the segment is treated as empty: its cost is 0 and its centroid is inherited (see below).

### 3. Seed: weighted Lloyd's algorithm at $b_0$ bits

Solve $\min J$ with $K = 2^{b_0}$ using Lloyd's algorithm on the sorted row:

1. **Initialize** with weighted greedy k-means++ (below).
2. **Assign**: the interior segment borders are the positions of the midpoints $(C_k + C_{k+1})/2$ in the sorted row, found by binary search.
3. **Update**: $C_k \leftarrow \mu(s_k, e_k)$ for non-empty segments; an empty segment keeps its previous centroid, and the codebook is re-sorted so borders stay monotone.
4. Repeat until the borders stop changing or a fixed iteration cap is reached (default 50, matching the reference).

Each iteration costs $O(K \log n)$ per row after the one-time $O(n \log n)$ sort. $J$ is non-increasing and the algorithm terminates, but it finds a local, not guaranteed global, optimum. An exact dynamic-programming solution ($O(K n^2)$, or faster with monotone-matrix tricks) was considered and rejected as unnecessary complexity for this reproduction.

**Initialization: weighted greedy k-means++.** This is the initialization of `flash1dkmeans`, which the reference implementation uses, and of scikit-learn. Let $D(j) = \min_{c \in C} (w_j - c)^2$ be the squared distance from weight $j$ to the nearest centroid chosen so far.

1. Draw the first centroid $w_j$ with probability proportional to $f_j$.
2. For each of the remaining $K - 1$ centroids, draw $L = 2 + \lfloor \ln K \rfloor$ candidates independently, each with probability proportional to $f_j\,D(j)$. Keep the candidate that yields the smallest potential $\sum_j f_j\,D(j)$ after it is added.
3. Sort the $K$ chosen values.

Sampling is driven by a seeded `torch.Generator`, so codebooks are reproducible for a fixed seed and device type. Seeds are derived hierarchically from `quantizer.seed` (default 0). Each layer's seed is a hash of `quantizer.seed` and the layer's module name, and each chunk of rows gets its seed as a hash of the layer seed and the chunk index. Every Lloyd fit starts a fresh generator from its chunk seed. This has three consequences:

- A layer's codebooks do not depend on the order in which layers are processed.
- In standalone mode, the fits at different bit-widths do not depend on each other.
- The standalone fit at $b_0$ is bitwise identical to the incremental seed (Section 4, property 4).

Because a weight equal to an existing centroid has $D(j) = 0$, it can never be drawn again, so no two centroids start at the same value unless the row has fewer than $K$ distinct values. Batched over $R$ rows, one step costs $O(L\,n)$ per row, and initialization costs $O(K L n)$ per row, which is negligible next to the Fisher pass.

**Why not a quantile initialization.** A deterministic alternative places $C_k$ at the weighted quantile $(k + \tfrac12)/K$ of $P^0$. It fails on Granite because the empirical Fisher is extremely concentrated: in attention rows, the top 1% of entries carry a median 77–91% of the row's mass. Quantiles of such a distribution fall on the same few weights, so many centroids start at identical values, and Lloyd can never separate them. Quantiles of the unweighted (count) distribution avoid that collapse, but they ignore where the sensitivity lives. The comparison below was measured on layer 14 `q_proj` and layer 2 `output_linear` in [notebooks/02_Fisher_and_KMeans.ipynb](../../notebooks/02_Fisher_and_KMeans.ipynb), section 10. Errors are Fisher-weighted and relative to k-means++.

| Initialization | Empty clusters at 8 bits | 3-bit seed error | 8-bit standalone error |
| --- | --- | --- | --- |
| Weighted k-means++ | 0% / 0% | 1× / 1× | 1× / 1× |
| Fisher-mass quantiles | 68% / 89% | 2.6× / 1.2× | 410× / 940× |
| Count quantiles | 0.1% / 0% | 2.0× / 1.1× | 270× / 33× |

### 4. Incremental upscaling: exact 2-way splits

Given the segments $\lbrace [s_k, e_k)\rbrace_{k=0}^{K_b-1}$ of bit-width $b$, produce bit-width $b+1$ by splitting every segment into two:

$$
m_k^\star = \arg\min_{s_k \le m \le e_k} \Big[\mathrm{cost}(s_k, m) + \mathrm{cost}(m, e_k)\Big],
\qquad
\text{children:} 2k \leftarrow [s_k, m_k^\star),\;\; 2k+1 \leftarrow [m_k^\star, e_k).
$$

Scanning all split points costs $O(e_k - s_k)$ per segment, so one upscaling step is $O(n)$ per row, and the solution is **exact** for this subproblem. A segment with fewer than two points, or with zero total sensitivity, is not split; both children inherit the parent centroid.

**Properties** (each one is a test target):

1. **Monotone improvement.** $J^{(b+1)} \le J^{(b)}$, because $m = s_k$ reproduces the parent cost.
2. **Nested indices.** The left child of $k$ is $2k$ and the right child is $2k+1$, so the new bit is appended as the least significant bit, and $a^{(b)}(j) = a^{(B)}(j) \gg (B - b)$ for every $b_0 \le b \le B$.
3. **Sorted codebooks.** Each $C^{(b)}$ is ascending, because segments are contiguous in sorted order.
4. **Exact seed, constrained upscales.** The seed is exactly what standalone weighted k-means produces at $b_0$ bits. For $b \gt b_0$, $J_{\text{IU}}^{(b)} \ge J^{\star(b)}$, since the upscaled model must keep all coarser borders. The paper measures this gap as under 0.1 perplexity in most cases (its Table 3).

### 5. Standalone baseline

To reproduce the paper's central comparison, the quantizer supports `mode: standalone`, which runs the seed procedure (Section 3) independently at every bit-width in $\lbrace b_0,\dots,B\rbrace$. Standalone artifacts are not nested and are used only for evaluation. The default `mode: incremental` is the any-precision method.

### 6. Pseudo-code

Shapes are annotated as `[rows, n]`; all functions process a chunk of rows of one layer at a time to bound memory.

**Empirical Fisher accumulation** (one pass over calibration data):

```python
def estimate_fisher(model, calibration_tokens, target_modules) -> dict[str, Tensor]:
    # Only quantizable weights need parameter gradients; activations still carry gradients.
    for p in model.parameters():
        p.requires_grad_(False)
    fisher = {}
    for name, linear in target_modules.items():
        W = linear.weight
        W.requires_grad_(True)
        fisher[name] = torch.zeros_like(W, dtype=torch.float32)       # [m, n]

        # Square the per-sequence gradient and discard .grad immediately to save memory.
        def hook(p, name=name):
            fisher[name].add_(p.grad.float().square())
            p.grad = None
        W.register_post_accumulate_grad_hook(hook)

    model.eval()
    for tokens in calibration_tokens:                                  # each [1, T]
        loss = model(input_ids=tokens, labels=tokens).loss             # mean token NLL
        loss.backward()
    return fisher
```

**Row preparation** (per layer, per chunk of rows):

```python
def prepare_rows(W, F):                                    # W, F: [R, n]
    f = F * (W != 0)                                       # zero weights carry no sensitivity
    f = torch.where(f.sum(1, keepdim=True) > 0, f, torch.ones_like(f))
    order = W.argsort(dim=1)                               # [R, n]
    w_s, f_s = W.gather(1, order), f.gather(1, order)      # sorted values and weights
    P0 = pad0(f_s.double().cumsum(1))                      # [R, n+1]
    P1 = pad0((f_s * w_s).double().cumsum(1))              # [R, n+1]
    P2 = pad0((f_s * w_s * w_s).double().cumsum(1))        # [R, n+1]
    return order, w_s, f_s, (P0, P1, P2)
```

**Segment statistics** (batched gathers):

```python
def segment_stats(P, s, e):                                # s, e: [R, K] integer borders
    M = P0.gather(1, e) - P0.gather(1, s)
    S = P1.gather(1, e) - P1.gather(1, s)
    Q = P2.gather(1, e) - P2.gather(1, s)
    mean = S / M.clamp_min(eps)
    cost = torch.where(M > eps, Q - S * S / M.clamp_min(eps), 0.0)
    return M, mean, cost                                   # each [R, K]
```

**Seed initialization: batched weighted greedy k-means++:**

```python
def weighted_kmeanspp_init(w_s, f_s, K, generator):          # w_s, f_s: [R, n]
    L = 2 + floor(log(K))                                    # local trials per centroid
    first = torch.multinomial(f_s, 1, generator=generator)   # [R, 1] index drawn by f
    C = [w_s.gather(1, first)]
    D = (w_s - C[0]).square()                                # [R, n] distance to nearest centroid
    for _ in range(1, K):
        cand = torch.multinomial(f_s * D, L, replacement=True, generator=generator)   # [R, L]
        c = w_s.gather(1, cand)                                                        # [R, L]
        D_try = torch.minimum(D[:, None, :], (w_s[:, None, :] - c[:, :, None]).square())  # [R, L, n]
        best = (f_s[:, None, :] * D_try).sum(-1).argmin(1, keepdim=True)            # [R, 1]
        C.append(c.gather(1, best))
        D = D_try.gather(1, best[:, :, None].expand(-1, 1, n)).squeeze(1)
    return torch.cat(C, 1).sort(dim=1).values                                        # [R, K]
```

**Seed: batched weighted Lloyd:**

```python
def weighted_lloyd(w_s, f_s, P, K, generator, max_iter=50):
    C = weighted_kmeanspp_init(w_s, f_s, K, generator)       # [R, K]
    borders = None
    for _ in range(max_iter):
        mid = 0.5 * (C[:, :-1] + C[:, 1:])                 # [R, K-1]
        inner = torch.searchsorted(w_s, mid)               # [R, K-1] positions in sorted row
        new_borders = cat([zeros(R, 1), inner, full(R, 1, n)], dim=1)   # [R, K+1]
        if borders is not None and torch.equal(new_borders, borders):
            break
        borders = new_borders
        M, mean, _ = segment_stats(P, borders[:, :-1], borders[:, 1:])
        C = torch.where(M > eps, mean, C).sort(dim=1).values
    return C, borders
```

**Upscale: batched exact split of every segment:**

```python
def split_all_segments(P, borders, C):                     # borders: [R, K+1], C: [R, K]
    m = arange(1, n)                                       # candidate interior splits, [n-1]
    k = torch.searchsorted(borders[:, 1:], m.expand(R, -1), right=True)   # owning segment, [R, n-1]
    s, e = borders.gather(1, k), borders.gather(1, k + 1)                 # [R, n-1]
    total = cost(P, s, m) + cost(P, m, e)                  # [R, n-1], via segment_stats
    total = torch.where((m > s) & (m < e), total, inf)     # a split must leave both sides non-empty

    # Segmented argmin: minimum per segment, then the first position attaining it.
    best = full(R, K, inf).scatter_reduce(1, k, total, "amin")
    hit = total == best.gather(1, k)
    m_star = full(R, K, n).scatter_reduce(1, k, where(hit, m, n), "amin")
    m_star = torch.where(best.isfinite(), m_star, borders[:, :-1])        # unsplittable -> left empty

    child_borders = interleave(borders[:, :-1], m_star) + [n]            # [R, 2K+1]
    M, mean, _ = segment_stats(P, child_borders[:, :-1], child_borders[:, 1:])
    parent_C = C.repeat_interleave(2, dim=1)                              # [R, 2K]
    child_C = torch.where(M > eps, mean, parent_C)
    return child_C, child_borders
```

**Per-layer driver:**

```python
def quantize_layer(W, F, seed_bits, parent_bits, layer_seed, row_chunk):   # incremental mode
    for c, rows in enumerate(chunks(range(W.shape[0]), row_chunk)):        # bounds peak memory
        order, w_s, f_s, P = prepare_rows(W[rows].float(), F[rows])
        generator = torch.Generator(device=W.device).manual_seed(stable_seed(layer_seed, f"chunk{c}"))
        C, borders = weighted_lloyd(w_s, f_s, P, 2 ** seed_bits, generator)
        luts = {seed_bits: C}
        for b in range(seed_bits, parent_bits):
            C, borders = split_all_segments(P, borders, C)
            luts[b + 1] = C                                # [R, 2**(b+1)]

        # Segment id of each sorted position at the parent bit-width, then undo the sort.
        idx_sorted = segment_ids_from_borders(borders, n)  # [R, n], values in [0, 2**parent_bits)
        idx = torch.empty_like(idx_sorted).scatter_(1, order, idx_sorted)
        store(rows, idx.to(torch.uint8), {b: lut.to(torch.float16) for b, lut in luts.items()})

def quantize_model(targets, fisher, cfg):
    for name, linear in targets.items():
        quantize_layer(linear.weight, fisher[name], cfg.seed_bits, cfg.parent_bits,
                       stable_seed(cfg.seed, name), cfg.row_chunk)
```

`stable_seed(seed, name)` is the first 8 bytes of `sha256(f"{seed}:{name}")`, masked to 63 bits. In standalone mode, each bit-width's Lloyd fit gets its own fresh generator from the same chunk seed.

### 7. Hyperparameters

| Parameter | Default | Source |
| --- | --- | --- |
| Seed bit-width $b_0$ | 3 | Paper and reference |
| Parent bit-width $B$ | 8 | Paper and reference |
| Codebook granularity | One per row (`group_count = 1`) | Reference default |
| Calibration | C4 train shard, 100 × 512 tokens, seed 0 | Reference default ([ADR 0002](0002-project-layout-and-architecture.md)) |
| Fisher variant | Empirical, float32 accumulation | This ADR |
| Seed initialization | Weighted greedy k-means++, $2 + \lfloor \ln K \rfloor$ local trials | Reference (`flash1dkmeans`) |
| Initialization seed (`quantizer.seed`) | 0, expanded per layer and per row chunk | This ADR |
| Rows per kernel call (`quantizer.row_chunk`) | 1024 | This ADR; bounds the `[R, L, n]` k-means++ trial tensor |
| Lloyd iteration cap | 50 | Reference |
| Empty-segment threshold $\varepsilon$ | `1e-12` (float64) | This ADR |
| Codebook storage dtype | float16 | Paper and reference |
| Mode | `incremental` (also `standalone` for baselines) | This ADR |

Dense-and-sparse outlier extraction (SqueezeLLM's optional sparse component, present but off by default in the reference) is out of scope.

## Consequences

The method is implemented in a few hundred lines of batched PyTorch whose key steps are exact or provably monotone, and whose invariants are directly testable.

- Positive: no Numba, no external k-means package, no CUDA extension; each property in Section 4 maps to a property-based test; the standalone mode reproduces the paper's main comparison.
- Negative: Lloyd's algorithm finds a local optimum that depends on the k-means++ draw, and our RNG stream differs from `flash1dkmeans`, so per-row codebooks will not match the reference bit-for-bit. Per-row codebooks are also expensive in memory for a model with 1024-wide rows (quantified in [ADR 0004](0004-evaluation-protocol.md)).
- Negative: the empirical Fisher is not bitwise reproducible on the GPU. CUDA backward kernels accumulate in a nondeterministic order, and two runs on the same data differed in the third significant digit of individual entries. Downstream results are reproducible because the Fisher diagonals are computed once and cached ([ADR 0002](0002-project-layout-and-architecture.md)). Enabling `torch.use_deterministic_algorithms` is an option if bitwise reproducibility is ever required.
- Future work:
  - **True Fisher**: sample labels from $p_\theta$ instead of using data tokens; a clean ablation of Approximation 1.
  - **Nested uniform round-to-nearest baseline**: split each uniform bin at its midpoint; the simplest nested comparator.
  - **Shared or grouped codebooks** to reduce the codebook memory overhead.
  - **Hadamard rotation** before clustering, specified in [ADR 0006](0006-hadamard-rotation.md).
  - **Chat-formatted calibration data** for the instruction-tuned model.
