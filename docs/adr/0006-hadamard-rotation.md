# ADR 0006: Hadamard Rotation Before Clustering (Opt-In, Deferred)

- Status: Proposed
- Date: 2026-09-24
- Deciders: Project maintainer

This ADR reserves an opt-in configuration for applying a randomized Hadamard rotation to each weight matrix before Fisher-weighted k-means, and specifies the mathematics and pseudo-code for when it is implemented. **In the initial implementation, selecting it raises `NotImplementedError`.** The rotation is wired into Hydra and validated now so that later enabling it changes no interfaces.

## Context

Rotation-based methods (QuIP and QuIP#, QuaRot, SpinQuant) multiply weights by an orthogonal matrix before quantizing. The rotation leaves the layer's function unchanged but spreads outlier magnitude across all coordinates, so that each row looks approximately Gaussian with a small ratio of maximum to root-mean-square value. This matters most for uniform quantizers, whose step size scales with the range.

The quantizer of [ADR 0003](0003-fisher-weighted-kmeans-methodology.md) is non-uniform and already adapts to the distribution, so the expected gain is smaller, and there is a specific interaction with Fisher weighting that may offset it (Section "Interaction with Fisher weighting" below). Whether the rotation helps is therefore an empirical question for a later milestone, answered with the metrics of [ADR 0004](0004-evaluation-protocol.md).

The rotation is compatible with incremental upscaling. The paper's AWQ failure (its Appendix E.2) came from preprocessing tuned to one bit-width; a Hadamard rotation is independent of both the data and the bit-width, so one rotation serves every precision and the nested index property is untouched.

## Decision

### Configuration (implemented now)

Two options in the `rotation` Hydra group, validated by a pydantic discriminated union on `kind`:

```yaml
# configs/rotation/none.yaml
kind: none
```

```yaml
# configs/rotation/hadamard.yaml
kind: hadamard
axis: in_features          # rotate along each row, i.e. the dimension sharing a codebook
randomized_signs: true     # required for the incoherence guarantee below
seed: 0                    # seeds the per-module sign vectors
```

`rotation.*` is part of the Fisher cache key ([ADR 0002](0002-project-layout-and-architecture.md)), since rotated Fisher diagonals differ from unrotated ones.

### Behaviour in the initial implementation

The quantization and evaluation entry points resolve the rotation immediately after config validation and **before** loading any model or data, so the failure is fast and cheap:

```python
def resolve_rotation(cfg: RotationConfig) -> None:
    match cfg:
        case RotationNone():
            return None
        case RotationHadamard():
            raise NotImplementedError(
                "Hadamard rotation is specified in ADR 0006 but not implemented yet; "
                "use rotation=none."
            )
```

Python's built-in `NotImplemented` is a sentinel constant for binary operators, not an exception; `raise NotImplemented` would itself fail with a `TypeError`. The exception is `NotImplementedError`.

Tests (per [ADR 0001](0001-code-maintainability.md)) assert that `rotation=hadamard` validates as configuration, that the entry pipeline raises `NotImplementedError`, and that it does so before any model loading is attempted.

### Mathematics (for the future implementation)

**Construction.** For $n$ a power of two, the Sylvester Hadamard matrix is defined recursively:

$$
H_1 = [1], \qquad H_{2k} = \begin{bmatrix} H_k & H_k \\ H_k & -H_k \end{bmatrix}, \qquad H_n H_n^\top = n I, \quad H_n = H_n^\top.
$$

With a diagonal matrix of independent uniform random signs $D = \mathrm{diag}(s_1,\dots,s_n)$, $s_k \in \lbrace \pm 1\rbrace$, define the randomized rotation

$$
\tilde H = \tfrac{1}{\sqrt n}\, D H_n, \qquad \tilde H \tilde H^\top = \tfrac{1}{n} D H_n H_n^\top D = I.
$$

All of Granite's quantizable input dimensions are powers of two (1024 for five layers, 2048 for `shared_mlp.output_linear`); a non-power-of-two dimension is a configuration error.

**Function-preserving rotation.** For a layer $y = Wx$ with $W \in \mathbb{R}^{m \times n}$:

$$
y = W x = (W \tilde H)(\tilde H^\top x) = W' x', \qquad W' = W\tilde H.
$$

Each row of $W'$ is $w' = w \tilde H$, so rotation mixes exactly the values that share a codebook. Quantization is applied to $W'$, and the effective quantized weight is $\hat W = \hat W' \tilde H^\top$. Because $\tilde H$ is orthogonal, $\lVert W - \hat W\rVert_F = \lVert W' - \hat W' \rVert_F$: weight error is the same in either basis. In simulated inference the rotation is folded back into the weight, so the Hugging Face model is not modified. (A deployed kernel would instead apply $\tilde H^\top x$ online with a fast Walsh–Hadamard transform in $O(n \log n)$.)

**Incoherence.** For a fixed row, $w'_j = \tfrac{1}{\sqrt n}\sum_k s_k H_{kj} w_k$ is a sum of independent, bounded, zero-mean terms. Hoeffding's inequality gives $\Pr(|w'_j| \ge t) \le 2\exp\big(-t^2 n / (2\lVert w\rVert_2^2)\big)$, and a union bound over $j$ gives, with probability at least $1-\delta$,

$$
\begin{aligned}
\max_j |w'_j| &\le \lVert w \rVert_2 \sqrt{\frac{2\ln(2n/\delta)}{n}}, \\
\frac{\max_j |w'_j|}{\mathrm{rms}(w')} &\le \sqrt{2\ln(2n/\delta)}.
\end{aligned}
$$

For $n = 1024$ and $\delta = 0.01$ the ratio is at most about 4.9, however spiky the original row was. The random signs are essential: without $D$, a row equal to a column of $H_n$ would be mapped to a single spike. Orthogonality preserves $\lVert w\rVert_2$, so the row variance $\sigma^2$ is unchanged and before/after comparisons are at equal variance.

**Expected effect on the quantizer.** For a uniform grid on $[-M, M]$, MSE $\approx \Delta^2/12$ with $\Delta = 2M/(2^b-1)$, so error scales with $M^2$ and the rotation can reduce it by an order of magnitude for rows with outliers. For an optimal non-uniform quantizer, the Panter–Dite high-resolution approximation gives

$$
D_b \approx \tfrac{1}{12}\, 4^{-b} \left(\int p(x)^{1/3}\,dx\right)^{3},
$$

which evaluates to $4.50\,\sigma^2 4^{-b}$ for a Laplacian density and $2.72\,\sigma^2 4^{-b}$ for a Gaussian. Gaussianizing a Laplacian-like row therefore lowers optimal-quantizer distortion by a factor of about 1.65, equivalent to about 0.36 bits, since each bit divides distortion by 4. This is the expected order of the benefit for k-means.

**Fisher in the rotated basis.** Writing $W = W'\tilde H^\top$ and $G = \partial \ell / \partial W$, the chain rule gives $\partial \ell / \partial W' = G \tilde H$. Squaring must happen **after** rotating each per-sequence gradient:

$$
\hat F'_{rj} = \sum_{i=1}^{N} \big( G_i \tilde H \big)_{rj}^{2} \;\neq\; \big(\hat F \tilde H\big)_{rj}.
$$

### Interaction with Fisher weighting

Expanding the rotated diagonal shows it mixes the within-row off-diagonal entries of the original empirical Fisher:

$$
\hat F'_{jj} = \sum_{k,l} \tilde H_{kj}\,\tilde H_{lj}\,\hat F_{kl} \quad \text{(within one row)}.
$$

This is partly beneficial, since it recovers interaction terms that the diagonal approximation of ADR 0003 discards. It also flattens sensitivity: if $\hat F$ is diagonal with a few highly sensitive weights, then, using $\tilde H_{kj}^2 = 1/n$, $\hat F'_{jj} = \tfrac{1}{n}\sum_k \hat F_{kk}$ for every $j$, and the weighted k-means degenerates toward unweighted k-means on a Gaussian-like row. Whether the approximately 0.36-bit shape gain outweighs the lost saliency weighting is the question the future experiment answers, measured by KL and perplexity rather than weight MSE.

### Pseudo-code (for the future implementation)

```python
def sylvester_hadamard(n: int) -> Tensor:                    # [n, n], entries ±1
    assert n > 0 and n & (n - 1) == 0, "n must be a power of two"
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H

def rotation_for(module_name: str, n: int, cfg: RotationHadamard) -> Tensor:
    # Independent, reproducible signs per module, derived from the global seed and the name.
    gen = torch.Generator().manual_seed(stable_seed(cfg.seed, module_name))
    s = torch.randint(0, 2, (n,), generator=gen) * 2 - 1     # [n] of ±1
    return (s[:, None] * sylvester_hadamard(n)) / n ** 0.5   # D H / sqrt(n), [n, n]

# Fisher accumulation: rotate the per-sequence gradient, then square.
def hook(p, name=name, R=rotations[name]):
    fisher[name].add_((p.grad.float() @ R).square())          # [m, n] @ [n, n] -> [m, n]
    p.grad = None

# Quantization: cluster the rotated weights with the unchanged ADR 0003 pipeline.
W_rot = W.float() @ R                                         # [m, n]
idx, luts = quantize_layer(W_rot, fisher[name], seed_bits, parent_bits, stable_seed(seed, name), row_chunk)

# Simulated inference: dequantize in the rotated basis, then rotate back.
W_hat = luts[bits][name].float().gather(1, idx.long() >> shift) @ R.T   # [m, n]
module.weight.copy_(W_hat.to(module.weight.dtype))
```

Artifact changes when implemented ([ADR 0005](0005-artifact-format-and-simulated-inference.md)): a `rotation.safetensors` file storing each module's sign vector as `int8 [n]`, and the rotation config in the manifest. Storing the signs, rather than re-deriving them from the seed, keeps artifacts valid even if the hashing scheme changes.

## Consequences

The configuration surface for rotation exists and is tested from the start, and the future implementation is specified down to the gradient transform.

- Positive: enabling the rotation later changes no configuration schema, cache-key structure, or entry-point signature; the combination of quantizer and rotation becomes a single Hydra sweep; the expected benefit and its main risk are stated up front.
- Negative: until implemented, `rotation=hadamard` is a configuration value that always fails; this is deliberate and loud rather than silently ignored.
- When implemented: amend this ADR's status and replace the `NotImplementedError` branch; add tests for orthogonality ($\tilde H\tilde H^\top = I$), function preservation at high bit-width, the rotated-gradient Fisher against a float64 reference, and the incoherence bound on synthetic spiky rows.
