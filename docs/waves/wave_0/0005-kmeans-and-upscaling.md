# Spec 0005: Weighted k-Means and Incremental Upscaling

- Status: Proposed
- Wave: [0](index.md)
- Implements: [ADR 0003](../../adr/0003-fisher-weighted-kmeans-methodology.md), Sections 2–6

This spec moves the batched kernels from notebook 02 into `anyprec.quantization`, and adds the two drivers the notebook lacks: `quantize_layer`, which chunks rows and emits storage-ready tensors, and `quantize_model`, which loops over all 168 modules. The three quantile initializations stay in the notebook as an ablation; the package implements only weighted k-means++.

## Files

Each kernel gets its own module so property tests map one-to-one onto files. All of them operate on a batch of rows `[R, n]` on whatever device the inputs are on.

| File | Contents |
| --- | --- |
| `quantization/rows.py` | `PreparedRows`, `prepare_rows`, `segment_stats` |
| `quantization/init.py` | `weighted_kmeanspp_init` |
| `quantization/lloyd.py` | `weighted_lloyd`, `LloydResult` |
| `quantization/split.py` | `split_all_segments`, `segment_ids` |
| `quantization/layer.py` | `LayerQuantization`, `quantize_layer` |
| `quantization/model.py` | `quantize_model`, `ModelQuantization` |

## Kernels

The kernels keep the notebook signatures, with one change: the constants become explicit arguments sourced from `QuantizerConfig`. Their bodies are the notebook's, which ADR 0003 Section 6 also shows as pseudo-code.

| Function | Signature | Shapes |
| --- | --- | --- |
| `prepare_rows` | `(weight: Tensor, fisher: Tensor) -> PreparedRows` | `[R, n]` float32 in; float64 prefix sums `[R, n + 1]` out |
| `segment_stats` | `(rows: PreparedRows, start: Tensor, end: Tensor, empty_eps: float) -> tuple[Tensor, Tensor, Tensor]` | int64 `[R, K]` in; float64 mass, mean, cost `[R, K]` out |
| `weighted_kmeanspp_init` | `(rows: PreparedRows, num_centroids: int, generator: torch.Generator) -> Tensor` | float64 `[R, K]`, ascending |
| `weighted_lloyd` | `(rows: PreparedRows, num_centroids: int, generator: torch.Generator, max_iter: int, empty_eps: float) -> LloydResult` | centroids `[R, K]`, borders int64 `[R, K + 1]` |
| `split_all_segments` | `(rows: PreparedRows, borders: Tensor, centroids: Tensor, empty_eps: float) -> tuple[Tensor, Tensor]` | centroids `[R, 2K]`, borders `[R, 2K + 1]` |
| `segment_ids` | `(borders: Tensor, n: int) -> Tensor` | int64 `[R, n]`, in sorted order |

`LloydResult` is a frozen dataclass with `centroids`, `borders`, and `iterations`. `PreparedRows` is the notebook's frozen dataclass, unchanged: `order`, `w_sorted`, `f_sorted`, `p0`, `p1`, `p2`.

Sensitivity conditioning (ADR 0003, Section 1) happens inside `prepare_rows`: zero weights get $f_j = 0$, and a row whose sensitivities sum to zero falls back to $f_j = 1$. The row sort is stable, so tied weights, which are common in bfloat16, keep their column order, and which tied column lands on which side of a split does not depend on the sort implementation.

## Layer driver

`quantize_layer` quantizes one weight matrix. It processes rows in chunks of `row_chunk`, turns sorted segment ids into indices in the original column order, and casts to the storage dtypes of ADR 0005.

```python
@dataclass(frozen=True)
class LayerQuantization:
    """Storage-ready quantization of one weight matrix (ADR 0005).

    :param indices: bits -> uint8 [m, n] on the CPU. Incremental mode stores only parent_bits;
        standalone mode stores every bit-width.
    :param luts: bits -> float16 [m, 2**bits] on the CPU, one codebook per row.
    :param relative_error: bits -> J / sum(f w^2) over the whole matrix, computed with the float16 LUTs;
        0 for an all-zero matrix, which every codebook reconstructs exactly. f is the conditioned
        sensitivity, as in notebook 02 (see the note below).
    :param lloyd_iterations: Maximum Lloyd iteration count over all chunks and bit-widths.
    """
    indices: dict[int, torch.Tensor]
    luts: dict[int, torch.Tensor]
    relative_error: dict[int, float]
    lloyd_iterations: int

def quantize_layer(weight: Tensor, fisher: Tensor, cfg: QuantizerConfig, generator_seed: int) -> LayerQuantization:
    """Quantize every row of one matrix; weight and fisher are [m, n] on the compute device.

    :raises ValueError: If n < 2**parent_bits, since a row cannot fill the parent codebook.
    """
```

```python
def quantize_layer(weight, fisher, cfg, generator_seed) -> LayerQuantization:
    m, n = weight.shape
    if n < 2**cfg.parent_bits:
        raise ValueError(...)
    bit_widths = range(cfg.seed_bits, cfg.parent_bits + 1)
    stored_bits = [cfg.parent_bits] if cfg.mode == "incremental" else list(bit_widths)
    indices = {b: torch.empty(m, n, dtype=torch.uint8) for b in stored_bits}
    luts = {b: torch.empty(m, 2**b, dtype=torch.float16) for b in bit_widths}
    error_sum = dict.fromkeys(bit_widths, 0.0)
    energy_sum = 0.0
    lloyd_iterations = 0

    for chunk_index, start in enumerate(range(0, m, cfg.row_chunk)):
        # One chunk of rows: [R, n] with R <= row_chunk. Each chunk gets its own seed, so every
        # Lloyd call below can start from a fresh generator in the same state.

        stop = min(start + cfg.row_chunk, m)
        rows = prepare_rows(weight[start:stop].float(), fisher[start:stop])
        energy_sum += (rows.f_sorted * rows.w_sorted.square()).sum().item()
        chunk_seed = stable_seed(generator_seed, f"chunk{chunk_index}")
        codebooks, iterations = _fit_codebooks(rows, cfg, chunk_seed)   # bits -> (centroids [R, K], borders [R, K + 1])
        lloyd_iterations = max(lloyd_iterations, iterations)

        for b, (centroids, borders) in codebooks.items():
            # Round the codebook to its storage dtype first, so the recorded error is the error
            # simulated inference will actually see.

            lut16 = centroids.to(torch.float16)                                      # [R, 2**b]
            ids_sorted = segment_ids(borders, n)                                     # [R, n] in sorted order
            residual = rows.w_sorted - lut16.double().gather(1, ids_sorted)          # [R, n]
            error_sum[b] += (rows.f_sorted * residual.square()).sum().item()
            luts[b][start:stop] = lut16.cpu()
            if b in indices:
                # Undo the sort: scatter sorted-position ids back to their original columns.

                ids = torch.empty_like(ids_sorted).scatter_(1, rows.order, ids_sorted)
                indices[b][start:stop] = ids.to(torch.uint8).cpu()

    relative_error = {b: error_sum[b] / energy_sum if energy_sum > 0 else 0.0 for b in bit_widths}
    return LayerQuantization(indices, luts, relative_error, lloyd_iterations)
```

`_fit_codebooks` is the private dispatch between the two modes. It returns the codebooks and the largest Lloyd iteration count of the chunk. Every `weighted_lloyd` call receives a fresh `torch.Generator(device=rows.w_sorted.device).manual_seed(chunk_seed)`.

- **Incremental.** One `weighted_lloyd` call at $K = 2^{b_0}$, then `parent_bits - seed_bits` calls to `split_all_segments`. Every level's `(centroids, borders)` is kept.
- **Standalone.** One independent `weighted_lloyd` call per bit-width. Because each call starts from the same generator state, the standalone fit at $b_0$ is bitwise identical to the incremental seed (ADR 0003, Section 4, property 4), and no width's codebook depends on the order in which the widths are fitted.

Three details matter for correctness:

- **The uint8 range.** Segment ids at `parent_bits = 8` lie in $[0, 255]$, so the cast to `uint8` is exact. `QuantizerConfig` rejects `parent_bits > 8` (spec 0002).
- **The nested property survives storage.** In incremental mode, only the parent indices are stored, and for every $b$ the index is `indices[parent_bits] >> (parent_bits - b)`. This holds because child $2k$ and $2k + 1$ come from parent $k$ (ADR 0003, Section 4, property 2). The unit tests check it on stored tensors, not just on borders.
- **Float16 rounding cannot break the indices.** Two adjacent centroids may round to the same float16 value, which merges them numerically. Rounding is monotone, so the LUT stays non-decreasing and every index still points to its own segment's value. The recorded `relative_error` already includes the effect.

**Note on `relative_error` and fallback rows.** The error and the energy both use the conditioned $f$, exactly as notebook 02 does, so the reference bands of spec 0011 apply unchanged. A row whose Fisher is entirely zero is clustered with $f_j = 1$, which is correct for that row's codebook, since only relative sensitivities within a row matter. But real Fisher values are tiny, so such a row can dominate its module's aggregated `relative_error`. This is a caveat of the metric, not of the codebooks. A module whose error looks out of line should first be checked for fallback rows.

## Model driver

`quantize_model` loops over the target weights in discovery order, derives each module's generator seed from the quantizer seed and the module name, and releases GPU memory between modules.

It takes weight tensors, not `nn.Linear` modules, so the caller decides what is clustered. The pipeline passes the module weights as they are (spec 0009). Once ADR 0006 is implemented, it passes the rotated weights $W R$ instead, and grouped codebooks would pass reshaped rows; neither change touches this driver or the kernels.

```python
@dataclass(frozen=True)
class ModelQuantization:
    layers: dict[str, LayerQuantization]           # discovery order
    seconds: float

def quantize_model(
    weights: Mapping[str, torch.Tensor],           # module name -> [m, n], discovery order
    fisher: Mapping[str, torch.Tensor],            # module name -> [m, n]
    cfg: QuantizerConfig,
    device: torch.device,
    progress: Callable[[str], None] | None = None,
) -> ModelQuantization:
    layers: dict[str, LayerQuantization] = {}
    with torch.no_grad():
        for name, weight in weights.items():
            # A mismatched Fisher would otherwise fail as an opaque gather error in the kernels.

            if fisher[name].shape != weight.shape:
                raise ValueError(f"{name}: Fisher shape ... differs from weight shape ...")

            # Per-module seeds make each codebook independent of processing order (ADR 0003, Section 3).

            w = weight.detach().to(device, torch.float32)                            # [m, n]
            f = fisher[name].to(device)                                              # [m, n]
            layers[name] = quantize_layer(w, f, cfg, stable_seed(cfg.seed, name))
            del w, f
            if progress is not None:
                progress(name)
    return ModelQuantization(layers, seconds=...)
```

`quantize_model` never modifies the tensors it is given, including live model parameters. Writing dequantized weights back is the job of spec 0007.

## Memory and time budget

The largest temporary is the k-means++ trial tensor `[R, L, n]` in float64, which `row_chunk` bounds. The figures below come from the notebook's measurements and the shapes of Granite's modules.

| Module kind | Rows `m` | `n` | Largest `[R, L, n]` at $K = 256$ ($L = 7$) with `row_chunk = 1024` |
| --- | --- | --- | --- |
| `input_linear` | 4096 | 1024 | 59 MB (4 chunks) |
| `output_linear` | 1024 | 2048 | 117 MB |
| `q_proj`, `o_proj` | 1024 | 1024 | 59 MB |
| `k_proj`, `v_proj` | 256 | 1024 | 15 MB |

`quantize_layer` was timed on the laptop GPU (RTX 3060, 6 GiB) on random bfloat16-rounded matrices of every Granite shape, with the default `row_chunk = 1024`:

| Mode | One layer (6 modules) | All 28 layers | Peak GPU memory |
| --- | --- | --- | --- |
| `incremental` | 0.8 s | about 0.4 min | 0.33 GiB |
| `standalone` | 12 s | about 5.6 min | 0.47 GiB |

Standalone mode is slower because it runs six Lloyd fits, up to $K = 256$, instead of one at $K = 8$. The acceptance run (spec 0011) records the figures on the real weights.

## Verification

Tests for this spec are listed in spec 0010 under `tests/quantization/`. The kernel tests are property-based (hypothesis) and run on the CPU with small rows, $n \le 64$.

- `prepare_rows` prefix sums match a Python brute force. Zero weights get zero sensitivity, and all-zero rows fall back to uniform weights.
- `segment_stats` matches a brute-force weighted mean and cost for random segments, including empty ones.
- `weighted_kmeanspp_init` returns sorted values drawn from the row. They are distinct whenever the row has at least $K$ distinct values with positive conditioned sensitivity (only those can be drawn), and identical for the same seed.
- `weighted_lloyd` never increases $J$ relative to its initialization. At convergence, every centroid is its segment's weighted mean, and the borders are a fixed point of the assignment step. On rows with $n \le 10$ and $K \le 3$, its $J$ is at least the exact optimum from a brute-force search over contiguous partitions.
- `split_all_segments` picks, for every segment, the split that a brute-force scan over all split points identifies as optimal. Its children are nested, and $J$ never increases.
- `quantize_layer` has these properties:
  - stored indices are `uint8` with values below $2^{b}$;
  - the nested right-shift identity holds for every stored bit-width;
  - LUTs are float16 and non-decreasing along each row;
  - `relative_error` is non-increasing in $b$ in incremental mode;
  - the same inputs and seed give bitwise-identical outputs on repeated calls;
  - incremental and standalone share the seed level exactly;
  - changing `row_chunk` preserves every chunk-independent property above (see the note below);
  - an all-zero matrix has `relative_error` 0, and rows shorter than $2^{B}$ raise `ValueError`.
- `quantize_model` produces one entry per target in order, leaves every model weight bitwise unchanged, gives each module the same codebooks whatever the module order, and raises `ValueError` when a Fisher shape differs from its weight's.

**Note on chunking.** The k-means++ draws for a chunk depend on which rows share that chunk, so changing `row_chunk` can change the codebooks. `row_chunk` is part of `QuantizerConfig`, and so part of the quantized cache key (spec 0002). The chunking test therefore asserts only the chunk-independent properties (shapes, dtypes, nesting, and monotone error), not equality across chunk sizes.
