# Spec 0013: Per-Item Gradient Hooks and Batched Sequence Mode

- Status: Proposed
- Wave: [1](index.md)
- Implements: [ADR 0008](../../adr/0008-batched-and-per-token-fisher.md) (Shared machinery, Sequence mode with `batch_size > 1`, Diagnostics)

This spec builds the machinery that every non-default estimator shares: a Fisher accumulator that can keep split halves, and layer hooks that yield one weight gradient per batch item. It then uses them for sequence mode with `batch_size > 1`, and turns `estimate_fisher` into the single entry point that dispatches on `FisherConfig`.

## Files

The token-mode estimators of spec 0014 are added to the same package and reuse everything here.

| File | Contents |
| --- | --- |
| `src/anyprec/sensitivity/accumulator.py` | `FisherAccumulator`, `SplitHalfCorrelation`, `split_half_correlation` |
| `src/anyprec/sensitivity/hooks.py` | `embedding_gradient_root`, `item_gradient_hooks` |
| `src/anyprec/sensitivity/fisher.py` | `FisherResult` gains `split_half`; `estimate_fisher` dispatches; the default loop moves to `_sequence_fisher_unbatched`; new `_sequence_fisher_batched` |
| `src/anyprec/models/heads.py` | `body_hidden_states_batched`, `token_nll`; `body_hidden_states` becomes a one-item wrapper |
| `tests/sensitivity/test_accumulator.py`, `tests/sensitivity/test_hooks.py`, `tests/sensitivity/test_fisher.py` | The tests below |

## Interface

**Model helpers.** The batched body is the existing `body_hidden_states` without its squeeze, so both paths share one implementation:

```python
def body_hidden_states_batched(model: CausalLM, input_ids: torch.Tensor) -> torch.Tensor:
    """Run the decoder body without the LM head: ``[Q, L]`` token ids -> ``[Q, L, H]``."""


def token_nll(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Per-position next-token NLL in float32: ``[Q, T, V]`` and ``[Q, T]`` -> ``[Q, T - 1]``."""
```

`body_hidden_states(model, ids)` returns `body_hidden_states_batched(model, ids)[0]`, so its `[1, T] -> [T, H]` contract is kept. Logits are always `logit_head(model)(hidden)`, which applies Granite's `logits_scaling`, and `check_sliced_logits` already pins that against `model(...).logits`. `logit_head`'s docstring is widened from `[S, H] -> [S, V]` to any leading shape, `[..., H] -> [..., V]`, which `nn.Linear` already supports.

**The accumulator.** It owns the float32 sums on the model's device and, with `split_half`, a second set:

```python
class FisherAccumulator:
    """Float32 Fisher sums per target, optionally split by calibration-sequence parity."""

    def __init__(self, targets: Mapping[str, nn.Linear], split_half: bool) -> None: ...

    def add(self, name: str, squared: torch.Tensor, half: int) -> None:
        """Add one ``[m, n]`` squared gradient into half ``half`` (0 or 1)."""

    def add_items(self, name: str, squared: torch.Tensor, halves: torch.Tensor) -> None:
        """Add ``[Q, m, n]`` per-item squared gradients; ``halves`` is int64 ``[Q]`` of 0 or 1."""

    def finish(self) -> tuple[dict[str, torch.Tensor], dict[str, SplitHalfCorrelation] | None]:
        """Return the CPU diagonals (the sum of both halves), and the split-half correlations."""
```

- With `split_half=False`, `half` and `halves` are ignored, and there is one accumulator per module, as in wave 0.
- A sequence's half is `i % 2`, where `i` is its index along the first dimension of the `[N, T]` calibration tensor. The halves are therefore sequences 0, 2, 4, … and 1, 3, 5, ….
- `finish` moves each module to the CPU in turn and computes its correlation there, so the GPU never holds more than the accumulators.

```python
@dataclass(frozen=True)
class SplitHalfCorrelation:
    """Spearman correlation of one module's two half-diagonals, and its Spearman–Brown projection.

    The projection ``2 r / (1 + r)`` only means something for a positive ``r``, so
    ``spearman_full`` is ``None`` when ``spearman_half <= 0``.
    """

    spearman_half: float
    spearman_full: float | None


def average_ranks(values: torch.Tensor) -> torch.Tensor:
    """Rank a 1-D tensor from 1 to ``len(values)`` in float64, giving tied values their mean rank."""


def split_half_correlation(first: torch.Tensor, second: torch.Tensor) -> SplitHalfCorrelation:
    """Rank-correlate two ``[m, n]`` half-diagonals, then project with ``r_full = 2 r / (1 + r)``.

    The correlation is Pearson's on the flattened entries' ``average_ranks``, which is Spearman's
    with ties averaged, as ``scipy.stats.spearmanr`` defines it.

    :raises ValueError: If either half is constant, where the rank correlation is undefined.
    """
```

Spearman's correlation is implemented in torch, because `scipy.stats` is untyped and would need a cast at every call under strict pyright. It stays small: `average_ranks` sorts, finds runs of equal values with `unique_consecutive(return_counts=True)`, and assigns each run the mean of its positions. The unit tests use `scipy.stats.spearmanr` as the oracle.

**The hooks.** One context manager installs everything ADR 0008's Shared machinery lists, and removes it in a `finally` block:

```python
@dataclass
class BlockState:
    """What the output-gradient hooks need to know about the block being back-propagated.

    :param halves: int64 ``[Q]``, the split half of each batch item's calibration sequence.
    :param scale: Factor on every squared gradient; 1 except for token-mode subsamples (spec 0014).
    """

    halves: torch.Tensor
    scale: float = 1.0


@contextmanager
def item_gradient_hooks(
    model: CausalLM,
    targets: Mapping[str, nn.Linear],
    accumulator: FisherAccumulator,
) -> Iterator[BlockState]:
    """Accumulate the squared per-item weight gradient of every target during each backward.

    The caller sets the yielded state's ``halves`` and ``scale`` for the current block before
    calling ``backward``; the hooks read them when they fire. Every parameter is frozen for the
    duration and its ``requires_grad`` flag restored afterwards, even if a pass raises.
    """
```

`embedding_gradient_root(model)` is the forward hook on `model.get_input_embeddings()` that calls `requires_grad_()` on its output. With every weight frozen, the embedding output is a leaf, so this is what gives layer outputs a gradient path. `item_gradient_hooks` installs it, and spec 0014's `output_capture` installs it too.

**The entry point.** `estimate_fisher` gains the estimator config and the calibration seed:

```python
@dataclass(frozen=True)
class FisherResult:
    diagonals: dict[str, torch.Tensor]
    losses: torch.Tensor
    seconds: float
    split_half: dict[str, SplitHalfCorrelation] | None


def estimate_fisher(
    model: CausalLM,
    calibration: torch.Tensor,
    targets: Mapping[str, nn.Linear],
    fisher: FisherConfig,
    seed: int,
    progress: Callable[[int, int], None] | None = None,
) -> FisherResult:
    """Accumulate the configured empirical Fisher diagonal (ADR 0003, Section 1; ADR 0008)."""
```

| `granularity` | `batch_size` | Runs |
| --- | --- | --- |
| `sequence` | 1 | `_sequence_fisher_unbatched`: wave 0's post-accumulate loop, with the accumulator in place of the bare tensors |
| `sequence` | > 1 | `_sequence_fisher_batched` (below) |
| `token` | any | spec 0014's `token_fisher`, with `token_method` choosing the method |

`seed` is used only for token mode's position draws; the pipeline passes `cfg.calibration.seed` (ADR 0008). `progress` receives `(sequences done, N)` on every path. `seconds` covers the accumulation loop, and not the split-half correlations computed by `finish`.

## Algorithm

The default path changes only where it adds. Its hook now calls `accumulator.add(name, grad.float().square(), half)`, where `half` is the current sequence's parity. Without `split_half`, this is the same single float32 sum as in wave 0.

The batched path follows ADR 0008's `estimate_fisher_batched`. The per-item hook does the following, with shapes for one module $y = Wa$:

```python
def _on_output_gradient(name: str, a: torch.Tensor, delta: torch.Tensor) -> None:
    # a: [Q, T, n] input saved by the forward hook, detached, in the model dtype.
    # delta: [Q, T, m] gradient of the summed per-item losses with respect to y.
    # Contract over tokens but keep the item index: [Q, T, m] x [Q, T, n] -> [Q, m, n].
    g = torch.einsum("qtm,qtn->qmn", delta.float(), a.float())
    # Square elementwise before summing over items, which is the point of the per-item path.
    accumulator.add_items(name, state.scale * g.square(), state.halves)
```

```python
def _sequence_fisher_batched(model, calibration, targets, fisher, accumulator, state):
    # calibration: [N, T] int64. Blocks of Q <= batch_size consecutive sequences.
    for start in range(0, N, fisher.batch_size):
        block = calibration[start : start + fisher.batch_size].to(device)       # [Q, T]
        state.halves = torch.arange(start, start + len(block), device=device) % 2   # [Q]
        hidden = body_hidden_states_batched(model, block)                      # [Q, T, H]
        logits = logit_head(model)(hidden).float()                             # [Q, T, V]
        nll = token_nll(logits, block)                                         # [Q, T - 1]
        losses[start : start + len(block)] = nll.mean(dim=1).detach().cpu()
        # Sum of per-item means, so item q's output gradient is d(l_q)/dy_q.
        nll.mean(dim=1).sum().backward()
```

- The saved input `a` is keyed by the input tensor's identity, so `q_proj`, `k_proj` and `v_proj` store one reference, not three copies.
- The output hook is registered in the forward hook with `output.register_hook`. PyTorch runs it once, after every gradient contribution to that output has been summed.
- Saved inputs are cleared after each backward, so one block's activations never outlive it.

## Memory budget

Batching trades memory for fewer, larger passes. The numbers below are for Granite at $T = 512$ in bfloat16, alongside wave 0's measured 2.63 GiB for the whole quantize stage.

| Item | Size |
| --- | --- |
| Model weights | about 0.7 GB |
| Accumulators, one set | 1.0 GB (249.6M float32 entries) |
| Second set with `split_half` | +1.0 GB |
| Autograd graph, per sequence | about 0.9 GB: wave 0's peak less the weights and accumulators. It includes the 0.41 GB of float32 logits and softmax |
| Largest per-item gradient `g`, per item | 16 MB (`input_linear`, $4096 \times 1024$) |

The saved inputs add nothing, because each linear's autograd node already keeps the same tensor. Without `split_half`, `batch_size: 4` needs about 0.7 + 1.0 + 3.6 = 5.3 GB, at the edge of the 6 GiB card. With `split_half`, `batch_size: 2` (4.5 GB) is the largest that fits. The larger batches ADR 0008 describes need the sliced head it leaves out of scope. The integration test in spec 0015 measures the real peak.

## Determinism

The halves and the block boundaries depend only on sequence indices, so a rerun with the same config makes the same assignments. GPU reductions stay non-deterministic, as ADR 0003 records. Batching changes the order of floating-point additions, so batched and unbatched Fishers agree to float32 tolerance, not bitwise.

## Verification

Every test runs on the tiny Granite of `tests/factories.py` in float32 on the CPU, with errors measured against the largest entry of the reference diagonal.

- **Batching is exact.** `batch_size` 1, 3 and 4 give the same diagonals to `1e-5` relative, with $N = 7$ so the last block is partial.
- **The hooks match autograd.** For a single sequence, the per-item gradient from the hooks equals `torch.autograd.grad` with respect to each target weight.
- **Items stay separate.** With two different sequences in one block, the accumulated diagonal equals the sum of the two sequences' squared gradients, and differs from the square of their summed gradient.
- **Split halves add up.** With `split_half`, the diagonals equal the run without it, and each half equals the Fisher of only the even or only the odd sequences.
- **Correlation.**
  - `split_half_correlation` of a tensor with itself gives 1 for both fields.
  - A strictly decreasing transform gives a `spearman_half` of -1 and no projection.
  - A constant half raises `ValueError`.
  - The projection is checked at 0.5 and 1.
- **Ranks match scipy.** On hypothesis-generated tensors with deliberate ties, `average_ranks` equals `scipy.stats.rankdata`, and the correlation equals `scipy.stats.spearmanr` to `1e-12`.
- **Scale.** A `state.scale` of 3 triples the accumulated diagonal.
- **Cleanup.** After a backward pass raises inside `item_gradient_hooks`, no hooks remain on any module, and every `requires_grad` flag is back to its original value.
- **Grad mode.** `estimate_fisher` still raises `RuntimeError` inside `torch.no_grad()` on every path.
