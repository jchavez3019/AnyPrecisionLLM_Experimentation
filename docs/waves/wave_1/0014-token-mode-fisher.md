# Spec 0014: Token-Mode Fisher

- Status: Proposed
- Wave: [1](index.md)
- Implements: [ADR 0008](../../adr/0008-batched-and-per-token-fisher.md) (Token mode, Start-up capability check, `positions_per_sequence`, Stored scale and diagnostics)

This spec implements the per-token empirical Fisher: the sum over sequences and evaluated positions of each position's squared weight gradient. It provides both of ADR 0008's methods, the start-up check that guards `batched_backward`, and the random position subsample. All of it reuses spec 0013's accumulator and hooks.

## Files

Token mode lives in its own module. `estimate_fisher` (spec 0013) calls `token_fisher` and does nothing token-specific itself.

| File | Contents |
| --- | --- |
| `src/anyprec/sensitivity/token.py` | `FisherCapabilityError`, `sample_positions`, `token_fisher`, `_prefix_batch`, `_batched_backward`, `check_batched_backward` |
| `src/anyprec/sensitivity/hooks.py` | `output_capture`, added beside spec 0013's hooks |
| `tests/sensitivity/test_token.py` | The tests below |

## Interface

The functions below are public within `anyprec.sensitivity`.

```python
class FisherCapabilityError(RuntimeError):
    """``batched_backward`` gave wrong or slow-path gradients in this environment (ADR 0008)."""


def sample_positions(seq_len: int, k: int | None, seed: int, index: int) -> torch.Tensor:
    """Return the sorted int64 positions of sequence ``index`` whose losses are evaluated.

    ``None`` gives all ``seq_len - 1`` positions. Otherwise ``k`` distinct positions in
    ``[0, seq_len - 2]`` are drawn without replacement, from a CPU generator seeded with
    ``stable_seed(seed, f"fisher_positions/{index}")``.
    """


def token_fisher(
    model: CausalLM,
    calibration: torch.Tensor,
    targets: Mapping[str, nn.Linear],
    fisher: FisherConfig,
    seed: int,
    accumulator: FisherAccumulator,
    progress: Callable[[int, int], None] | None,
) -> torch.Tensor:
    """Accumulate the per-token Fisher of every sequence, and return the float32 ``[N]`` losses.

    Each sequence's loss is its mean NLL over the evaluated positions.

    :raises FisherCapabilityError: From ``check_batched_backward``, before any accumulation.
    """


@contextmanager
def output_capture(
    model: CausalLM, targets: Mapping[str, nn.Linear]
) -> Iterator[dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    """Keep every target's output ``Y`` ``[1, T, m]`` in the graph and its detached input ``A``
    ``[1, T, n]``, for one forward pass. Freezes every parameter, installs
    ``embedding_gradient_root``, and restores both in a ``finally`` block.
    """


def check_batched_backward(
    model: CausalLM, sequence: torch.Tensor, targets: Mapping[str, nn.Linear]
) -> None:
    """Compare batched and one-at-a-time output gradients at three positions (ADR 0008).

    :raises FisherCapabilityError: On a mismatch above the dtype's tolerance, or if PyTorch
        falls back to the vmap slow path. The message names ``fisher.token_method=prefix_batch``.
    """
```

## Algorithm

`token_fisher` runs the same outer loop for both methods, and differs only in the inner block:

```python
def token_fisher(model, calibration, targets, fisher, seed, accumulator, progress):
    # calibration: [N, T] int64. T - 1 loss positions per sequence, 0 .. T - 2.
    if fisher.token_method == "batched_backward":
        check_batched_backward(model, calibration[0].to(device), targets)
    for i in range(N):
        sequence = calibration[i].to(device)                                   # [T]
        positions = sample_positions(T, fisher.positions_per_sequence, seed, i).to(device)  # [P]
        # Scale so a subsample stays an unbiased estimate of the all-positions sum.
        scale = (T - 1) / len(positions)
        losses[i] = method(model, sequence, positions, targets, fisher.batch_size,
                           accumulator, half=i % 2, scale=scale)
        progress(i + 1, N)
```

Each method adds `scale * g.square()` for every position of its block into half `i % 2`. All items in a block come from one sequence, so they share one half. With all positions, `scale` is 1.

**`_prefix_batch`** is ADR 0008's `token_fisher_prefix_batch`, run under spec 0013's `item_gradient_hooks`. The hooks do the squaring and scaling, so this method only sets the block state and builds the batch and its loss:

```python
for block in positions.split(batch_size):                              # [Q] positions
    state.halves = torch.full((len(block),), half, device=device)      # [Q], one sequence
    state.scale = scale
    length = int(block[-1]) + 1
    items = sequence[:length].expand(len(block), length)               # [Q, L], identical items
    hidden = body_hidden_states_batched(model, items)                  # [Q, L, H]
    # Read each item at its own position only, which avoids the [Q, L, V] logits.
    picked = hidden[torch.arange(len(block)), block]                   # [Q, H]
    logits = logit_head(model)(picked).float()                         # [Q, V]
    nll = F.cross_entropy(logits, sequence[block + 1], reduction="none")   # [Q]
    nll.sum().backward()
```

`expand` makes a view, so the items share one token buffer. Hooks, not `.grad`, receive the gradient, which is why spec 0013's `BlockState` carries `scale`; the sequence paths leave it at 1.

**`_batched_backward`** is ADR 0008's `token_fisher_batched_backward`, with one saving that is exact. Position $t$'s loss depends only on tokens $0, \dots, t$, so $\Delta$ is zero after position $t$. For a block whose last position is $t_Q$, the contraction runs over the first $L = t_Q + 1$ tokens only:

```python
# delta: [Q, 1, T, m] from autograd.grad(..., is_grads_batched=True); a: [1, T, n].
length = int(block[-1]) + 1
# Keep only the prefix that can carry gradient: [Q, L, m] x [L, n] -> [Q, m, n].
g = torch.einsum("qtm,tn->qmn", delta[:, 0, :length].float(), a[0, :length].float())
accumulator.add_items(name, scale * g.square(), halves)             # halves: [Q], all equal to i % 2
```

This halves the contraction work over all positions, since the average prefix is $T / 2$. The contraction costs $2 L \cdot 249.6\text{M}$ floating-point operations per position, about 6.5 PFLOP for all 51,100 positions. That is 15 to 60 minutes of float32 matrix work on this GPU, depending on thermal throttling. The 12,775 vectorized backward calls at `batch_size: 4` add about 13 minutes at the probe's 0.06 s each. With `positions_per_sequence: 64`, both figures shrink about 8 times. Spec 0016's timing probe confirms them.

**`check_batched_backward`** follows ADR 0008's four steps:

1. It uses positions $0$, $\lfloor (T-1)/2 \rfloor$ and $T - 2$, which cover the exactly-zero `q_proj` gradient at position 0 and the full prefix at the end.
2. It computes $\Delta$ both ways, batched and one `torch.autograd.grad` per position, and forms each position's `g` for every target.
3. The error is the largest absolute difference over all targets and positions, divided by the largest absolute entry of the looped `g`. Normalizing by that global scale avoids dividing by position 0's zero gradient. The tolerance is `1e-4` in float32 and `1e-1` in bfloat16.
4. The batched call runs under `warnings.catch_warnings()`, with `warnings.filterwarnings("error", message=".*the batching rule for.*")`. PyTorch's vmap fallback warns with "…we have not yet implemented the batching rule for <op>…", so a fallback to the slow path raises, and it is converted to `FisherCapabilityError`.

## Memory budget

Both methods keep spec 0013's accumulators, 1 GB or 2 GB with `split_half`. Beyond those, the peaks for Granite at $T = 512$ in bfloat16 are:

| Method | Main extra memory at `batch_size` $Q$ |
| --- | --- |
| `prefix_batch` | $Q$ prefix graphs of up to 0.5 GB each; there are no full-vocabulary logits |
| `batched_backward` | One sequence's graph (0.9 GB), kept by `retain_graph`. Each of the block's $Q$ positions adds about 220 MB of $\Delta$ and 205 MB of float32 logit gradients |

For `batched_backward` with `split_half`, $Q = 4$ comes to about 0.7 + 2.0 + 0.9 + 0.9 + 0.8 = 5.3 GB. Spec 0016's probe picks between 2 and 4 from the measured peak.

## Determinism

`sample_positions` uses a CPU generator seeded per sequence, so the subsample is the same on every device and rerun, and independent of `batch_size`. The two methods and all batch sizes agree to the tolerances below, but not bitwise.

## Verification

The unit tests use the tiny Granite in float32 on the CPU. The reference is one `torch.autograd.grad` per position with respect to the target weights, squared and summed. Errors are normalized by the reference's largest entry.

- **Both methods match the reference.** `prefix_batch` and `batched_backward`, each at `batch_size` 1 and 3, match the reference to `1e-5` for all positions of two sequences.
- **Prefix truncation is exact.** `_batched_backward` gives the same result with and without the prefix cut.
- **An exact oracle.** For every attention layer, the per-position `q_proj` and `k_proj` gradients at position 0 are exactly zero from both methods.
- **Token mode is not sequence mode.** On a sequence of at least 8 tokens, the token Fisher differs from $(T-1)^2$ times the sequence Fisher by more than `1e-2` relative. This guards against the cross terms being dropped or kept by mistake.
- **Subsampling.**
  - `sample_positions` returns sorted, distinct positions in range, and is identical for equal arguments.
  - Different sequence indices give different draws.
  - With `k = T - 1`, the subsample is every position and the Fisher equals the all-positions Fisher.
  - With a smaller `k`, the Fisher equals $(T-1)/k$ times the sum of the reference's per-position squared gradients over exactly the drawn positions.
  - Unbiasedness is a property of the draw and the scale, so it is tested without the model. For a hypothesis-generated vector of positive per-position contributions, the scaled subset sum averaged over 2,000 sequence indices is within 5% of the full sum.
- **Capability check.**
  - It passes on the tiny model.
  - It raises `FisherCapabilityError` naming `prefix_batch` when a monkeypatched `torch.autograd.grad` perturbs the batched result.
  - It raises the same error when the patch emits the slow-path warning.
- **Losses.** With all positions, the returned loss of each sequence equals `causal_lm_loss` on it to `1e-6`.
- **Cleanup.** After a pass raises inside `output_capture`, no hooks remain, and every `requires_grad` flag is restored.
