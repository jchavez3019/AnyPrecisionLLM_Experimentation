# Spec 0004: Fisher Estimation

- Status: Proposed
- Wave: [0](index.md)
- Implements: [ADR 0003](../../adr/0003-fisher-weighted-kmeans-methodology.md), Section 1 (empirical Fisher) and Section 6 (`estimate_fisher`)

This spec turns the notebook's `estimate_fisher` into `anyprec.sensitivity.estimate_fisher`. It fixes the hook mechanics, the memory budget on a 6 GB GPU, the guarantees the function makes about the model's state afterwards, and how the result reaches the cache.

## Files

The whole feature is one module. Caching is the artifact store's job (spec 0006), not this module's.

| File | Contents |
| --- | --- |
| `src/anyprec/sensitivity/fisher.py` | `estimate_fisher`, `FisherResult` |

## Interface

The function takes the model, the calibration tokens, and the target modules, and returns CPU tensors, so the caller can free GPU memory before quantization starts.

```python
@dataclass(frozen=True)
class FisherResult:
    """Empirical Fisher diagonals and calibration diagnostics.

    :param diagonals: float32 CPU tensors keyed by module name, each shaped like its weight [m, n].
    :param losses: Per-sequence mean token NLL, float32 [N].
    :param seconds: Wall-clock time of the accumulation loop.
    """
    diagonals: dict[str, torch.Tensor]
    losses: torch.Tensor
    seconds: float

def estimate_fisher(
    model: PreTrainedModel,
    calibration: torch.Tensor,
    targets: Mapping[str, nn.Linear],
    progress: Callable[[int, int], None] | None = None,
) -> FisherResult:
    """Accumulate the empirical Fisher diagonal of every target weight (ADR 0003, Section 1)."""
```

`progress(i, N)` is an optional callback. The pipeline passes a `tqdm` updater, and the kernel itself never imports `tqdm`.

## Algorithm

This is the notebook implementation, with three changes: the state is restored even on failure, the accumulators are moved to the CPU at the end, and gradients of every non-target parameter are provably never materialized.

```python
def estimate_fisher(model, calibration, targets, progress=None) -> FisherResult:
    # Remember every parameter's requires_grad flag so the model is returned exactly as received.

    saved_flags = {name: p.requires_grad for name, p in model.named_parameters()}
    device = next(model.parameters()).device
    accumulators: dict[str, torch.Tensor] = {}
    handles: list[RemovableHandle] = []
    losses = torch.empty(calibration.shape[0], dtype=torch.float32)
    try:
        # Freeze everything, then enable gradients only on target weights.
        # Activations still carry gradients through the frozen embedding and norms.

        for p in model.parameters():
            p.requires_grad_(False)
        for name, linear in targets.items():
            linear.weight.requires_grad_(True)
            accumulators[name] = torch.zeros_like(linear.weight, dtype=torch.float32)   # [m, n]
            handles.append(linear.weight.register_post_accumulate_grad_hook(_square_into(accumulators[name])))

        # One sequence per backward pass: the gradient must be squared before summing over sequences.

        started = time.perf_counter()
        for i in range(calibration.shape[0]):
            tokens = calibration[i : i + 1].to(device)                                    # [1, T]
            loss = model(input_ids=tokens, labels=tokens, use_cache=False).loss           # mean token NLL
            loss.backward()
            losses[i] = loss.detach().float().cpu()
            if progress is not None:
                progress(i + 1, calibration.shape[0])
        seconds = time.perf_counter() - started
    finally:
        # Remove hooks, drop any gradient left by a failed pass, and restore the original flags.

        for handle in handles:
            handle.remove()
        for name, p in model.named_parameters():
            p.grad = None
            p.requires_grad_(saved_flags[name])

    # Move to CPU one module at a time so peak GPU memory never holds two copies.

    diagonals = {name: accumulators.pop(name).cpu() for name in list(accumulators)}
    return FisherResult(diagonals=diagonals, losses=losses, seconds=seconds)

def _square_into(accumulator: torch.Tensor) -> Callable[[torch.Tensor], None]:
    """Hook: add grad**2 in float32 to the accumulator, then free .grad."""
    def hook(parameter: torch.Tensor) -> None:
        grad = parameter.grad
        if grad is None:
            raise RuntimeError("post-accumulate hook fired without a gradient")
        accumulator.add_(grad.float().square())                                          # [m, n]
        parameter.grad = None
    return hook
```

The forward pass runs under the default grad mode, so the pipeline must not wrap the call in `torch.no_grad()`. `estimate_fisher` asserts `torch.is_grad_enabled()` on entry and raises a `RuntimeError` otherwise. Otherwise a stray `no_grad` would return all-zero Fisher diagonals.

## Memory budget

On the 6 GB laptop GPU, the notebook measured a peak of 2.63 GiB for this exact configuration. The table breaks that down so a regression is easy to spot.

| Item | Size |
| --- | --- |
| Model weights, bfloat16 | 0.70 GB |
| Float32 accumulators, 249.6M entries | 1.00 GB |
| One target's bfloat16 gradient (freed by the hook) | at most 8 MB |
| Logits and their gradient, one 512-token sequence, float32 | about 0.41 GB |
| Activations, one 512-token sequence | about 0.1 GB |

The integration test in spec 0011 asserts a peak below 3.5 GiB, which leaves headroom for allocator fragmentation.

## Determinism

The empirical Fisher is not bitwise reproducible on CUDA (ADR 0003, Consequences). This function does not try to make it so. The pipeline computes it once and caches it (spec 0006), so every downstream artifact built from one cached Fisher is reproducible.

## Verification

Tests for this spec are listed in spec 0010 under `tests/sensitivity/`. They run on the CPU with the tiny Granite fixture.

- **Correctness against a reference.** For three random sequences, the result equals $\sum_i g_i^2$ where each $g_i$ comes from `torch.autograd.grad` on sequence $i$ alone ($\mathrm{rtol} = 10^{-5}$, float32 model).
- **Squared per sequence, not per batch.** The result differs from $(\sum_i g_i)^2$ for the same sequences. This guards against someone batching the loop.
- **State restoration.** After the call, every parameter's `requires_grad` equals its value before, every `.grad` is `None`, and no hooks remain: a subsequent `backward()` leaves the returned accumulators unchanged.
- **Failure safety.** A model whose forward raises on the second sequence leaves the same restored state, and the exception propagates.
- **Grad mode guard.** Calling it inside `torch.no_grad()` raises `RuntimeError`.
- **Output contract.** Diagonals are float32, on the CPU, shaped like their weights, keyed in `targets` order, and non-negative. `losses` has shape `[N]`.
