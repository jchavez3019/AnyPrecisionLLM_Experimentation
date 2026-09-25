# Spec 0007: Simulated Inference

- Status: Proposed
- Wave: [0](index.md)
- Implements: [ADR 0005](../../adr/0005-artifact-format-and-simulated-inference.md) (simulated inference)

This spec defines `set_precision`, which writes a chosen bit-width's dequantized weights into an unmodified Hugging Face model, and `restore_weights`, which undoes it. Evaluation (spec 0008) sweeps bit-widths by calling `set_precision` repeatedly on one loaded model.

## Files

Inference is one module. It depends on the artifact types from spec 0006 and on `torch`, but not on `transformers`: it only needs `named_modules()`.

| File | Contents |
| --- | --- |
| `inference/precision.py` | `set_precision`, `snapshot_weights`, `restore_weights`, `PrecisionError` |

## Interface

The functions take any `nn.Module`, so the tiny test model and Granite are handled identically.

```python
class PrecisionError(ValueError): ...

@torch.no_grad()
def set_precision(model: nn.Module, artifact: QuantizedArtifact, bits: int) -> None:
    """Overwrite every quantized weight with its bits-wide dequantization (ADR 0005)."""

def snapshot_weights(model: nn.Module, names: Sequence[str]) -> dict[str, torch.Tensor]:
    """Copy the named modules' weights to the CPU, so they can be restored later."""

@torch.no_grad()
def restore_weights(model: nn.Module, saved: Mapping[str, torch.Tensor]) -> None:
    """Copy saved weights back in place."""
```

## Algorithm

The artifact stays on the CPU. For each module, the indices and one codebook are moved to the weight's device, gathered, and written in place, so at most one module's artifact data is on the GPU at a time.

```python
@torch.no_grad()
def set_precision(model, artifact, bits) -> None:
    manifest = artifact.manifest
    if not manifest.seed_bits <= bits <= manifest.parent_bits:
        raise PrecisionError(f"bits={bits} outside [{manifest.seed_bits}, {manifest.parent_bits}]")

    # Resolve every module before writing any weight, so a mismatch cannot leave the model half-updated.

    modules = dict(model.named_modules())
    targets = [_checked_linear(modules, entry) for entry in manifest.modules]

    for entry, linear in zip(manifest.modules, targets, strict=True):
        weight = linear.weight                                                  # [m, n], eval dtype
        idx = _indices_for(artifact, entry.name, bits).to(weight.device)        # uint8 [m, n]
        lut = artifact.luts[bits][entry.name].to(weight.device, weight.dtype)   # [m, 2**bits]

        # Gather one codebook entry per weight: [m, 2**bits] indexed by [m, n] -> [m, n].

        weight.copy_(lut.gather(1, idx.long()))

def _indices_for(artifact, name, bits) -> torch.Tensor:
    """Incremental: parent indices shifted right. Standalone: that width's own indices."""
    manifest = artifact.manifest
    if manifest.mode == "incremental":
        return artifact.indices[manifest.parent_bits][name] >> (manifest.parent_bits - bits)
    return artifact.indices[bits][name]
```

`_checked_linear` raises `PrecisionError` if a name is missing from the model, is not an `nn.Linear`, or has a weight shape different from the manifest's. The right shift is applied to the `uint8` tensor on the CPU, before the transfer, because shifting a `uint8` is exact and keeps the transfer at one byte per weight.

Only the listed modules are written. The embedding, the tied LM head, the norms, and every bias are never touched (ADR 0005).

## Dtype

Evaluation loads the model in `model.eval_dtype`, float32 by default (ADR 0005). `set_precision` casts the float16 codebook to the weight's dtype. In float32 this is exact; in bfloat16 it rounds, and the rounding can merge neighbouring centroids. That is the documented caveat of `eval_dtype: bfloat16`. The function does not warn about it, because the choice is explicit in the config.

## Restoring the reference

Evaluation keeps a separate unmodified reference model, so it never needs to restore weights (spec 0008). `snapshot_weights` and `restore_weights` exist for tests, and for callers that want to reuse one model. Restoring a float32 copy of a bfloat16 checkpoint is exact.

## Verification

Tests for this spec are listed in spec 0010 under `tests/inference/`. They use the tiny Granite fixture and an artifact built by `quantize_model` on it (spec 0005).

- After `set_precision(bits=b)`, every quantized weight equals `lut_b.gather(1, idx_b)` exactly. Every other parameter is bitwise unchanged.
- In incremental mode, each row of a weight at $b$ bits has at most $2^b$ distinct values, and every value appears in that row's `lut_b`.
- Calling `set_precision` for 8, then 3, then 8 bits gives the same weights as calling it for 8 bits once. The operation depends only on the artifact, not on the model's current weights.
- `bits` outside the artifact's range raises `PrecisionError`, and so does an artifact whose module list names a missing module or a wrong shape. In both cases the model's weights are unchanged.
- In standalone mode, the weights come from `indices_<b>`, not from a shifted parent. The test builds an artifact whose standalone and shifted indices differ, and checks which one was used.
- `restore_weights(snapshot_weights(...))` returns the model to bitwise-identical weights.
- Artifact tensors stay on the CPU after the call.
