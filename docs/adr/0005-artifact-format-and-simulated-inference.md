# ADR 0005: Quantized Artifact Format and Simulated Inference

- Status: Proposed
- Date: 2026-09-24
- Deciders: Project maintainer

This ADR specifies how quantization results are stored on disk and how a model is run at a chosen bit-width. Because the project measures answer quality rather than latency, inference is *simulated*: the stored indices and codebooks are dequantized into ordinary weight tensors inside the unmodified Hugging Face model.

## Context

The paper's software engine (its Section 5) stores the parent model as bitplanes and uses a custom CUDA kernel, so that running at $b$ bits reads only $b$ bitplanes from memory. That machinery affects speed only; the numbers a model produces depend solely on which centroid each weight is mapped to. The reference implementation already has a quality-only path (`fake_pack` in `any_precision/evaluate/eval.py`) that writes centroids back into a standard model.

[ADR 0003](0003-fisher-weighted-kmeans-methodology.md) produces, per quantizable layer, a `uint8` index matrix at the parent bit-width and one float16 codebook per row per bit-width, with the nested property $a^{(b)} = a^{(B)} \gg (B - b)$.

Two dtype facts matter:

- Granite's checkpoint is bfloat16 (8 significant bits). Two adjacent 8-bit centroids in a row can differ by less than one bfloat16 unit at larger magnitudes, so writing centroids into a bfloat16 model can merge distinct centroids and blur the difference between bit-widths.
- Every bfloat16 value and every float16 value is exactly representable in float32.

## Decision

### On-disk layout

All artifacts live under the cache directory from [ADR 0002](0002-project-layout-and-architecture.md), keyed by configuration hash, in `safetensors` format.

```
outputs/cache/
├── fisher/<fisher_key>/
│   ├── fisher.safetensors          # {module_name: float32 [m, n]}
│   ├── losses.safetensors          # {"losses": float32 [N]}  per-sequence calibration NLL
│   └── manifest.json
└── quantized/<quant_key>/
    ├── indices.safetensors         # {module_name: uint8 [m, n]}  parent-bit indices
    ├── lut_3.safetensors           # {module_name: float16 [m, 8]}
    ├── lut_4.safetensors           # {module_name: float16 [m, 16]}
    ├── ...
    ├── lut_8.safetensors           # {module_name: float16 [m, 256]}
    ├── stats.json                  # optional: per-module, per-bit Fisher-weighted relative error
    └── manifest.json
```

Directory names are the first 16 hex characters of the key; the manifest stores the full key.

`stats.json` records $J^{(b)} / \sum_j f_j w_j^2$ for every module and bit-width, computed with the float16 codebooks. It lets the ADR 0003 properties be checked on a finished artifact, without re-running k-means. It is optional, so an artifact without it is still valid. The per-sequence calibration losses serve the same purpose for the Fisher.

Tensor keys are fully qualified module names as returned by `named_modules()` (for example `model.layers.0.self_attn.q_proj`), so an artifact maps onto a freshly loaded model without guessing. In `standalone` mode ([ADR 0003](0003-fisher-weighted-kmeans-methodology.md), Section 5) indices are not nested, so the layout instead stores one `indices_<b>.safetensors` per bit-width; the manifest's `mode` field says which layout applies.

Weights are stored as one `uint8` index per weight, not as bitplanes. Bitplanes would change nothing about model outputs and are out of scope; bits-per-weight is computed analytically ([ADR 0004](0004-evaluation-protocol.md)) rather than read from file sizes.

### Manifest

Each artifact directory has a `manifest.json` validated by a pydantic schema:

| Field | Content |
| --- | --- |
| `schema_version` | Artifact format version; a mismatch on load is a hard error |
| `kind` | `fisher` or `quantized` |
| `mode` | `incremental` or `standalone` (quantized only) |
| `model_id`, `revision` | Source checkpoint |
| `seed_bits`, `parent_bits` | Bit-width range (quantized only) |
| `modules` | Ordered list of module names with their `[m, n]` shapes |
| `config_snapshot` | The configuration subtrees that determine this artifact |
| `key` | Full SHA-256 of `config_snapshot`; guards against collisions of the 16-character directory name |
| `parent_key` | For a quantized artifact, the Fisher key it was built from |
| `device` | Device type that produced the artifact (`cuda` or `cpu`); k-means++ draws differ between the two generators |
| `versions` | `anyprec`, `torch`, `transformers` versions |
| `created_at` | ISO-8601 UTC timestamp |

A cache lookup accepts an artifact only if `schema_version` matches, `key` and `config_snapshot` equal the requested ones, and `modules` equals the modules discovered in the loaded model.

### Writing is atomic

An artifact is written into a temporary sibling directory and renamed into place only after every tensor file and the manifest are complete. An interrupted run therefore never leaves a partial artifact that a later cache lookup could mistake for a valid one.

### Simulated inference

A model is loaded once with `from_pretrained` in `eval_dtype` (float32 by default). The artifact stays in CPU memory; setting a bit-width moves one layer's indices and codebook to the weight's device at a time, and overwrites that layer's weight in place. Keeping the artifact off the GPU saves about 0.5 GB of VRAM, which the evaluation's two float32 models and their logits need.

$$
\hat W^{(b)}_{r,j} = C^{(b)}_{r}\left[\, a^{(B)}_{r,j} \gg (B - b) \,\right].
$$

```python
@torch.no_grad()
def set_precision(model, artifact, bits):
    """Overwrite every quantized layer's weight with its bits-wide dequantization."""
    modules = dict(model.named_modules())
    shift = artifact.parent_bits - bits
    for name in artifact.module_names:
        weight = modules[name].weight                                          # [m, n] on the GPU
        idx = (artifact.indices[name] >> shift).to(weight.device).long()       # uint8 shift on the CPU -> [m, n]
        lut = artifact.luts[bits][name].to(weight.device, weight.dtype)        # [m, 2**bits]
        weight.copy_(lut.gather(1, idx))                                       # [m, n]
```

Switching bit-width is a gather per layer, so a sweep over $b = 3,\dots,8$ reuses one loaded model. Unquantized parameters (embedding, LM head, norms) are never touched. For the reference model, a separate model instance is loaded and left unmodified ([ADR 0004](0004-evaluation-protocol.md) runs both side by side).

Float32 is the default evaluation dtype because it represents both the original bfloat16 weights and the float16 centroids exactly, so reference and quantized models differ only in the centroid assignment. At 350M parameters a float32 model is about 1.4 GB, which fits two copies comfortably in the laptop's 6 GB of VRAM. `eval_dtype: bfloat16` remains available for speed, with the caveat above.

## Consequences

The format is simple, inspectable, and sufficient for every quality question the project asks, and it keeps the Hugging Face model unmodified.

- Positive: no custom modules or kernels; the nested property is visible directly in the data (one shift); artifacts are self-describing and safe against partial writes; evaluation runs without dtype confounds.
- Negative: storage is `uint8` per weight regardless of bit-width, so files are larger than a packed format would be; this is irrelevant to the measured quality and is accounted for analytically.
- Out of scope: bitplane packing, the CUDA kernel, `generate()` speed, and any serving integration.
