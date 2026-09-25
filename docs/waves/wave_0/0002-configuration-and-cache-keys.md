# Spec 0002: Configuration and Cache Keys

- Status: Proposed
- Wave: [0](index.md)
- Implements: [ADR 0002](../../adr/0002-project-layout-and-architecture.md) (Hydra and pydantic, caching), [ADR 0006](../../adr/0006-hadamard-rotation.md) (rotation group)

This spec defines every Hydra YAML file, the pydantic schemas that validate them, how a `DictConfig` becomes a frozen run config, how cache keys are computed, and the small utilities (seeding, dtypes, versions) the rest of the package relies on.

## Hydra files

The groups follow ADR 0002. The `seed`, `device`, and `modes` fields come from ADR 0002, `quantizer.row_chunk` from ADR 0003, and `max_chunks` from ADR 0004.

```yaml
# configs/quantize.yaml
defaults:
  - model: granite_4_0_350m
  - calibration: c4
  - quantizer: kmeans_iu
  - rotation: none
  - _self_

seed: 0
device: cuda

output:
  base_dir: outputs
  cache_dir: ${output.base_dir}/cache

hydra:
  job:
    chdir: false
  run:
    dir: ${output.base_dir}/quantize/${now:%Y-%m-%d}/${now:%H-%M-%S}
```

```yaml
# configs/evaluate.yaml
defaults:
  - model: granite_4_0_350m
  - calibration: c4
  - quantizer: kmeans_iu
  - rotation: none
  - eval: default
  - _self_

seed: 0
device: cuda
modes: [incremental, standalone]

output:
  base_dir: outputs
  cache_dir: ${output.base_dir}/cache

hydra:
  job:
    chdir: false
  run:
    dir: ${output.base_dir}/evaluate/${now:%Y-%m-%d}/${now:%H-%M-%S}
```

Evaluation composes the same `model`, `calibration`, `quantizer`, and `rotation` groups as quantization. It needs them to recompute the cache keys of the artifacts it evaluates, and it never retrains anything.

```yaml
# configs/model/granite_4_0_350m.yaml
model_id: ibm-granite/granite-4.0-350m
revision: bd8a1497065c0d6ba1ef19af6b0d2b14bacf71c2   # Hub commit of 2025-10-23; part of the Fisher key
dtype: bfloat16
eval_dtype: float32
quantizable_modules:
  pattern: '^model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|shared_mlp\.(input|output)_linear)$'
  expected_count: 168
```

```yaml
# configs/calibration/c4.yaml
path: allenai/c4
data_files: {train: en/c4-train.00000-of-01024.json.gz}
split: train
text_field: text
num_sequences: 100
seq_len: 512
seed: 0
```

```yaml
# configs/quantizer/kmeans_iu.yaml
mode: incremental          # or standalone (ADR 0003, Section 5)
seed_bits: 3
parent_bits: 8
seed: 0                    # k-means++ generator seed
lloyd_max_iter: 50
empty_eps: 1.0e-12
row_chunk: 1024            # rows per kernel call; bounds GPU memory
```

`configs/rotation/none.yaml` and `configs/rotation/hadamard.yaml` are exactly the files in ADR 0006.

```yaml
# configs/eval/default.yaml
chunk_len: 2048
max_chunks: null           # cap chunks per dataset for smoke runs; null = all
lm_head_chunk_tokens: 256  # positions per LM-head slice (a power of two); null = all positions at once
datasets:
  wikitext2: {path: Salesforce/wikitext, name: wikitext-2-raw-v1, split: test, text_field: text, joiner: "\n\n"}
  c4: {path: allenai/c4, data_files: {validation: en/c4-validation.00000-of-00008.json.gz},
       split: validation, text_field: text, joiner: " ", max_tokens: 524288}
kl:
  dataset: wikitext2
  quantile: 0.99
bits: [3, 4, 5, 6, 7, 8]
```

## Pydantic schemas

All schemas live in `anyprec/config/schemas.py`. Every model is frozen and forbids extra keys, so a typo in YAML or on the command line is a validation error rather than a silently ignored field.

```python
class FrozenModel(BaseModel):
    """Base for every config, manifest (spec 0006), and results (spec 0008) schema: immutable, unknown keys rejected."""
    model_config = ConfigDict(frozen=True, extra="forbid")

DTypeName = Literal["bfloat16", "float16", "float32"]

class QuantizableModules(FrozenModel):
    pattern: str                       # validated: re.compile succeeds
    expected_count: PositiveInt

class ModelConfig(FrozenModel):
    model_id: str
    revision: str
    dtype: DTypeName
    eval_dtype: DTypeName
    quantizable_modules: QuantizableModules

class CalibrationConfig(FrozenModel):
    path: str
    data_files: dict[str, str] | None = None
    name: str | None = None
    split: str
    text_field: str
    num_sequences: PositiveInt
    seq_len: PositiveInt
    seed: int

class QuantizerConfig(FrozenModel):
    mode: Literal["incremental", "standalone"]
    seed_bits: int = Field(ge=1, le=8)
    parent_bits: int = Field(ge=1, le=8)     # uint8 indices cap the parent at 8 bits
    seed: int
    lloyd_max_iter: PositiveInt
    empty_eps: PositiveFloat
    row_chunk: PositiveInt
    # model_validator: seed_bits <= parent_bits

class RotationNone(FrozenModel):
    kind: Literal["none"]

class RotationHadamard(FrozenModel):
    kind: Literal["hadamard"]
    axis: Literal["in_features"]
    randomized_signs: bool
    seed: int

RotationConfig = Annotated[RotationNone | RotationHadamard, Field(discriminator="kind")]

class OutputConfig(FrozenModel):
    base_dir: Path
    cache_dir: Path

class EvalDatasetConfig(FrozenModel):
    path: str
    name: str | None = None
    data_files: dict[str, str] | None = None
    split: str
    text_field: str
    joiner: str
    max_tokens: PositiveInt | None = None

class KLConfig(FrozenModel):
    dataset: Literal["wikitext2", "c4"]
    quantile: float = Field(gt=0.0, lt=1.0)

class EvalConfig(FrozenModel):
    chunk_len: PositiveInt
    max_chunks: PositiveInt | None = None
    lm_head_chunk_tokens: PositiveInt | None = 256
    datasets: dict[Literal["wikitext2", "c4"], EvalDatasetConfig]
    kl: KLConfig
    bits: list[int]
    # field_validator: lm_head_chunk_tokens is None or a power of two
    # model_validator: kl.dataset in datasets; bits sorted, unique, each in [1, 8]

class QuantizeRunConfig(FrozenModel):
    seed: int
    device: str                         # "cuda", "cuda:0", or "cpu"
    model: ModelConfig
    calibration: CalibrationConfig
    quantizer: QuantizerConfig
    rotation: RotationConfig
    output: OutputConfig

class EvaluateRunConfig(FrozenModel):
    seed: int
    device: str
    modes: list[Literal["incremental", "standalone"]]   # non-empty, unique
    model: ModelConfig
    calibration: CalibrationConfig
    quantizer: QuantizerConfig
    rotation: RotationConfig
    eval: EvalConfig
    output: OutputConfig
    # model_validator: every eval.bits entry lies in [quantizer.seed_bits, quantizer.parent_bits]
```

The pydantic models are the single source of truth for test fixtures (ADR 0002). Tests build configs by instantiating these classes, never by writing dictionaries by hand.

## From Hydra to pydantic

`anyprec/config/loading.py` is the only library module that sees a `DictConfig`. It resolves interpolations, drops the `hydra` key, and validates.

```python
def load_quantize_config(cfg: DictConfig) -> QuantizeRunConfig:
    """Resolve a composed Hydra config and validate it into a frozen run config."""
    # Resolve ${...} interpolations, then drop Hydra's own runtime node, which is not part of the schema.

    container = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(container, dict):
        raise TypeError("the composed Hydra config must be a mapping")
    container.pop("hydra", None)
    return QuantizeRunConfig.model_validate(container)
```

`load_evaluate_config` is identical, with `EvaluateRunConfig`.

## Rotation resolution

`anyprec/rotation/resolve.py` implements `resolve_rotation` exactly as ADR 0006 specifies. Both pipelines call it immediately after config validation, and before any model or dataset is loaded (spec 0009).

```python
def resolve_rotation(cfg: RotationConfig) -> None:
    """Accept rotation=none and fail fast on rotation=hadamard, which is not implemented."""
    match cfg:
        case RotationNone():
            return None
        case RotationHadamard():
            raise NotImplementedError(
                "Hadamard rotation is specified in ADR 0006 but not implemented yet; use rotation=none."
            )
```

## Cache keys

ADR 0002 keys each cached artifact by a SHA-256 hash of exactly the configuration it depends on. The functions live in `anyprec/artifacts/keys.py` and build on `anyprec/utils/hashing.py`.

```python
FISHER_SCHEMA_VERSION: int = 1
QUANTIZED_SCHEMA_VERSION: int = 1

def canonical_json(value: JsonValue) -> str:
    """Serialize with sorted keys and no whitespace, so equal configs hash equally."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

def sha256_key(value: JsonValue) -> str:
    """Full 64-character hex digest of the canonical JSON."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()

def fisher_snapshot(model: ModelConfig, calibration: CalibrationConfig, rotation: RotationConfig) -> dict[str, JsonValue]:
    """The exact configuration a Fisher artifact depends on (ADR 0002, caching table)."""
    return {
        "schema_version": FISHER_SCHEMA_VERSION,
        "model": {"model_id": model.model_id, "revision": model.revision, "dtype": model.dtype},
        "calibration": calibration.model_dump(mode="json"),
        "rotation": TypeAdapter(RotationConfig).dump_python(rotation, mode="json"),
    }

def quantized_snapshot(fisher_key: str, quantizer: QuantizerConfig) -> dict[str, JsonValue]:
    """The exact configuration a quantized artifact depends on."""
    return {
        "schema_version": QUANTIZED_SCHEMA_VERSION,
        "fisher_key": fisher_key,
        "quantizer": quantizer.model_dump(mode="json"),
    }

def fisher_key(...) -> str:      # sha256_key(fisher_snapshot(...))
def quantized_key(...) -> str:   # sha256_key(quantized_snapshot(...))
```

Four rules follow from these definitions:

- **The key sees only what changes the result.** `model.eval_dtype` and `quantizable_modules` are excluded from the Fisher key. The dtype is only used at evaluation. A changed pattern or count fails discovery (spec 0003) before any cache lookup.
- **Changing the quantizer never invalidates the Fisher.** Changing `quantizer.mode` or `quantizer.seed` produces a new quantized key and reuses the cached Fisher.
- **Directory names use the first 16 hex characters of the key.** The manifest stores the full key and the snapshot, and a lookup compares both (spec 0006).
- **`device` is not part of any key.** k-means++ draws differ between CPU and CUDA generators, so an artifact is reproducible only on the device type that produced it. The manifest records the device (spec 0006).

## Utilities

These are small helpers in `anyprec/utils/`. They are listed here because every other spec depends on them.

| Function | Signature | Behaviour |
| --- | --- | --- |
| `seed_everything` | `(seed: int) -> None` | Seeds `random`, `numpy.random`, the default CPU generator, and every CUDA device |
| `stable_seed` | `(seed: int, name: str) -> int` | First 8 bytes of `sha256(f"{seed}:{name}")` as an unsigned int, masked to 63 bits so `torch.Generator.manual_seed` accepts it |
| `torch_dtype` | `(name: DTypeName) -> torch.dtype` | A lookup table; unknown names are impossible after validation |
| `resolve_device` | `(name: str) -> torch.device` | Parses `cfg.device`; requesting CUDA without CUDA raises, rather than falling back to the CPU (spec 0009) |
| `library_versions` | `() -> dict[str, str]` | Versions of `anyprec`, `torch`, `transformers`, `datasets`, and the CUDA runtime (or `"cpu"`) |

## Verification

Tests for this spec are listed in spec 0010 under `tests/config/`, `tests/utils/`, `tests/rotation/`, and `tests/artifacts/test_keys.py`.

- Every YAML file composes with Hydra's `compose` API (no `@hydra.main`) and validates into its run config.
- Invalid inputs raise `ValidationError`: an unknown key, `seed_bits > parent_bits`, `parent_bits = 9`, an eval bit outside the quantizer's range, `kl.dataset` missing from `datasets`, or `lm_head_chunk_tokens` of 0, 300, or -256. `null` and 1, 256, and 2048 are accepted.
- `rotation=hadamard` validates as configuration, and `resolve_rotation` raises `NotImplementedError` for it.
- Keys are stable, insensitive to dictionary key order, and sensitive to exactly the fields in the snapshots. A parametrized table perturbs one field at a time, covering model, calibration, and quantizer fields plus one field outside every snapshot, and asserts which keys change.
- `stable_seed` is deterministic across processes (a pinned expected value), and different names give different seeds.
