# Spec 0006: Artifact Store

- Status: Proposed
- Wave: [0](index.md)
- Implements: [ADR 0005](../../adr/0005-artifact-format-and-simulated-inference.md) (on-disk layout, manifest, atomic writes), [ADR 0002](../../adr/0002-project-layout-and-architecture.md) (caching)

This spec defines how Fisher diagonals and quantized models are written to and read from `outputs/cache/`. It covers the manifest schema, the atomic write protocol, the checks a cache lookup performs, and the errors a failed lookup raises. Key computation lives in spec 0002.

## Files

The store is split into the manifest schema and the read/write functions. Neither module imports `transformers` or Hydra.

| File | Contents |
| --- | --- |
| `artifacts/keys.py` | Snapshots and keys (spec 0002) |
| `artifacts/manifest.py` | `ModuleEntry`, `FisherManifest`, `QuantizedManifest`, `ArtifactStats` |
| `artifacts/store.py` | `ArtifactStore`, `QuantizedArtifact`, errors, and the atomic writer |

## On-disk layout

The layout is ADR 0005's, including the optional `stats.json`. The directory name is the first 16 hex characters of the full key.

```
outputs/cache/
├── fisher/<key16>/
│   ├── fisher.safetensors          # {module_name: float32 [m, n]}
│   ├── losses.safetensors          # {"losses": float32 [N]} per-sequence calibration NLL
│   └── manifest.json
└── quantized/<key16>/
    ├── indices.safetensors         # incremental: {module_name: uint8 [m, n]} at parent_bits
    ├── indices_<b>.safetensors     # standalone only: one file per bit-width
    ├── lut_<b>.safetensors         # {module_name: float16 [m, 2**b]}, one file per b in [seed_bits, parent_bits]
    ├── stats.json                  # per-module, per-bit relative error; absent is allowed
    └── manifest.json
```

`losses.safetensors` lets the acceptance run (spec 0011) check the calibration NLL without recomputing the Fisher.

## Manifest schema

The manifests are pydantic models with the same frozen, extra-forbidding base as the config schemas (spec 0002). One schema per artifact kind keeps the fields that apply to only one kind out of the other.

```python
class ModuleEntry(FrozenModel):
    name: str
    shape: tuple[PositiveInt, PositiveInt]            # [m, n]

class ManifestBase(FrozenModel):
    schema_version: int
    key: str                                           # full 64-hex SHA-256
    model_id: str
    revision: str
    modules: list[ModuleEntry]                         # discovery order
    config_snapshot: dict[str, JsonValue]
    device: str                                        # device type that produced the artifact: "cuda" or "cpu"
    versions: dict[str, str]
    created_at: datetime                               # UTC

class FisherManifest(ManifestBase):
    kind: Literal["fisher"]
    num_sequences: PositiveInt
    seq_len: PositiveInt
    mean_loss: float
    seconds: float

class QuantizedManifest(ManifestBase):
    kind: Literal["quantized"]
    mode: Literal["incremental", "standalone"]
    seed_bits: int
    parent_bits: int
    parent_key: str                                    # full Fisher key
    seconds: float

class ArtifactStats(FrozenModel):
    relative_error: dict[str, dict[int, float]]        # module -> bits -> J / sum(f w^2)
    lloyd_iterations: dict[str, int]
```

`key` is the full SHA-256 of `config_snapshot` (ADR 0005). `parent_key` also holds a full key, never the 16-character directory name.

## Interface

`ArtifactStore` is a thin object bound to a cache directory. The pipelines (spec 0009) use it as a get-or-compute cache for the Fisher, and as a write-once, read-many store for quantized artifacts.

```python
class ArtifactNotFoundError(FileNotFoundError): ...
class ArtifactMismatchError(RuntimeError): ...

@dataclass(frozen=True)
class QuantizedArtifact:
    """A loaded quantized artifact; every tensor is on the CPU (ADR 0005)."""
    manifest: QuantizedManifest
    indices: dict[int, dict[str, torch.Tensor]]        # stored bits -> module -> uint8 [m, n]
    luts: dict[int, dict[str, torch.Tensor]]           # bits -> module -> float16 [m, 2**bits]
    stats: ArtifactStats | None

    @property
    def module_names(self) -> list[str]: ...

class ArtifactStore:
    def __init__(self, cache_dir: Path) -> None: ...

    def fisher_dir(self, key: str) -> Path: ...        # cache_dir / "fisher" / key[:16]
    def quantized_dir(self, key: str) -> Path: ...

    def has_fisher(self, key: str) -> bool: ...
    def save_fisher(self, key: str, snapshot: dict[str, JsonValue], result: FisherResult, meta: FisherMeta) -> Path: ...
    def load_fisher(self, key: str, snapshot: dict[str, JsonValue], modules: Sequence[ModuleEntry]) -> dict[str, torch.Tensor]: ...

    def has_quantized(self, key: str) -> bool: ...
    def save_quantized(self, key: str, snapshot: dict[str, JsonValue], result: ModelQuantization, meta: QuantizedMeta) -> Path: ...
    def load_quantized(self, key: str, snapshot: dict[str, JsonValue], modules: Sequence[ModuleEntry]) -> QuantizedArtifact: ...
```

`FisherMeta` and `QuantizedMeta` are frozen dataclasses carrying the manifest fields the store cannot derive: the model id and revision, the device, and the counts or mode. The pipelines build them from the run config.

`has_*` returns `True` only if `manifest.json` exists and parses. It does not validate the snapshot; that is `load_*`'s job, so a stale or mismatched artifact is reported loudly instead of being silently recomputed on top of.

## Atomic writes

Every artifact is written into a temporary sibling directory and moved into place with one rename, which is atomic on a single filesystem. A crash at any point leaves either no artifact or a complete one.

```python
@contextmanager
def _atomic_directory(final: Path) -> Iterator[Path]:
    """Yield a temporary directory next to `final`, and rename it into place on success."""
    # An existing final directory means another run already wrote this key. Keys are content
    # hashes, so the existing artifact is equivalent, and overwriting it would only add risk.

    if final.exists():
        raise FileExistsError(f"artifact already exists: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{final.name}.", suffix=".tmp", dir=final.parent))
    try:
        yield tmp
        os.replace(tmp, final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
```

`save_*` writes the tensor files first and `manifest.json` last, inside the temporary directory. Tensors are made contiguous and moved to the CPU, serialized with `safetensors.torch.save`, and written with `Path.write_bytes`; the bytes API is used because its signatures are fully typed under strict pyright. Leftover `.*.tmp` directories from a killed process are never read, because lookups only open `<key16>/manifest.json`.

## Load-time checks

A lookup either returns an artifact that matches the request exactly, or raises. The checks run in this order, and the first failure raises.

| Check | Failure |
| --- | --- |
| The directory and `manifest.json` exist | `ArtifactNotFoundError` naming the path and the command that would create it |
| The manifest parses into the expected schema and `kind` | `ArtifactMismatchError` |
| `schema_version` equals the current version | `ArtifactMismatchError` |
| `key` equals the requested full key (a 16-hex prefix collision guard) | `ArtifactMismatchError` |
| `config_snapshot` equals the requested snapshot | `ArtifactMismatchError`, with the differing top-level keys listed |
| `modules` equals the discovered modules, names and shapes in order | `ArtifactMismatchError` |
| Every expected tensor file exists, and holds exactly the manifest's module names with the expected dtype and shape | `ArtifactMismatchError` |

The module check is what catches a changed `quantizable_modules.pattern` that happens to keep the same count, since that field is not part of the Fisher key (spec 0002). All tensors are loaded on the CPU with `safetensors.torch.load(path.read_bytes())`.

## Verification

Tests for this spec are listed in spec 0010 under `tests/artifacts/`. They use `tmp_path` and small synthetic tensors; no model is loaded.

- A saved Fisher or quantized artifact loads back with bitwise-equal tensors, and a manifest equal to the one written.
- Both layouts round-trip: incremental writes a single `indices.safetensors`, and standalone writes one `indices_<b>` per bit-width.
- An exception raised midway through a save leaves no `<key16>` directory and no temporary directory.
- Saving over an existing key raises `FileExistsError`, and leaves the original untouched.
- Each row of the load-time table raises its documented error when the corresponding condition is violated. The tests edit a manifest on disk, delete a tensor file, rename a module, or change one snapshot field.
- An artifact without `stats.json` loads with `stats=None`.
- Loaded tensors are on the CPU regardless of the device they were saved from.
