"""Library versions recorded in manifests and results (ADR 0002, Reproducibility)."""

from importlib.metadata import PackageNotFoundError, version

import torch

_PACKAGES: tuple[str, ...] = ("anyprec", "torch", "transformers", "datasets")


def library_versions() -> dict[str, str]:
    """Collect the versions that determine numerical results.

    Versions are read from package metadata, so ``transformers`` and ``datasets`` are never
    imported here (spec 0001, heavy dependencies stay at the boundary).

    :return: Package name to version, plus ``"cuda"`` (the CUDA runtime version, or ``"cpu"``).
    """
    versions: dict[str, str] = {}
    for package in _PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    cuda: str | None = torch.version.cuda
    versions["cuda"] = cuda if cuda is not None and torch.cuda.is_available() else "cpu"
    return versions
