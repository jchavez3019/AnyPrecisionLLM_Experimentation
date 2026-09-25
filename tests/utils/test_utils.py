"""Tests for the layer-0 utilities (spec 0002, utilities table)."""

import subprocess
import sys
from importlib.metadata import PackageNotFoundError

import pytest
import torch

import anyprec.utils.versions as versions_module
from anyprec.utils.devices import resolve_device
from anyprec.utils.dtypes import torch_dtype
from anyprec.utils.hashing import canonical_json, sha256_key, stable_seed
from anyprec.utils.seeding import seed_everything
from anyprec.utils.versions import library_versions


def test_canonical_json_is_insensitive_to_key_order() -> None:
    """
    Given: two dictionaries with the same content inserted in different orders.
    When: both are serialized canonically and hashed.
    Then: the strings and the keys are identical, with no whitespace.
    """
    first = canonical_json({"b": 1, "a": {"y": [1, 2], "x": None}})
    second = canonical_json({"a": {"x": None, "y": [1, 2]}, "b": 1})

    assert first == second == '{"a":{"x":null,"y":[1,2]},"b":1}'
    assert sha256_key({"b": 1, "a": 2}) == sha256_key({"a": 2, "b": 1})


def test_stable_seed_matches_pinned_value_and_fits_63_bits() -> None:
    """
    Given: the base seed 0 and a Granite module name.
    When: stable_seed derives a generator seed.
    Then: it equals the value computed independently from SHA-256, and fits in 63 bits.
    """
    seed = stable_seed(0, "model.layers.0.self_attn.q_proj")

    assert seed == 5976411218850630258
    assert 0 <= seed < 2**63


def test_stable_seed_is_identical_in_a_fresh_interpreter() -> None:
    """
    Given: a seed derived in this process.
    When: the same seed is derived in a new Python process with a different hash salt.
    Then: both are equal, unlike Python's built-in hash().
    """
    code = "from anyprec.utils.hashing import stable_seed; print(stable_seed(7, 'chunk3'))"
    output = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env={"PYTHONHASHSEED": "123"},
    ).stdout.strip()

    assert int(output) == stable_seed(7, "chunk3")


def test_stable_seed_differs_across_names_and_base_seeds() -> None:
    """
    Given: several names and two base seeds.
    When: seeds are derived for every combination.
    Then: all of them are distinct.
    """
    names = [f"model.layers.{i}.self_attn.q_proj" for i in range(28)] + ["chunk0", "chunk1"]

    seeds = {stable_seed(base, name) for base in (0, 1) for name in names}

    assert len(seeds) == 2 * len(names)


def test_seed_everything_makes_torch_draws_repeatable() -> None:
    """
    Given: the global torch generator seeded twice with the same value.
    When: a random tensor is drawn after each seeding.
    Then: the two tensors are identical.
    """
    seed_everything(11)
    first = torch.rand(5)
    seed_everything(11)
    second = torch.rand(5)

    assert torch.equal(first, second)


def test_torch_dtype_maps_every_configured_name() -> None:
    """
    Given: each dtype name accepted by the config schemas.
    When: it is converted.
    Then: the matching torch dtype is returned.
    """
    assert torch_dtype("bfloat16") is torch.bfloat16
    assert torch_dtype("float16") is torch.float16
    assert torch_dtype("float32") is torch.float32


def test_resolve_device_raises_when_cuda_requested_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Given: a machine where CUDA reports unavailable.
    When: device "cuda" is resolved.
    Then: a RuntimeError names the explicit CPU alternative, rather than falling back silently.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="device=cpu"):
        resolve_device("cuda")
    assert resolve_device("cpu") == torch.device("cpu")


def test_library_versions_reports_packages_and_cuda_runtime() -> None:
    """
    Given: the project environment.
    When: library versions are collected.
    Then: every recorded package has a version string, and the CUDA entry is present.
    """
    versions = library_versions()

    assert set(versions) == {"anyprec", "torch", "transformers", "datasets", "cuda"}
    assert versions["torch"].startswith("2.")
    assert all(value for value in versions.values())


def test_library_versions_marks_missing_package_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Given: package metadata in which "datasets" is not installed.
    When: library versions are collected.
    Then: "datasets" is reported as "not installed", and the other packages keep their versions.
    """

    def fake_version(package: str) -> str:
        """Report every package as version 1.0, except datasets, which is missing.

        :param package: Distribution name.
        :return: A fixed version string.
        :raises PackageNotFoundError: For ``datasets``.
        """
        if package == "datasets":
            raise PackageNotFoundError(package)
        return "1.0"

    # Patch the name library_versions resolves, so no real distribution metadata is read.

    monkeypatch.setattr(versions_module, "version", fake_version)

    versions = library_versions()

    assert versions["datasets"] == "not installed"
    assert versions["torch"] == "1.0"
