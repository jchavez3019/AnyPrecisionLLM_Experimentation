"""Mechanical enforcement of the import rules of spec 0001 (spec 0010, layering test)."""

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT: Path = Path(__file__).resolve().parents[1] / "src" / "anyprec"

LAYERS: dict[str, int] = {
    "anyprec.utils": 0,
    "anyprec.config": 1,
    "anyprec.models": 2,
    "anyprec.data": 2,
    "anyprec.rotation": 2,
    "anyprec.quantization.rows": 2,
    "anyprec.quantization.init": 2,
    "anyprec.quantization.lloyd": 2,
    "anyprec.quantization.split": 2,
    "anyprec.sensitivity": 3,
    "anyprec.quantization.layer": 3,
    "anyprec.quantization.model": 3,
    "anyprec.artifacts": 3,
    "anyprec.evaluation.metrics": 4,
    "anyprec.evaluation.bits": 4,
    "anyprec.evaluation.results": 4,
    "anyprec.inference": 4,
    "anyprec.quantization.pipeline": 5,
    "anyprec.evaluation.pipeline": 5,
}

HYDRA_ALLOWED: frozenset[str] = frozenset({"anyprec.config.loading"})
HF_ALLOWED_PREFIXES: tuple[str, ...] = (
    "anyprec.models",
    "anyprec.data",
    "anyprec.quantization.pipeline",
    "anyprec.evaluation.pipeline",
)


def _module_name(path: Path) -> str:
    """Convert a source path to its dotted module name.

    :param path: A ``.py`` file under ``src/anyprec``.
    :return: For example ``anyprec.quantization.layer``, or ``anyprec.config`` for an ``__init__``.
    """
    parts = path.relative_to(PACKAGE_ROOT.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _imports(path: Path) -> list[str]:
    """List every absolute module a source file imports.

    :param path: A ``.py`` file.
    :return: Imported module names, in source order.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None and node.level == 0:
            names.append(node.module)
    return names


def _layer(module: str) -> int | None:
    """Find the layer of a module by its longest matching ``LAYERS`` prefix.

    :param module: Dotted module name.
    :return: The layer, or ``None`` if no prefix matches.
    """
    matches = [p for p in LAYERS if module == p or module.startswith(p + ".")]
    return LAYERS[max(matches, key=len)] if matches else None


def _is_package(module: str) -> bool:
    """Whether a dotted ``anyprec`` name refers to a package rather than a module file.

    :param module: Dotted module name.
    :return: ``True`` if the name resolves to a directory with an ``__init__.py``.
    """
    return (PACKAGE_ROOT.parent.joinpath(*module.split(".")) / "__init__.py").is_file()


SOURCES: list[Path] = sorted(PACKAGE_ROOT.rglob("*.py"))
MODULES: list[Path] = [p for p in SOURCES if p.name != "__init__.py"]
INITS: list[Path] = [p for p in SOURCES if p.name == "__init__.py"]


@pytest.mark.parametrize("path", MODULES, ids=_module_name)
def test_module_imports_respect_layers_and_library_boundaries(path: Path) -> None:
    """
    Given: one library module file.
    When: its imports are parsed.
    Then: it is classified in LAYERS, imports only equal or lower layers, names modules rather
        than packages, and imports hydra, omegaconf, transformers, or datasets only where allowed.
    """
    module = _module_name(path)
    own_layer = _layer(module)
    assert own_layer is not None, f"{module} is missing from LAYERS"

    for imported in _imports(path):
        root = imported.split(".")[0]
        if root == "anyprec":
            # Package imports run a subpackage __init__ and its layer-5 re-exports (spec 0001).

            assert not _is_package(imported), f"{module} imports package {imported}"
            imported_layer = _layer(imported)
            assert imported_layer is not None, f"{module} imports unclassified {imported}"
            assert imported_layer <= own_layer, f"{module} (L{own_layer}) imports {imported}"
        elif root in {"hydra", "omegaconf"}:
            assert module in HYDRA_ALLOWED, f"{module} imports {root}"
        elif root in {"transformers", "datasets"}:
            assert module.startswith(HF_ALLOWED_PREFIXES), f"{module} imports {root}"


@pytest.mark.parametrize("path", INITS, ids=_module_name)
def test_package_init_imports_only_from_its_own_subpackage(path: Path) -> None:
    """
    Given: one package __init__.py.
    When: its anyprec imports are parsed.
    Then: the top-level anyprec/__init__ imports nothing from the package, and every other
        __init__ imports only modules inside its own subpackage.
    """
    package = _module_name(path)
    internal = [name for name in _imports(path) if name.split(".")[0] == "anyprec"]

    if package == "anyprec":
        assert internal == []
    else:
        assert all(name.startswith(package + ".") for name in internal), internal
