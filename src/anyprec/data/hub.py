"""The package's only call to ``datasets.load_dataset`` (spec 0009).

``load_dataset`` is only partially typed, so this module alone silences pyright's unknown-member
check. The text column is narrowed to ``list[str]`` value by value with ``isinstance``.
"""

# pyright: reportUnknownMemberType=false

from collections.abc import Iterable
from typing import cast

import datasets


def load_texts(
    path: str, name: str | None, data_files: dict[str, str] | None, split: str, text_field: str
) -> list[str]:
    """Load one split of a Hub dataset and return its text column.

    :param path: ``datasets.load_dataset`` path.
    :param name: Dataset configuration name, if any.
    :param data_files: Explicit data files, if any.
    :param split: Split to load.
    :param text_field: Column holding the raw text.
    :return: One string per document, in dataset order.
    :raises TypeError: If a value in the column is not a string.
    """
    dataset = datasets.load_dataset(path, name=name, data_files=data_files, split=split)

    # A column is a lazy iterable in recent datasets releases, not a list.

    texts: list[str] = []
    for value in cast("Iterable[object]", dataset[text_field]):
        if not isinstance(value, str):
            raise TypeError(f"{path}[{text_field!r}] holds {type(value).__name__}, not str")
        texts.append(value)
    return texts
