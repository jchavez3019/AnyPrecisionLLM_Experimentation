"""Offline stand-ins for the Hub loaders that both pipelines call (spec 0010, ``offline_loaders``).

Each stand-in returns a real object of the type the pipeline expects: a genuine tiny Granite
model, a real ``Encoder``, and a list of strings. Every call is recorded, so a test can prove
that a run failed before it loaded anything.
"""

from dataclasses import dataclass, field

import torch
from torch import nn
from transformers import GraniteMoeHybridForCausalLM

from anyprec.config.schemas import ModelConfig
from anyprec.data.calibration import Encoder
from tests import factories


@dataclass
class OfflineLoaders:
    """Record-keeping replacements for ``load_model``, the tokenizer, and ``load_texts``.

    :param calls: Names of the loaders called so far, in call order.
    :param model_seeds: Seeds of successive ``load_model`` calls; when exhausted, seed 0 is used,
        so every model of a run holds the same weights unless a test asks otherwise.
    """

    calls: list[str] = field(default_factory=list[str])
    model_seeds: list[int] = field(default_factory=list[int])

    def load_model(
        self, cfg: ModelConfig, dtype: torch.dtype, device: torch.device
    ) -> GraniteMoeHybridForCausalLM:
        """Build the tiny model in the requested dtype and device.

        :param cfg: The model config; only its type matters here.
        :param dtype: Parameter dtype.
        :param device: Target device.
        :return: A fresh tiny model in ``eval()`` mode.
        """
        self.calls.append("load_model")
        seed = self.model_seeds.pop(0) if self.model_seeds else 0
        model = factories.tiny_model(seed)

        # Hugging Face wraps ``to`` in an untyped decorator; the nn.Module method is the same
        # in-place move, with a type pyright can check.

        nn.Module.to(model, device=device, dtype=dtype)
        return model

    def load_tokenizer(self, cfg: ModelConfig) -> None:
        """Stand in for the tokenizer; ``make_encoder`` below never looks at it.

        :param cfg: The model config.
        """
        self.calls.append("load_tokenizer")

    def make_encoder(self, tokenizer: None) -> Encoder:
        """Return the byte-level encoder that matches the tiny model's 256-token vocabulary.

        :param tokenizer: The stand-in tokenizer.
        :return: ``factories.char_encode``.
        """
        self.calls.append("make_encoder")
        return factories.char_encode

    def load_texts(
        self,
        path: str,
        name: str | None,
        data_files: dict[str, str] | None,
        split: str,
        text_field: str,
    ) -> list[str]:
        """Return the fixed in-memory corpus for every dataset and split.

        :param path: Dataset path, recorded in the call log.
        :param name: Dataset configuration name.
        :param data_files: Data files mapping.
        :param split: Split, recorded in the call log.
        :param text_field: Text column.
        :return: ``factories.corpus()``.
        """
        self.calls.append(f"load_texts:{path}:{split}")
        return factories.corpus()
