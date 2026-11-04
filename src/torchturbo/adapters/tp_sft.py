"""Shared exact-count TP SFT/PT support for HF-style adapter trainers."""

from __future__ import annotations

from typing import Any

import torch

from ..engine.core import EpochDrivenDataLoader


class TpSftTrainerSupport:
    """TP SFT/PT loss accounting shared by the adapter trainer mixins.

    Mixed in ahead of a concrete HF-style trainer; non-TP behavior reaches
    the framework through ``super()``. Subclasses provide ``_tp_engine()``
    returning the active TorchTurbo engine.
    """

    _tp_sft_enabled: bool = False
    _tp_main_loss: torch.Tensor | None = None

    def _tp_engine(self) -> Any:
        """Return the active TorchTurbo engine for TP accounting."""
        raise NotImplementedError

    def _activate_tp_sft(self, engine: Any, stage: str | None) -> None:
        if engine.get_tp_info()["tp_size"] == 1:
            return
        if stage not in {"sft", "pt"}:
            raise ValueError("TP adapters support SFT/PT training only")
        if self.model.is_diffusion_model:
            raise ValueError("TP adapters do not support diffusion training")
        if engine.config.gbs is None:
            raise ValueError("TP adapters require explicit tt_gbs")
        self._tp_sft_enabled = True
        # The prepared loss consumes the exact step count, so HF must not divide GA.
        self.model_accepts_loss_kwargs = True

    def _get_num_items_in_batch(
        self, batch_samples: list[dict[str, Any]], device: torch.device,
    ) -> int | torch.Tensor | None:
        if not self._tp_sft_enabled:
            return super()._get_num_items_in_batch(batch_samples, device)
        cleaned = [dict(batch) for batch in batch_samples]
        for batch in cleaned:
            for key in ("compute_loss_func", "loss_scale", "logits_to_keep"):
                batch.pop(key, None)
        prepared, count = self._tp_engine().prepare_loss_step(
            self.model, cleaned, device
        )
        batch_samples[:] = prepared
        return count

    def get_total_train_batch_size(self, args: Any) -> int:
        """Report the nominal batch using logical data replicas for TP."""
        if not self._tp_sft_enabled:
            return super().get_total_train_batch_size(args)
        return (
            self._train_batch_size * args.gradient_accumulation_steps
            * self._tp_engine().get_dp_info()["dp_size"]
        )

    def num_examples(self, dataloader: Any) -> int:
        """Use the engine sampler's epoch item estimate, including block padding."""
        if self._tp_sft_enabled and isinstance(dataloader, EpochDrivenDataLoader):
            return dataloader.sampler.total_size
        return super().num_examples(dataloader)

    def _track_num_input_tokens(self, inputs: dict[str, Any]) -> None:
        previous = self.state.num_input_tokens_seen
        super()._track_num_input_tokens(inputs)
        if self._tp_sft_enabled:
            engine = self._tp_engine()
            replicas = engine.get_sp_info()["sp_size"] * engine.get_tp_info()["tp_size"]
            increment = self.state.num_input_tokens_seen - previous
            self.state.num_input_tokens_seen = previous + increment // replicas
