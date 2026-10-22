"""Ownership helpers for checkpoint tests whose models have no expert units."""

from torch import nn

from torchturbo.engine.backends.fsdp2.parallel_context import ParameterOwnership


class AutoRegisteringOwnership(ParameterOwnership):
    """Register models without expert units on first use.

    ``DcpCheckpoint`` asks the ownership registry for the expert names of
    every model it saves or loads. Tests that exercise plain, EP-free modules
    would otherwise have to register each saved and restored model by hand.
    """

    def expert_fqns(self, model: nn.Module) -> frozenset[str]:
        if model not in self._models:
            self.register(model, set())
        return super().expert_fqns(model)
