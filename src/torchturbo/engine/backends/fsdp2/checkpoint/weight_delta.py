"""Bit-exact sparse deltas for locally exported HF weights."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


_INTEGER_VIEW = {
    1: torch.uint8,
    2: torch.uint16,
    4: torch.uint32,
    8: torch.uint64,
}


@dataclass
class WeightDeltaEntry:
    name: str
    indices: torch.Tensor
    values: torch.Tensor


@dataclass
class WeightDelta:
    is_seed: bool
    full_state: dict[str, torch.Tensor] | None = None
    entries: dict[str, WeightDeltaEntry] = field(default_factory=dict)
    changed_elements: int = 0
    total_elements: int = 0

    @property
    def changed_ratio(self) -> float:
        if self.total_elements == 0:
            return 0.0
        return self.changed_elements / self.total_elements


def diff_state_dict(
    baseline: dict[str, torch.Tensor],
    current: dict[str, torch.Tensor],
) -> WeightDelta:
    """Return only the elements whose bits changed between two state dicts."""
    if baseline.keys() != current.keys():
        raise ValueError("HF delta state dicts have different parameter sets")

    entries: dict[str, WeightDeltaEntry] = {}
    changed_elements = 0
    total_elements = 0
    for name, old_tensor in baseline.items():
        new_tensor = current[name]
        if old_tensor.shape != new_tensor.shape:
            raise ValueError(f"HF delta shape mismatch for {name}")
        if old_tensor.dtype != new_tensor.dtype:
            raise ValueError(f"HF delta dtype mismatch for {name}")

        old_flat = old_tensor.detach().contiguous().reshape(-1)
        new_flat = new_tensor.detach().contiguous().reshape(-1)
        integer_dtype = _INTEGER_VIEW.get(old_flat.element_size())
        if integer_dtype is None:
            raise ValueError(
                f"Unsupported HF delta element size {old_flat.element_size()}"
            )

        mask = old_flat.view(integer_dtype) != new_flat.view(integer_dtype)
        indices = mask.nonzero(as_tuple=False).view(-1)
        numel = old_flat.numel()
        total_elements += numel
        if indices.numel():
            changed_elements += int(indices.numel())
            entries[name] = WeightDeltaEntry(
                name=name,
                indices=indices,
                values=new_flat.index_select(0, indices),
            )

    return WeightDelta(
        is_seed=False,
        entries=entries,
        changed_elements=changed_elements,
        total_elements=total_elements,
    )


def apply_state_dict_delta(
    baseline: dict[str, torch.Tensor],
    delta: WeightDelta,
) -> dict[str, torch.Tensor]:
    """Apply a sparse delta to a copied baseline and return the new state."""
    if delta.is_seed:
        if delta.full_state is None:
            raise ValueError("Seed delta does not contain full weights")
        return {name: tensor.clone() for name, tensor in delta.full_state.items()}

    result = {name: tensor.clone() for name, tensor in baseline.items()}
    for entry in delta.entries.values():
        result[entry.name].reshape(-1).index_copy_(
            0, entry.indices, entry.values
        )
    return result
