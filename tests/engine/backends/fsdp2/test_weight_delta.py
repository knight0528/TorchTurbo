"""Tests for sparse HF weight deltas."""

import torch
from types import SimpleNamespace

from torchturbo.engine.backends.fsdp2.checkpoint.weight_delta import (
    apply_state_dict_delta,
    diff_state_dict,
)
from torchturbo.engine.backends.fsdp2.checkpoint.weight_saver import (
    ShardedParamManager,
)
from torchturbo.engine.backends.fsdp2.checkpoint.manager import (
    FSDP2CheckpointManager,
)


def test_sparse_delta_is_bit_exact_for_bfloat16():
    baseline = {
        "a.weight": torch.arange(8, dtype=torch.bfloat16).reshape(2, 4),
        "b.weight": torch.ones(3, dtype=torch.bfloat16),
    }
    current = {
        "a.weight": baseline["a.weight"].clone(),
        "b.weight": baseline["b.weight"].clone(),
    }
    current["a.weight"][1, 2] = 33
    current["b.weight"][0] = 7

    delta = diff_state_dict(baseline, current)
    restored = apply_state_dict_delta(baseline, delta)

    assert delta.changed_elements == 2
    assert delta.total_elements == 11
    assert delta.changed_ratio == 2 / 11
    assert set(delta.entries) == {"a.weight", "b.weight"}
    for name in current:
        assert torch.equal(restored[name], current[name])


def test_delta_preserves_values_that_round_to_the_same_bfloat16():
    baseline = {"weight": torch.tensor([1.0], dtype=torch.bfloat16)}
    current = {
        "weight": (baseline["weight"].float() + 1e-6).to(torch.bfloat16)
    }

    delta = diff_state_dict(baseline, current)
    restored = apply_state_dict_delta(baseline, delta)

    assert delta.changed_elements == 0
    assert delta.entries == {}
    assert torch.equal(restored["weight"], current["weight"])


def test_sharded_param_manager_returns_seed_then_sparse_delta():
    saver = ShardedParamManager.__new__(ShardedParamManager)
    seed = {"weight": torch.zeros(4, dtype=torch.bfloat16)}
    changed = {"weight": torch.zeros(4, dtype=torch.bfloat16)}
    changed["weight"][2] = 3
    exports = iter([seed, changed])
    saver.get_local_hf_state_dict = lambda: next(exports)
    saver._hf_delta_baseline = None

    first = saver.get_local_hf_weight_delta()
    second = saver.get_local_hf_weight_delta()

    assert first.is_seed
    assert first.changed_ratio == 1.0
    assert torch.equal(
        apply_state_dict_delta({}, first)["weight"], seed["weight"]
    )
    assert not second.is_seed
    assert second.changed_elements == 1
    assert second.entries["weight"].indices.tolist() == [2]


def test_seed_return_tensors_do_not_corrupt_delta_baseline():
    saver = ShardedParamManager.__new__(ShardedParamManager)
    seed = {"weight": torch.zeros(4, dtype=torch.bfloat16)}
    changed = {"weight": torch.zeros(4, dtype=torch.bfloat16)}
    changed["weight"][2] = 3
    exports = iter([seed, changed])
    saver.get_local_hf_state_dict = lambda: next(exports)
    saver._hf_delta_baseline = None

    seed_delta = saver.get_local_hf_weight_delta()
    seed_delta.full_state["weight"][0] = 9
    delta = saver.get_local_hf_weight_delta()

    assert delta.entries["weight"].indices.tolist() == [2]


def test_checkpoint_manager_reuses_local_hf_saver(monkeypatch):
    manager = FSDP2CheckpointManager.__new__(FSDP2CheckpointManager)
    manager._local_hf_saver = None
    manager._local_hf_saver_key = None
    model = object()
    converter = object()
    savers = []

    def hf_saver(actual_model, source_path, actual_converter):
        assert actual_model is model
        assert source_path == "/source"
        assert actual_converter is converter
        saver = SimpleNamespace(get_local_hf_state_dict=lambda: {"weight": object()})
        savers.append(saver)
        return saver

    monkeypatch.setattr(manager, "_hf_saver", hf_saver)

    manager.get_local_hf_state_dict(
        model, "/source", weight_converter=converter
    )
    manager.get_local_hf_state_dict(
        model, "/source", weight_converter=converter
    )

    assert len(savers) == 1
