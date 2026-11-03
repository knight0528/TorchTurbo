"""GLM-MoE-DSA MFU recipe coverage."""

from torchturbo.engine.metrics import op_flops as flops
from torchturbo.engine.metrics.flops import ComputeProfile
from torchturbo.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
from torchturbo.models.glm_moe_dsa.mfu import GlmMoeDsaFlops
from torchturbo.models.glm_moe_dsa.model import TurboGlmMoeDsaForCausalLM


def _small_config() -> GlmMoeDsaConfig:
    return GlmMoeDsaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=8,
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
        qk_nope_head_dim=8,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        index_topk=8,
        index_head_dim=8,
        index_n_heads=2,
        indexer_types=["full", "shared", "full", "shared"],
        mlp_layer_types=["dense", "sparse", "dense", "sparse"],
    )


def _components(sequence_length: int = 16):
    recipe = GlmMoeDsaFlops(
        _small_config(),
        ComputeProfile(batch_size=1, seq_length=sequence_length),
    )
    return {component.name: component for component in recipe.components()}


def test_glm_components_include_dsa_attention_and_moe() -> None:
    components = _components()
    names = " | ".join(components)

    assert "attention/dsa" in names
    assert "dsa/indexer" in names
    assert "ffn/dense" in names
    assert "ffn/moe" in names
    assert "logits" in names
    assert all(component.fwd_flops > 0 for component in components.values())


def test_glm_attention_scales_with_index_topk() -> None:
    config = _small_config()
    profile = ComputeProfile(batch_size=1, seq_length=32)
    small = GlmMoeDsaFlops(config, profile)
    small_flops = small._attention_flops_for_seg(32)

    config.index_topk = 16
    large = GlmMoeDsaFlops(config, profile)
    large_flops = large._attention_flops_for_seg(32)

    assert large_flops > small_flops


def test_glm_indexer_scores_all_visible_tokens() -> None:
    recipe = GlmMoeDsaFlops(_small_config(), ComputeProfile(batch_size=1, seq_length=4))

    rows = 4
    projections = flops.linear_flops(rows, 16, 16)
    projections += flops.linear_flops(rows, 64, 8)
    projections += flops.linear_flops(rows, 64, 2)
    visible_pairs = 4 * 5 // 2
    scores = 2 * visible_pairs * 2 * 8
    reductions = 2 * visible_pairs * 2

    assert recipe._indexer_flops_for_seg(4) == projections + scores + reductions


def test_glm_model_exposes_flops_recipe() -> None:
    model = object.__new__(TurboGlmMoeDsaForCausalLM)

    assert model.get_flops_class() is GlmMoeDsaFlops
