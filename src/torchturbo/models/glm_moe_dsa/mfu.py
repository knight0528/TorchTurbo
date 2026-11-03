"""Theoretical forward-FLOP recipe for GLM-MoE-DSA.

Counts sparse DSA attention, dense+MoE FFN work, and the DSA indexer.
"""

from __future__ import annotations

from ...engine.metrics import op_flops as flops
from ...engine.metrics.flops import Component, LLMFlops

__all__ = ["GlmMoeDsaFlops"]


class GlmMoeDsaFlops(LLMFlops):
    """GLM-MoE-DSA forward FLOPs."""

    def _parse_config(self) -> None:
        super()._parse_config()
        g = self._cfg

        q_lora_rank = g("q_lora_rank", default=None)
        kv_lora_rank = g("kv_lora_rank", default=None)
        self.q_lora_rank: int | None = None if q_lora_rank is None else int(q_lora_rank)
        self.kv_lora_rank: int | None = None if kv_lora_rank is None else int(kv_lora_rank)
        self.qk_nope_head_dim: int = int(g("qk_nope_head_dim", default=0))
        self.qk_rope_head_dim: int = int(g("qk_rope_head_dim", default=0))
        self.v_head_dim: int = int(g("v_head_dim", default=self.qk_nope_head_dim))

        self.index_topk: int = int(g("index_topk", default=0))
        self.indexer_num_heads: int = int(g("index_n_heads", default=0))
        self.indexer_head_dim: int = int(g("index_head_dim", default=0))

        self.num_experts: int = int(g("num_experts", "n_routed_experts", default=0))
        self.top_k: int = int(g("num_experts_per_tok", default=0))
        self.expert_intermediate_size: int = int(
            g("moe_intermediate_size", "intermediate_size", default=0)
        )
        self.num_shared_experts: int = int(g("num_shared_experts", "n_shared_experts", default=0))
        self.shared_expert_intermediate_size: int = (
            self.expert_intermediate_size * self.num_shared_experts
        )

        self.mlp_layer_types: list[str] = list(g("mlp_layer_types", default=[]) or [])
        if not self.mlp_layer_types:
            dense_layers = min(3, self.num_layers)
            self.mlp_layer_types = ["dense"] * dense_layers + ["sparse"] * (
                self.num_layers - dense_layers
            )

        self.indexer_types: list[str] = list(g("indexer_types", default=[]) or [])
        if not self.indexer_types:
            self.indexer_types = ["full"] * self.num_layers

        self.num_dense_layers: int = sum(
            1 for layer_type in self.mlp_layer_types if layer_type == "dense"
        )
        self.num_moe_layers: int = self.num_layers - self.num_dense_layers
        self.num_full_indexer_layers: int = sum(
            1 for indexer_type in self.indexer_types if indexer_type == "full"
        )

    def _attention_topk(self, seg: int) -> int:
        return seg if self.index_topk <= 0 else min(seg, self.index_topk)

    def _attention_flops_for_seg(self, seg: int) -> int:
        rows = self.B * seg
        dhead = self.qk_nope_head_dim + self.qk_rope_head_dim
        sparse_width = self._attention_topk(seg)

        if self.q_lora_rank is None:
            q_proj = flops.linear_flops(rows, self.hidden_size, self.num_heads * dhead)
        else:
            q_proj = flops.linear_flops(rows, self.hidden_size, self.q_lora_rank)
            q_proj += flops.linear_flops(rows, self.q_lora_rank, self.num_heads * dhead)

        if self.kv_lora_rank is None:
            kv_proj = flops.linear_flops(
                rows,
                self.hidden_size,
                self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            )
        else:
            kv_proj = flops.linear_flops(
                rows,
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
            )
            kv_proj += flops.linear_flops(
                rows,
                self.kv_lora_rank,
                self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            )

        qk_product = 2 * self.B * seg * sparse_width * self.num_heads * dhead
        pv_product = 2 * self.B * seg * sparse_width * self.num_heads * self.v_head_dim
        o_proj = flops.linear_flops(rows, self.num_heads * self.v_head_dim, self.hidden_size)
        return q_proj + kv_proj + qk_product + pv_product + o_proj

    def _indexer_flops_for_seg(self, seg: int) -> int:
        rows = self.B * seg
        query_in_dim = self.hidden_size if self.q_lora_rank is None else self.q_lora_rank
        projections = flops.linear_flops(
            rows,
            query_in_dim,
            self.indexer_num_heads * self.indexer_head_dim,
        )
        projections += flops.linear_flops(rows, self.hidden_size, self.indexer_head_dim)
        projections += flops.linear_flops(rows, self.hidden_size, self.indexer_num_heads)

        visible_pairs = seg * (seg + 1) // 2
        scores = 2 * self.B * visible_pairs * self.indexer_num_heads * self.indexer_head_dim
        reductions = 2 * self.B * visible_pairs * self.indexer_num_heads
        return projections + scores + reductions

    def _dense_ffn_flops(self) -> int:
        if self.intermediate_size <= 0:
            return 0
        return flops.gated_mlp_flops(
            self.B,
            self.S_eff,
            self.hidden_size,
            self.intermediate_size,
        )

    def _moe_ffn_flops(self) -> int:
        if self.num_experts <= 0 or self.top_k <= 0:
            return self._dense_ffn_flops()
        return flops.moe_flops(
            self.B,
            self.S_eff,
            self.hidden_size,
            self.expert_intermediate_size,
            self.num_experts,
            self.top_k,
            shared_expert_intermediate_size=self.shared_expert_intermediate_size,
            use_gate=True,
        )

    def components(self) -> list[Component]:
        attention_per = sum(self._attention_flops_for_seg(seg) for seg in self.seq_segments)
        components = [
            Component(
                f"attention/dsa ({self.num_layers} layers)",
                attention_per * self.num_layers,
                note=(
                    f"sparse MLA topk={self.index_topk}; per-layer fwd={attention_per:,}"
                ),
            )
        ]

        if self.num_full_indexer_layers:
            indexer_per = sum(self._indexer_flops_for_seg(seg) for seg in self.seq_segments)
            components.append(
                Component(
                    f"dsa/indexer ({self.num_full_indexer_layers} layers)",
                    indexer_per * self.num_full_indexer_layers,
                    note=(
                        f"heads={self.indexer_num_heads}, dim={self.indexer_head_dim}, "
                        f"topk={self.index_topk}"
                    ),
                )
            )

        if self.num_dense_layers:
            dense_per = self._dense_ffn_flops()
            components.append(
                Component(
                    f"ffn/dense ({self.num_dense_layers} layers)",
                    dense_per * self.num_dense_layers,
                    note=(
                        f"gated MLP inter={self.intermediate_size}; per-layer fwd={dense_per:,}"
                    ),
                )
            )

        if self.num_moe_layers:
            moe_per = self._moe_ffn_flops()
            components.append(
                Component(
                    f"ffn/moe ({self.num_moe_layers} layers)",
                    moe_per * self.num_moe_layers,
                    note=(
                        f"top_k={self.top_k} experts={self.num_experts} "
                        f"shared_inter={self.shared_expert_intermediate_size}; "
                        f"per-layer fwd={moe_per:,}"
                    ),
                )
            )

        components.append(self.logits_component())
        return components
