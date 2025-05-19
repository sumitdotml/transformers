import torch
import torch.nn as nn
from attention import GroupedQueryAttention
from rmsnorm import RMSNorm


class MistralDecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        max_positional_embeddings: int,
        sliding_window: int,
    ) -> None:
        super().__init__()
        self.pre_gqa_norm = RMSNorm(d_model=d_model, eps=eps)
        self.gqa = GroupedQueryAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            rope_base=rope_base,
            max_position_embeddings=max_positional_embeddings,
            sliding_window=sliding_window,
        )

        self.pre_ffn_norm = RMSNorm(d_model=d_model, eps=eps)
