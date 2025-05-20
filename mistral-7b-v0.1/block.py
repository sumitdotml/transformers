from typing import Optional, Tuple

import torch
import torch.nn as nn
from attention import GroupedQueryAttention
from ff import FeedForward
from rmsnorm import RMSNorm


class MistralDecoderBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        eps: float,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        max_positional_embeddings: int,
        sliding_window: int,
        ffn_dim: int,
    ) -> None:
        super().__init__()

        # ===== Sublayer 1: RMSNorm, GQA, Residual Connection
        self.pre_gqa_norm = RMSNorm(d_model=hidden_dim, eps=eps)
        self.gqa = GroupedQueryAttention(
            d_model=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            rope_base=rope_base,
            max_position_embeddings=max_positional_embeddings,
            sliding_window=sliding_window,
        )

        # ===== Sublayer 2: RMSNorm, FFN (SwiGLU), Residual Connection

        self.pre_ffn_norm = RMSNorm(d_model=hidden_dim, eps=eps)
        self.ff = FeedForward(hidden_dim=hidden_dim, ffn_dim=ffn_dim)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for the Mistral decoder block.

        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            padding_mask: Optional padding mask from input sequence
                            (Causal and sliding window masks are created in the attention module)
            past_key_value: Optional cached key-value pair from previous forward passes
            offset: Position offset for the current sequence

        Returns:
            output: Output tensor after attention and feed-forward layers
            current_key_value: Updated KV cache for this layer
        """
        # ======Sublayer 1
        pre_gqa_residual = x
        x = self.pre_gqa_norm(x)
        attn_output, current_key_value = self.gqa(
            x,
            past_key_value=past_key_value,
            offset=offset,
            padding_mask=padding_mask,
        )

        h = attn_output + pre_gqa_residual

        # ====== Sublayer 2
        pre_ffn_residual = h

        h = self.pre_ffn_norm(h)
        h = self.ff(h)

        out = h + pre_ffn_residual

        return out, current_key_value
