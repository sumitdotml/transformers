import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from cache import RollingBufferCache
from mask import SlidingWindowMask
from rope import RoPE


# GQA with Sliding Window Attention
class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention with Sliding Window for the Mistral architecture.

    This implementation handles:
    1. Grouped queries (where multiple query heads share the same key/value heads)
    2. Sliding window attention (limiting context to a fixed window)
    3. Rotary positional embeddings
    4. KV caching for efficient autoregressive generation
    5. Comprehensive attention masking (padding + causal + sliding window)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        sliding_window: int,
        max_position_embeddings: int,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model is not divisible by num_heads"
        assert (
            num_heads % num_kv_heads == 0
        ), "num_heads should be divisible by num_kv_heads"
        self.d_model = d_model  # hidden_size
        self.num_heads = num_heads  # num_attention_heads
        self.num_kv_heads = num_kv_heads  # num_key_value_heads
        self.rope_base = rope_base  # rope_theta
        self.sliding_window = sliding_window

        self.head_dim = d_model // num_heads
        self.repeats = self.num_heads // self.num_kv_heads

        # Projection matrices
        self.wq = nn.Linear(in_features=d_model,
                            out_features=d_model, bias=False)
        self.wk = nn.Linear(
            in_features=d_model, out_features=num_kv_heads * self.head_dim, bias=False
        )
        self.wv = nn.Linear(
            in_features=d_model, out_features=num_kv_heads * self.head_dim, bias=False
        )
        self.wo = nn.Linear(in_features=d_model,
                            out_features=d_model, bias=False)

        self.apply_rope = RoPE(
            d_model=self.head_dim, max_seq_len=max_position_embeddings, base=rope_base
        )

        # Creating the rolling buffer for keys and values
        self.kv_cache = RollingBufferCache(
            buffer_size=sliding_window,
            kv_dim=self.head_dim,
        )

        # Scaling factor for attention scores
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Creating the attention mask generator - handles both causal constraint and sliding window
        self.mask_generator = SlidingWindowMask(sliding_window=sliding_window)

    def forward(
        self,
        x: torch.Tensor,
        offset: int = 0,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for Grouped Query Attention with Sliding Window.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            offset: Position offset for current sequence
            past_key_value: Optional cached key-value pair from previous forward passes
            padding_mask: Optional padding mask. This mask is crucial and must handle:
                            1. Padding in the current query `x`.
                            2. Padding in the historical key/value pairs retrieved from the
                               cache (if prefill involved padding).
                            It will be combined with the sliding window causal mask.
                            Expected shape like (batch, 1, q_len, kv_len) or (batch, 1, 1, kv_len).

        Returns:
            output: Output tensor [batch, seq_len, d_model]
            new_kv: Tuple of key-value cache tensors
        """
        batch, seq_len, _ = x.shape

        q = self.wq(x)  # [batch, seq_len, d_model]
        k = self.wk(x)  # [batch, seq_len, num_kv_heads * head_dim]
        v = self.wv(x)  # [batch, seq_len, num_kv_heads * head_dim]

        q = q.view(batch, seq_len, self.num_heads,
                   self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_kv_heads,
                   self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_kv_heads,
                   self.head_dim).transpose(1, 2)

        # RoPE positional embeddings
        q = self.apply_rope(q, offset=offset)
        k = self.apply_rope(k, offset=offset)

        # Initializing the cache from past_key_value if provided
        if past_key_value is not None:
            past_k, past_v = past_key_value
            if (
                self.kv_cache.k_cache is None
                or self.kv_cache.k_cache.shape != past_k.shape
                or self.kv_cache.k_cache.device != past_k.device
            ):
                self.kv_cache.k_cache = past_k
                self.kv_cache.v_cache = past_v

        current_total_len = offset + seq_len

        # Updating the cache and getting the sliding window
        k_window, v_window = self.kv_cache.update(k, v, current_total_len)

        # GQA by repeating KV heads to match query heads
        if self.repeats > 1:
            k_expanded = torch.repeat_interleave(
                k_window, repeats=self.repeats, dim=1)
            v_expanded = torch.repeat_interleave(
                v_window, repeats=self.repeats, dim=1)
        else:
            k_expanded = k_window
            v_expanded = v_window

        # Generating the attention mask for the sliding window
        # Note: This includes THREE types of masking in one tensor:
        # 1. Padding mask: From the attention_mask input parameter
        # 2. Causal mask: Each token can only attend to itself and previous tokens
        # 3. Sliding window: Each token can only attend to at most sliding_window previous tokens
        combined_mask = self.mask_generator.get_mask(
            batch_size=batch,
            q_len=seq_len,
            kv_len=k_expanded.size(2),
            offset=offset,
            input_padding_mask=padding_mask,
            device=x.device,
            dtype=x.dtype,
        )

        q_scaled = q * self.scale  # scaling before the attention computation
        attn_scores = torch.matmul(q_scaled, k_expanded.transpose(2, 3))
        attn_scores = attn_scores + combined_mask
        attn_weights = torch.softmax(
            attn_scores.float(), dim=-1).to(attn_scores.dtype)
        attn_output = torch.matmul(attn_weights, v_expanded)

        # Reshaping and projecting back to the model dimension
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(
                batch, seq_len, self.d_model)
        )
        output = self.wo(attn_output)

        # Output and the updated KV cache
        return output, (self.kv_cache.k_cache, self.kv_cache.v_cache)
