from typing import Optional, Tuple

import torch
import torch.nn as nn
from block import MistralDecoderBlock
from rmsnorm import RMSNorm


class Mistral(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        max_positional_embeddings: int,
        eps: float,
        rope_base: float,
        sliding_window: int,
        ffn_dim: int,
        num_blocks: int,
        num_heads: int,
        num_kv_heads: int,
        tie_word_embeddings: bool = False,
    ):
        super().__init__()

        self.input_embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=hidden_dim
        )
        self.blocks = nn.ModuleList(
            [
                MistralDecoderBlock(
                    hidden_dim=hidden_dim,
                    max_positional_embeddings=max_positional_embeddings,
                    eps=eps,
                    rope_base=rope_base,
                    sliding_window=sliding_window,
                    ffn_dim=ffn_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                )
                for _ in range(num_blocks)
            ]
        )

        self.vocab_size = vocab_size
        self.final_norm = RMSNorm(d_model=hidden_dim, eps=eps)
        self.lm_head = nn.Linear(
            in_features=hidden_dim, out_features=vocab_size, bias=False
        )

        self.sliding_window = sliding_window

        if tie_word_embeddings is True:
            self.lm_head.weight = self.input_embedding.weight

    def forward(
        self,
        input_ids,
        padding_mask: Optional[torch.Tensor] = None,
        offset: int = 0,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """
        Forward pass of the Mistral model.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            padding_mask: Optional padding mask [batch_size, 1, 1/seq_len, seq_len]
                            This is just the padding mask from the input sequence.
                            Causal and sliding window masks are handled internally.
            offset: Position offset for the current sequence
            past_key_values: Optional cached KV tensors for generation

        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
            new_kv_cache: Updated key-value cache for incremental decoding
        """

        x = self.input_embedding(input_ids)

        # Determine if we're processing a prompt or doing autoregressive generation
        is_prompt_processing = past_key_values is None

        new_caches = []

        for i, decoder_block in enumerate(self.blocks):
            # getting the cache for the current layer
            layer_past_kv_cache = (
                past_key_values[i] if not is_prompt_processing else None
            )

            # Each decoder block handles mask creation including:
            # 1. Padding (from attention_mask)
            # 2. Causal masking (created internally)
            # 3. Sliding window (created internally)
            x, current_kv_cache = decoder_block(
                x,
                padding_mask=padding_mask,
                offset=offset,
                past_key_value=layer_past_kv_cache,
            )

            # storing the updated cache irrespective of prompt processing or autoregressive generation
            new_caches.append(current_kv_cache)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        returned_cache = tuple(new_caches)
        return logits, returned_cache
