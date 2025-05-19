import torch
import torch.nn as nn
from typing import Tuple


class RollingBufferCache(nn.Module):
    """
    Implements a rolling buffer cache for keys and values, specifically designed
    for sliding window attention mechanisms.

    This cache stores a fixed-size window of the most recent key/value pairs.
    When new keys/values are added and the buffer is full, the oldest entries
    are overwritten, creating a "rolling" effect. It is suitable for scenarios
    like autoregressive decoding where only a limited history of keys and values
    is needed for attention computation.
    """

    def __init__(
        self,
        buffer_size: int,
        kv_dim: int,
    ):
        """
        Initializes the RollingBufferCache.

        Args:
            buffer_size: The maximum number of key/value tokens to store in the cache.
                         This should typically match the sliding window size of the
                         attention mechanism.
            kv_dim: The dimension of individual key and value vectors per head
                    (often referred to as head_dim).
        """
        super().__init__()
        self.buffer_size = buffer_size
        self.kv_dim = kv_dim  # Dimension of K/V vectors per head

        # Cache for keys, e.g., shape [batch, num_kv_heads, buffer_size, head_dim].
        # Not saved in state_dict.
        self.register_buffer("k_cache", None, persistent=False)
        # Cache for values, e.g., shape [batch, num_kv_heads, buffer_size, head_dim].
        # Not saved in state_dict.
        self.register_buffer("v_cache", None, persistent=False)

    def _ensure_cache_initialized(
        self,
        batch: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        """Initialize cache tensors if not already created or if batch size has changed"""
        if self.k_cache is None or self.k_cache.size(0) != batch:
            self.k_cache = torch.zeros(
                (batch, num_heads, self.buffer_size, head_dim),
                dtype=dtype,
                device=device,
            )
            self.v_cache = torch.zeros(
                (batch, num_heads, self.buffer_size, head_dim),
                dtype=dtype,
                device=device,
            )

    def update(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        current_seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the cache with new keys and values, and return the appropriate sliding window.

        Args:
            k: New key tensors [batch, num_kv_heads, seq_len, head_dim]
            v: New value tensors [batch, num_kv_heads, seq_len, head_dim]
            current_seq_len: Current position in the sequence (offset + seq_len)

        Returns:
            Tuple of windowed keys and values for attention
        """
        batch, num_heads, seq_len, head_dim = k.shape

        # Initializing the cache if needed
        self._ensure_cache_initialized(batch, num_heads, head_dim, k.dtype, k.device)

        # Calculating the indices for the circular buffer update
        # The positions being written are [current_seq_len - seq_len, ..., current_seq_len - 1]
        start_idx = (current_seq_len - seq_len) % self.buffer_size

        # Creating the indices for the new tokens
        indices = torch.arange(seq_len, device=k.device)
        cache_indices = (start_idx + indices) % self.buffer_size

        # Updating the cache with the vectorized operation
        self.k_cache[:, :, cache_indices] = k
        self.v_cache[:, :, cache_indices] = v

        # Determining the sliding window size
        # Min of current sequence length and buffer size
        window_size = min(current_seq_len, self.buffer_size)

        if window_size > 0:
            # Calculating the absolute positions of tokens in the sliding window
            # These are the last 'window_size' tokens: [current_seq_len - window_size, ..., current_seq_len - 1]
            start_pos = current_seq_len - window_size
            window_positions = torch.arange(start_pos, current_seq_len, device=k.device)

            # Mapping the absolute positions to the physical indices in the circular buffer
            physical_indices = window_positions % self.buffer_size

            # Extracting the window from the cache
            k_window = self.k_cache[:, :, physical_indices]
            v_window = self.v_cache[:, :, physical_indices]
        else:
            # Handling the edge case of the empty window
            k_window = self.k_cache.new_zeros((batch, num_heads, 0, head_dim))
            v_window = self.v_cache.new_zeros((batch, num_heads, 0, head_dim))

        return k_window, v_window
