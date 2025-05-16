import torch
import torch.nn as nn
from typing import Tuple

class RoPE(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_seq_len: int,
        base: float = 10000.0,
    ):
        """
        Initializes the RoPE module.

        Args:
            d_model (int): Dimension of the features RoPE is applied to (e.g., head_dim). Must be even.
            max_seq_len (int): Maximum sequence length for precomputation.
            base (float): The base used in the positional encoding calculation.

        After initialization, we can also add the argument `offset` to the initialized module like this:

        ```python
        >> apply_rope = RoPE(d_model=4096, max_seq_len=32768)
        >> apply_rope(x_input, offset=10)
        ```


        """
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"Feature dimension {d_model} must be even for RoPE.")
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.base = base

        # precomputing frequencies and cache them.
        # these will have shape (max_seq_len, d_model / 2)
        # initializing on CPU, will be moved to appropriate device in forward pass if needed.
        self.cos_cached, self.sin_cached = self._build_cache(
            d_model, max_seq_len, base, device=torch.device("cpu")
        )

    @staticmethod
    def _build_cache(
        d_model: int, max_seq_len: int, base: float, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Precomputes the RoPE frequencies (cosine and sine components).

        Args:
            d_model (int): Dimension of the features.
            max_seq_len (int): Maximum sequence length.
            base (float): The base for positional encoding.
            device: The torch device to create tensors on.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tensors containing precomputed cos and sin values
                                              of shape (max_seq_len, d_model / 2).
        """
        d_half = d_model // 2

        # inverse frequencies: theta_i = 1.0 / (base^(2i / d_model)) for i in [0, d_half-1]
        theta_indices = torch.arange(0, d_half, dtype=torch.float32, device=device)
        theta = 1.0 / (base ** ((2 * theta_indices) / d_model)) # Shape: (d_half,)

        # positions m = [0, ..., max_seq_len-1]
        position_indices = torch.arange(max_seq_len, dtype=torch.float32, device=device) # Shape: (max_seq_len,)

        # outer product of positions and inverse frequencies: m * theta_i
        # angles shape: (max_seq_len, 1) * (1, d_half) -> (max_seq_len, d_half)
        angles = position_indices.unsqueeze(1) * theta.unsqueeze(0)

        cos_cached = torch.cos(angles) # (max_seq_len, d_half)
        sin_cached = torch.sin(angles) # (max_seq_len, d_half)
        return cos_cached, sin_cached

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Applies RoPE to the input tensor x.

        Args:
            x (torch.Tensor): Input tensor. Expected shape (Batch, SeqLen, ..., Dim).
                              RoPE is applied along the SeqLen (dim 1) and Dim (last dim).
                              The last dimension (Dim) must match self.d_model.
            offset (int): The starting absolute position index for the sequence `x`.
                          Defaults to 0. Used for KV caching during generation.

        Returns:
            torch.Tensor: Output tensor with RoPE applied, same shape as x.
        """
        assert x.shape[-1] == self.d_model, \
            f"Input tensor's last dimension ({x.shape[-1]}) must match RoPE's d_model ({self.d_model})"

        seq_len = x.shape[1]
        
        # ensuring positions are in bounds
        if offset < 0 or offset + seq_len > self.max_seq_len:
            raise ValueError(
                f"Positions {offset}:{offset+seq_len} out of bounds for max_seq_len {self.max_seq_len}"
            )
        
        # moving cache to correct device if needed
        if self.cos_cached.device != x.device:
            self.cos_cached = self.cos_cached.to(x.device)
            self.sin_cached = self.sin_cached.to(x.device)
        
        # getting relevant position encodings
        cos = self.cos_cached[offset:offset+seq_len]  # (seq_len, d_half)
        sin = self.sin_cached[offset:offset+seq_len]  # (seq_len, d_half)
        
        # creating appropriate view for broadcasting
        # for x with shape [batch, seq_len, ...], we need [1, seq_len, 1, ..., d_half]
        view_shape = [1, seq_len] + [1] * (x.ndim - 3) + [-1]
        cos = cos.view(*view_shape)  # (1, seq_len, 1, ..., d_half)
        sin = sin.view(*view_shape)  # (1, seq_len, 1, ..., d_half)
        
        # applying rotation directly to even and odd indices
        # this avoids the need for rotate_half and repeat_interleave (like in my llama implementation)
        x_out = torch.empty_like(x)
        x_out[..., ::2] = x[..., ::2] * cos - x[..., 1::2] * sin
        x_out[..., 1::2] = x[..., 1::2] * cos + x[..., ::2] * sin
        
        return x_out