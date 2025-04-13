from typing import Tuple, Optional, List, Union
import torch
import torch.nn as nn
from config import CONFIG


class RoPE(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_seq_len: int,
        base: float,
    ):
        """
        Initializes the RoPE module.

        Args:
            d_model (int): Dimension of the embeddings. Must be even.
            max_seq_len (int): Maximum sequence length for precomputation.
            base (int): The base used in the positional encoding calculation.
        """
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"Embedding dimension {d_model} must be even for RoPE.")
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.base = base
        self.d = d_model // 2

        # Precomputing cos and sin frequencies
        # Creating cache on CPU initially, moving to appropriate device in forward pass if needed
        self.cos_cached, self.sin_cached = self.RoPEFrequencyCache(
            self.d_model, self.max_seq_len, self.base, device=torch.device("cpu")
        )

    @staticmethod
    def RoPEFrequencyCache(
        d_model: int, max_seq_len: int, base: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Precomputes the cosine and sine frequencies for RoPE.

        Args:
            d_model (int): Dimension of the embeddings.
            max_seq_len (int): Maximum sequence length.
            base (int): The base used in the positional encoding calculation.
            device: The torch device to create tensors on.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tensors containing precomputed cos and sin values
                                              of shape (max_seq_len, d_model / 2).
        """
        d = d_model // 2
        # Calculating theta_j = 1.0 / (base ** (2j / d_model)) for j = 0..d-1
        indices_j = torch.arange(
            0, d, dtype=torch.float32, device=device
        )  # Shape: (d,)
        theta = 1.0 / (base ** ((2 * indices_j) / d_model))  # Shape: (d,)

        # Creating position indices m = 0..max_seq_len-1
        position_indices_m = torch.arange(
            max_seq_len, dtype=torch.float32, device=device
        )  # Shape: (max_seq_len,)

        # Calculating angles m * theta_j using broadcasting
        # angles shape: (max_seq_len, 1) * (1, d) -> (max_seq_len, d)
        angles = position_indices_m[:, None] * theta[None, :]  # Shape: (max_seq_len, d)

        cos_cached = torch.cos(angles)  # Shape: (max_seq_len, d)
        sin_cached = torch.sin(angles)  # Shape: (max_seq_len, d)

        return cos_cached, sin_cached

    @staticmethod
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        # Example input shape: (batch, seq_len, heads, dim) or (batch, seq_len, dim)
        # Rotate on the last dimension (dim)
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def apply_rope(
        x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies the RoPE rotation using the original algorithm.

        Args:
            x (torch.Tensor): Input tensor, e.g., shape (batch, seq_len, dim) or (batch, heads, seq_len, dim).
                              RoPE is applied to the last dimension.
            cos (torch.Tensor): Precomputed cosine values, broadcastable to x.
                                E.g., shape (seq_len, dim/2) -> unsqueezed to (1, seq_len, 1, ..., 1, dim/2) for 4D x.
            sin (torch.Tensor): Precomputed sine values, same shape as cos.

        Returns:
            torch.Tensor: Rotated tensor, same shape as x.
        """
        # cos/sin inputs have shape (seq_len, d_half)
        # We need to unsqueeze them to broadcast correctly with x
        # x typically has shape (batch, seq_len, ...) e.g. (batch, seq_len, dim) or (batch, seq_len, heads, dim)
        # We want cos/sin to broadcast along all dims except seq_len (dim 1) and the feature dim (last dim)
        # Target shape for cos/sin (before cat): (1, seq_len, 1, ..., 1, d_half)

        # Adding a singleton dim for batch
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

        # Adding singleton dims for any dimensions between seq_len and the final feature dim
        # Example: x=(b, s, h, d), ndim=4. Need cos=(1, s, 1, d_half).
        # cos starts as (s, d_half). After unsqueeze(0) -> (1, s, d_half).
        # Need to add ndim - 3 = 4 - 3 = 1 singleton dim before d_half.
        # Example: x=(b, s, d), ndim=3. Need cos=(1, s, d_half).
        # cos starts as (s, d_half). After unsqueeze(0) -> (1, s, d_half).
        # Need to add ndim - 3 = 3 - 3 = 0 singleton dims.
        if x.ndim > 3:
            num_intermediate_dims = x.ndim - 3
            for _ in range(num_intermediate_dims):
                cos = cos.unsqueeze(-2)  # Add dim before the last one (d_half)
                sin = sin.unsqueeze(-2)

        # Using the standard RoPE implementation method directly
        # Splitting the input tensor into even and odd dimensions along the last dimension
        x_even = x[..., 0::2]  # Shape: (batch, seq_len, ..., d/2)
        x_odd = x[..., 1::2]  # Shape: (batch, seq_len, ..., d/2)

        # Applying rotation to even and odd dimensions separately
        # The broadcast of cos/sin already handles the sequence dimension correct alignment
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_even * sin + x_odd * cos

        # Interleaving (i.e., merging) the dimensions again
        x_rotated = torch.empty_like(x)
        x_rotated[..., 0::2] = x_rotated_even
        x_rotated[..., 1::2] = x_rotated_odd

        return x_rotated

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Forward pass applying RoPE across the sequence dimension.

        Args:
            x (torch.Tensor): Input tensor, shape (Batch, SeqLen, ..., Dim).
                              The last dimension must be self.d_model. RoPE is applied
                              positionally along the SeqLen dimension.

            offset (int): The starting absolute position index for the sequence `x`.
                          Defaults to 0 (for processing prompts or full sequences).
                          Used during generation with KV caching.

        Returns:
            torch.Tensor: Output tensor with RoPE applied, same shape as x.
        """
        # Assuming x has shape like (batch_size, seq_len, ...) or (batch_size, seq_len, num_heads, dim)
        seq_len = x.shape[1]

        # Ensuring sequence length (with offset) is within bounds
        if not (0 <= offset and offset + seq_len <= self.max_seq_len):
            raise ValueError(
                f"Request absolute positions [{offset}:{offset + seq_len}] (i.e., "
                f"sequence length {seq_len}) + offset {offset} are out of bounds for "
                f"max_seq_len {self.max_seq_len}"
            )

        # Ensuring cache tensors are on the same device as input x
        if self.cos_cached.device != x.device:
            self.cos_cached = self.cos_cached.to(x.device)
            self.sin_cached = self.sin_cached.to(x.device)

        # Getting the precomputed cos and sin values for the sequence length
        # cos_cached/sin_cached shape: (max_seq_len, d/2)
        # Slicing for the actual sequence length
        cos_values = self.cos_cached[offset : offset + seq_len]  # Shape: (seq_len, d/2)
        sin_values = self.sin_cached[offset : offset + seq_len]  # Shape: (seq_len, d/2)

        # Applying RoPE using the static helper method
        # The apply_rope method handles broadcasting cos/sin to match x's dimensions
        return self.apply_rope(x, cos_values, sin_values)


#######################################
# TESTING (GENERATED WITH GEMINI 2.5) #
#######################################
if __name__ == "__main__":
    config = CONFIG

    torch.manual_seed(0)
    d_model = config["hidden_size"]
    max_seq_len = config["max_position_embeddings"]
    base = config["rope_theta"]

    # Instantiate RoPE
    try:
        rope = RoPE(d_model=d_model, max_seq_len=max_seq_len, base=base)
        print(f"Initialized RoPE with d_model={d_model}, max_seq_len={max_seq_len}")
        print(f"Cache shape: cos={rope.cos_cached.shape}, sin={rope.sin_cached.shape}")
    except Exception as e:
        print(f"Error initializing RoPE: {e}")
        exit()

    # Test Case 1: Sequence Processing (Main test)
    print("\n--- Test Case 1: Sequence Processing ---")
    seq_len = 10
    batch_size = 2
    x_seq = torch.randn(batch_size, seq_len, d_model)
    print(f"Input sequence shape: {x_seq.shape}")
    try:
        rotated_seq = rope(x_seq)  # Apply RoPE across the sequence
        print(f"Rotated sequence shape: {rotated_seq.shape}")

        # Verification: Apply RoPE manually position by position and compare
        rotated_manual = torch.zeros_like(x_seq)
        rope_single_pos = RoPE(
            d_model=d_model, max_seq_len=max_seq_len, base=base
        )  # Use original implementation for single pos check

        # Need the *original* apply_rope and forward that took position_index
        # Let's redefine it here locally for the test comparison
        @staticmethod
        def apply_rope_single_pos(
            x_pos: torch.Tensor, cos_pos: torch.Tensor, sin_pos: torch.Tensor
        ) -> torch.Tensor:
            x_even = x_pos[..., 0::2]
            x_odd = x_pos[..., 1::2]
            x_rotated_even = x_even * cos_pos - x_odd * sin_pos
            x_rotated_odd = x_even * sin_pos + x_odd * cos_pos
            x_rotated = torch.empty_like(x_pos)
            x_rotated[..., 0::2] = x_rotated_even
            x_rotated[..., 1::2] = x_rotated_odd
            return x_rotated

        cos_cache_test = rope_single_pos.cos_cached.to(x_seq.device)
        sin_cache_test = rope_single_pos.sin_cached.to(x_seq.device)

        # Since we are now using offset=0 as default in the forward method
        offset = 0  # Match the default in the forward method
        for i in range(seq_len):
            pos_i = (
                offset + i
            )  # Calculate position using offset (consistent with KV caching)
            cos_pos_i = cos_cache_test[pos_i]  # Shape (d/2,)
            sin_pos_i = sin_cache_test[pos_i]  # Shape (d/2,)
            rotated_manual[:, i, :] = apply_rope_single_pos(
                x_seq[:, i, :], cos_pos_i, sin_pos_i
            )

        difference = torch.sum(torch.abs(rotated_seq - rotated_manual)).item()
        print(
            f"Sum of absolute difference between batch RoPE and manual RoPE: {difference:.4e}"
        )
        assert torch.allclose(
            rotated_seq, rotated_manual, atol=1e-5
        ), "Batch RoPE output does not match manual RoPE."
        print("Assertion Passed: Batch RoPE matches manual RoPE.")

        # Print shapes to debug
        print(f"Testing shapes - x[0,0,:10]: {x_seq[0,0,:10].shape}")
        print(f"Single position cos shape: {cos_cache_test[0].shape}")
        print(f"Position range used: {offset} to {offset+seq_len-1}")

    except Exception as e:
        print(f"Error processing sequence: {e}")
        import traceback

        traceback.print_exc()

    # Test Case 2: First Token Behavior (At position 0)
    print("\n--- Test Case 2: First Token Behavior (At position 0) ---")
    try:
        first_token_input = x_seq[:, 0:1, :]  # Shape (batch, 1, dim)
        first_token_rotated = rope(first_token_input)  # Apply RoPE to seq length 1
        difference_pos0 = torch.sum(
            torch.abs(first_token_input - first_token_rotated)
        ).item()

        # With offset=0 (default), a sequence of length 1 should use position 0
        # Which should be close to identity operation
        print(f"Applying RoPE to sequence of length 1 (position 0, default offset)")
        print(f"Sum of difference between input and output: {difference_pos0:.4e}")

        # Verify with manual application to the correct position
        actual_pos = 0  # Match default offset
        first_token_manual = apply_rope_single_pos(
            first_token_input.squeeze(1),  # Remove seq_len dimension for single token
            cos_cache_test[actual_pos],
            sin_cache_test[actual_pos],
        )

        # Compare batch implementation with manual application
        manual_diff = torch.sum(
            torch.abs(first_token_rotated.squeeze(1) - first_token_manual)
        ).item()
        print(f"Difference between batch and manual application: {manual_diff:.4e}")
        assert torch.allclose(
            first_token_rotated.squeeze(1), first_token_manual, atol=1e-5
        ), "First token rotation doesn't match expected"
        print("Assertion Passed: First token rotation is correct at position 0.")

        # Also verify identity property at position 0
        assert torch.allclose(
            first_token_input.squeeze(1), first_token_rotated.squeeze(1), atol=1e-6
        ), "RoPE at pos 0 should be close to identity"
        print("Assertion Passed: RoPE at pos 0 is close to identity.")
    except Exception as e:
        print(f"Error during position 0 check: {e}")
        import traceback

        traceback.print_exc()

    # Test Case 3: Multi-dimensional Input (e.g., with Heads)
    print("\n--- Test Case 3: Multi-dimensional Input (e.g., with Heads) ---")
    num_heads = 4
    head_dim = d_model // num_heads  # Assuming d_model is divisible by num_heads
    # Reshape x_seq to simulate head dimension: (batch, seq_len, num_heads, head_dim)
    # NOTE: This assumes RoPE is applied *after* splitting into heads.
    # If RoPE is applied *before* splitting, the input shape would be (batch, seq_len, d_model) as in Case 1.
    # Let's test the case where RoPE is applied *per head* on head_dim
    # We need a RoPE instance for head_dim
    rope_per_head = RoPE(d_model=head_dim, max_seq_len=max_seq_len, base=base)
    x_multi_dim = torch.randn(batch_size, seq_len, num_heads, head_dim)
    print(f"Input shape (per-head RoPE): {x_multi_dim.shape}")
    try:
        # Apply RoPE across seq_len, acting on the last dim (head_dim)
        rotated_multi_dim = rope_per_head(x_multi_dim)
        print(f"Rotated output shape: {rotated_multi_dim.shape}")
        print(f"Successfully applied RoPE to multi-dimensional input.")
        # Add verification similar to Test Case 1 if needed
    except Exception as e:
        print(f"Error processing multi-dimensional input (per-head): {e}")
        import traceback

        traceback.print_exc()

    # Test Case 4: Boundary Checks (Sequence Length)
    print("\n--- Test Case 4: Boundary Checks (Sequence Length) ---")
    try:
        # Test max valid sequence length
        print(f"Testing max sequence length: {max_seq_len}")
        x_max_seq = torch.randn(batch_size, max_seq_len, d_model)
        _ = rope(x_max_seq)
        print("Max sequence length test successful.")
    except Exception as e:
        print(f"Error testing max sequence length: {e}")

    try:
        # Test invalid sequence length (too high)
        invalid_seq_len = max_seq_len + 1
        print(f"Testing invalid sequence length: {invalid_seq_len}")
        x_invalid_seq = torch.randn(batch_size, invalid_seq_len, d_model)
        _ = rope(x_invalid_seq)
        print(
            "ERROR: Should have raised ValueError for out-of-bounds sequence length."
        )  # Should not reach here
    except ValueError as e:
        print(f"Successfully caught expected error for seq_len {invalid_seq_len}: {e}")
    except Exception as e:
        print(f"Caught unexpected error for seq_len {invalid_seq_len}: {e}")

    try:
        # Test invalid sequence length (zero)
        invalid_seq_len_zero = 0
        print(f"Testing invalid sequence length: {invalid_seq_len_zero}")
        x_invalid_seq_zero = torch.randn(batch_size, invalid_seq_len_zero, d_model)
        _ = rope(x_invalid_seq_zero)
        print(
            "ERROR: Should have raised ValueError for out-of-bounds sequence length."
        )  # Should not reach here
    except ValueError as e:
        print(
            f"Successfully caught expected error for seq_len {invalid_seq_len_zero}: {e}"
        )
    except Exception as e:
        print(f"Caught unexpected error for seq_len {invalid_seq_len_zero}: {e}")

    # Bonus Test Case: KV Caching with explicit offset
    print("\n--- Bonus Test Case: KV Caching with explicit offset ---")
    try:
        # Simulate generating a new token with KV cache offset
        # After processing a sequence of length 10, next token would be at position 10
        cache_offset = 10
        new_token = torch.randn(batch_size, 1, d_model)  # Shape (batch, 1, dim)

        # Call forward with explicit offset
        new_token_rotated = rope(new_token, offset=cache_offset)

        # Verify with manual application
        new_token_manual = apply_rope_single_pos(
            new_token.squeeze(1),
            cos_cache_test[cache_offset],
            sin_cache_test[cache_offset],
        )

        # Compare
        new_token_diff = torch.sum(
            torch.abs(new_token_rotated.squeeze(1) - new_token_manual)
        ).item()
        print(
            f"Using offset={cache_offset} - Difference between batch and manual: {new_token_diff:.4e}"
        )
        assert torch.allclose(
            new_token_rotated.squeeze(1), new_token_manual, atol=1e-5
        ), "Offset RoPE doesn't match expected"
        print(
            f"Assertion Passed: RoPE with explicit offset {cache_offset} works correctly."
        )
    except Exception as e:
        print(f"Error during KV caching test: {e}")
        import traceback

        traceback.print_exc()
