from typing import Tuple
import torch
import torch.nn as nn
from config import CONFIG


class RoPE(nn.Module):
    def __init__(
        self,
        d_model=CONFIG["hidden_size"],
        max_seq_len=CONFIG.get("max_seq_len", 2048),
        base=CONFIG.get("rope_base", 10000),
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
    def apply_rope(
        x: torch.Tensor, cos_values: torch.Tensor, sin_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies the RoPE rotation to the input tensor using precomputed cos/sin values.

        Args:
            x (torch.Tensor): Input tensor, shape (..., d_model).
            cos (torch.Tensor): Precomputed cosine values for the position, shape (d,).
            sin (torch.Tensor): Precomputed sine values for the position, shape (d,).

        Returns:
            torch.Tensor: Rotated tensor, shape (..., d_model).
        """
        # Splitting x into even and odd parts along the last dimension
        x_even = x[..., 0::2]  # Shape: (..., d)
        x_odd = x[..., 1::2]  # Shape: (..., d)

        # Applying rotation using RoPE formulas
        # cos and sin have shape (d,), they will broadcast correctly against x_even/x_odd
        # x'_even = x_even * cos - x_odd * sin
        # x'_odd = x_even * sin + x_odd * cos
        x_rotated_even = x_even * cos_values - x_odd * sin_values
        x_rotated_odd = x_even * sin_values + x_odd * cos_values

        # Combining back into a single tensor
        x_rotated = torch.empty_like(x)  # Use empty_like for potential efficiency
        x_rotated[..., 0::2] = x_rotated_even
        x_rotated[..., 1::2] = x_rotated_odd

        return x_rotated

    def forward(self, x: torch.Tensor, position_index: int) -> torch.Tensor:
        """
        Forward pass applying RoPE.

        Args:
            x (torch.Tensor): Input tensor, shape (Batch, SeqLen, ..., Dim) or (..., Dim).
                              The last dimension must be self.d_model.
            position_index (int): The absolute position index in the sequence.
                                  Must be less than max_seq_len.

        Returns:
            torch.Tensor: Output tensor with RoPE applied, same shape as x.
        """
        # Ensuring position_index is within bounds
        if not (0 <= position_index < self.max_seq_len):
            raise ValueError(
                f"Position index {position_index} is out of bounds for "
                f"max_seq_len {self.max_seq_len}"
            )

        # Ensuring cache tensors are on the same device as input x
        # Moving cache to device on first forward call or if device changes
        if self.cos_cached.device != x.device:
            self.cos_cached = self.cos_cached.to(x.device)
            self.sin_cached = self.sin_cached.to(x.device)

        # Getting the precomputed cos and sin values for the given position index
        # cos_cached shape: (max_seq_len, d), sin_cached shape: (max_seq_len, d)
        # We need the slice for the specific position_index
        cos_cached = self.cos_cached[position_index]  # Shape: (d,)
        sin_cached = self.sin_cached[position_index]  # Shape: (d,)

        # Applying RoPE using the static helper method
        return self.apply_rope(x, cos_cached, sin_cached)


#######################################
# TESTING (GENERATED WITH GEMINI 2.5) #
#######################################
if __name__ == "__main__":
    config = CONFIG

    torch.manual_seed(0)
    d_model = config["hidden_size"]
    max_seq_len = config["max_position_embeddings"]

    # Instantiate RoPE
    try:
        rope = RoPE(d_model=d_model, max_seq_len=max_seq_len)
        print(f"Initialized RoPE with d_model={d_model}, max_seq_len={max_seq_len}")
        print(f"Cache shape: cos={rope.cos_cached.shape}, sin={rope.sin_cached.shape}")
    except Exception as e:
        print(f"Error initializing RoPE: {e}")
        exit()

    # Test Case 1: Single vector
    print("\n--- Test Case 1: Single Vector ---")
    x_single = torch.randn(1, d_model)  # Shape (Batch=1, Dim=d_model)
    position_index = 5  # Example position
    print(f"Input shape: {x_single.shape}, Position: {position_index}")
    try:
        x_rotated_single = rope(x_single, position_index)
        print(f"Output shape: {x_rotated_single.shape}")
        # print("Input (first 10):", x_single[0, :10])
        # print("Output (first 10):", x_rotated_single[0, :10])
    except Exception as e:
        print(f"Error during forward pass: {e}")

    # Test Case 2: Check position 0 (should be near identity)
    print("\n--- Test Case 2: Position 0 (Identity Check) ---")
    position_index_zero = 0
    try:
        x_rotated_pos0 = rope(x_single, position_index_zero)
        difference = torch.sum(torch.abs(x_single - x_rotated_pos0)).item()
        print(f"Applying RoPE at position {position_index_zero}")
        print(
            f"Sum of absolute difference between input and RoPE(pos=0): {difference:.4e}"
        )  # Should be close to 0
        assert torch.allclose(
            x_single, x_rotated_pos0, atol=1e-6
        ), "RoPE at pos 0 should be close to identity"
        print("Assertion Passed: RoPE at pos 0 is close to identity.")
    except Exception as e:
        print(f"Error during position 0 check: {e}")

    # Test Case 3: Sequence of vectors
    print("\n--- Test Case 3: Sequence Processing ---")
    seq_len = 10
    batch_size = 2
    x_seq = torch.randn(batch_size, seq_len, d_model)
    print(f"Input sequence shape: {x_seq.shape}")
    rotated_seq = torch.zeros_like(x_seq)
    try:
        for i in range(seq_len):
            # Apply RoPE position by position
            # Input to rope: (batch_size, d_model)
            rotated_seq[:, i, :] = rope(x_seq[:, i, :], i)
        print(f"Rotated sequence shape: {rotated_seq.shape}")
    except Exception as e:
        print(f"Error processing sequence: {e}")

    # Test Case 4: Input with multiple leading dimensions (e.g., including heads)
    # RoPE should work regardless of leading dimensions as it acts on the last one.
    print("\n--- Test Case 4: Multi-dimensional Input (e.g., with Heads) ---")
    num_heads = 4
    # Assuming RoPE is applied on the full dimension before/after head splitting/merging
    x_multi_dim = torch.randn(batch_size, seq_len, num_heads, d_model)
    print(f"Input shape: {x_multi_dim.shape}")
    rotated_multi_dim = torch.zeros_like(x_multi_dim)
    try:
        for i in range(seq_len):
            # Apply RoPE position by position to the last dimension
            # Input to rope: (batch_size, num_heads, d_model)
            rotated_multi_dim[:, i, :, :] = rope(x_multi_dim[:, i, :, :], i)
        print(f"Rotated output shape: {rotated_multi_dim.shape}")
    except Exception as e:
        print(f"Error processing multi-dimensional input: {e}")

    # Test Case 5: Boundary Checks
    print("\n--- Test Case 5: Boundary Checks ---")
    try:
        # Test max valid position index
        max_pos = max_seq_len - 1
        print(f"Testing max position index: {max_pos}")
        _ = rope(x_single, max_pos)
        print("Max position index test successful.")
    except Exception as e:
        print(f"Error testing max position index: {e}")

    try:
        # Test invalid position index (too high)
        invalid_pos = max_seq_len
        print(f"Testing invalid position index: {invalid_pos}")
        _ = rope(x_single, invalid_pos)
        print(
            "ERROR: Should have raised ValueError for out-of-bounds index."
        )  # Should not reach here
    except ValueError as e:
        print(f"Successfully caught expected error for index {invalid_pos}: {e}")
    except Exception as e:
        print(f"Caught unexpected error for index {invalid_pos}: {e}")

    try:
        # Test invalid position index (negative)
        invalid_pos_neg = -1
        print(f"Testing invalid position index: {invalid_pos_neg}")
        _ = rope(x_single, invalid_pos_neg)
        print(
            "ERROR: Should have raised ValueError for out-of-bounds index."
        )  # Should not reach here
    except ValueError as e:
        print(f"Successfully caught expected error for index {invalid_pos_neg}: {e}")
    except Exception as e:
        print(f"Caught unexpected error for index {invalid_pos_neg}: {e}")
