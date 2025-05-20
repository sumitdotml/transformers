from typing import Optional

import torch
import torch.nn as nn


class SlidingWindowMask(nn.Module):
    """
    Implements comprehensive masking for Mistral's attention mechanism.

    This class creates a combined mask that handles three distinct aspects:

    1. Causal Masking: Ensures each token can only attend to itself and previous tokens
                       (standard in autoregressive language models)

    2. Sliding Window: Limits attention to a fixed window of previous tokens
                       (reduces computational complexity for long sequences)

    3. Padding Masking: Handles padding in the input sequence
                       (prevents attending to padding tokens)

    The resulting mask is combined into a single tensor for efficient computation.
    """

    def __init__(self, sliding_window: int):
        """
        Initialize the sliding window mask generator.

        Args:
            sliding_window: The size of the attention window. Each token can attend
                           to at most this many previous tokens.
        """
        super().__init__()
        self.sliding_window = sliding_window

    def get_mask(
        self,
        batch_size: int,
        q_len: int,
        kv_len: int,
        offset: int,  # Absolute position offset of the query sequence
        input_padding_mask: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Create a comprehensive attention mask combining causal, sliding window, and padding constraints.

        Args:
            batch_size: Batch size of the input
            q_len: Length of the current query sequence
            kv_len: Length of the key/value window from cache
            offset: Absolute starting position of the query sequence
            input_padding_mask: Optional padding mask (B, 1, Q, K) or (B, 1, 1, K) with -inf for masked padding tokens
            device: Device for the mask
            dtype: Data type for the mask

        Returns:
            Combined mask tensor (batch_size, 1, q_len, kv_len) with the following properties:
            - Value 0.0: Positions that can be attended to
            - Value -inf: Positions that should not be attended to, due to any of:
                * Causal constraint (future tokens)
                * Sliding window constraint (tokens too far in the past)
                * Padding mask (padding tokens in the input)
        """
        # PART 1: Generate the causal + sliding window mask
        q_indices = torch.arange(q_len, device=device)  # [0, 1, ..., q_len-1]
        # [0, 1, ..., kv_len-1]
        k_indices = torch.arange(kv_len, device=device)

        # Computing the absolute position of each query token
        # [offset, offset+1, ..., offset+q_len-1]
        q_positions = offset + q_indices

        # Computing the minimum absolute position each key token could have
        # based on the sliding window size - this enforces the sliding window constraint
        min_k_position = q_positions.unsqueeze(-1) - self.sliding_window + 1

        # Converting the key indices to the absolute positions
        # For the sliding window, the key positions start at (offset + q_len - kv_len)
        # This is because we're looking at the last kv_len tokens
        k_start = offset + q_len - kv_len
        # [k_start, k_start+1, ..., k_start+kv_len-1]
        k_positions = k_start + k_indices

        # A query at position i can attend to keys from (i-sliding_window+1) up to position i
        # Creating the mask where -inf means "don't attend"
        # We mask if:
        # 1. key position < min_key_position (sliding window constraint - too far back)
        # 2. key position > query position (causal constraint - in the future)
        k_positions = k_positions.unsqueeze(0)  # [1, kv_len]
        min_k_position = min_k_position  # [q_len, 1]
        q_positions = q_positions.unsqueeze(-1)  # [q_len, 1]

        # Combining the causal and sliding window constraints
        mask_condition = (k_positions < min_k_position) | (
            k_positions > q_positions)
        causal_window_mask = torch.where(
            mask_condition,
            torch.tensor(float("-inf"), device=device, dtype=dtype),
            torch.tensor(0.0, device=device, dtype=dtype),
        )  # [q_len, kv_len]

        # Expanding the mask to match the expected shape [batch_size, 1, q_len, kv_len]
        causal_window_mask = causal_window_mask.unsqueeze(0).unsqueeze(0)

        # Applying the batch dimension if needed
        if batch_size > 1:
            causal_window_mask = causal_window_mask.expand(
                batch_size, 1, q_len, kv_len)

        # PART 2: Incorporating the padding mask if provided
        if input_padding_mask is not None:
            # Processing the padding mask to match the dimensions
            if input_padding_mask.size(2) == 1 and q_len > 1:
                # Expanding the singleton query dimension
                input_padding_mask = input_padding_mask.expand(
                    -1, -1, q_len, -1)

            # Handling the key dimension mismatch
            if input_padding_mask.size(-1) != kv_len:
                if kv_len < input_padding_mask.size(-1):
                    # Using the last kv_len elements of the padding mask
                    input_padding_mask = input_padding_mask[..., -kv_len:]
                else:
                    # Padding with -inf for positions beyond the original mask
                    input_padding_mask = torch.nn.functional.pad(
                        input_padding_mask,
                        (0, kv_len - input_padding_mask.size(-1)),
                        value=float("-inf"),
                        mode="constant",
                    )

            # Ensuring the causal+window mask has the same batch size as the padding mask
            if causal_window_mask.size(0) != input_padding_mask.size(0):
                causal_window_mask = causal_window_mask.expand(
                    input_padding_mask.size(0), -1, -1, -1
                )

            # PART 3: Combine all masks
            # Adding the masks (both contain 0 for attended positions, -inf for masked)
            # This effectively applies an AND operation for allowed attention positions
            combined_mask = causal_window_mask + input_padding_mask
            return combined_mask

        # If no padding mask, just return the causal + sliding window mask
        return causal_window_mask
