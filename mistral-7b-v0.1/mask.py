import torch
import torch.nn as nn
from typing import Optional


class SlidingWindowMask(nn.Module):
    """
    Implements causal sliding window attention masking.

    This class handles the creation of attention masks for sliding window attention,
    where each query position can only attend to a fixed window of previous tokens
    to limit the computational complexity for long sequences.
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
        Create a causal attention mask for sliding window attention.

        Args:
            batch_size: Batch size of the input
            q_len: Length of the current query sequence
            kv_len: Length of the key/value window from cache
            offset: Absolute starting position of the query sequence
            input_padding_mask: Optional padding mask (B, 1, Q, K) or (B, 1, 1, K) with -inf for masked positions
            device: Device for the mask
            dtype: Data type for the mask

        Returns:
            Combined causal and padding mask (batch_size, 1, q_len, kv_len)
        """
        # Generating the causal mask for the sliding window attention
        q_indices = torch.arange(q_len, device=device)  # [0, 1, ..., q_len-1]
        k_indices = torch.arange(kv_len, device=device)  # [0, 1, ..., kv_len-1]

        # Computing the absolute position of each query token
        q_positions = offset + q_indices  # [offset, offset+1, ..., offset+q_len-1]

        # Computing the minimum absolute position each key token could have
        # based on the sliding window size
        min_k_position = q_positions.unsqueeze(-1) - self.sliding_window + 1

        # Converting the key indices to the absolute positions
        # For the sliding window, the key positions start at (offset + q_len - kv_len)
        # This is because we're looking at the last kv_len tokens
        k_start = offset + q_len - kv_len
        k_positions = k_start + k_indices  # [k_start, k_start+1, ..., k_start+kv_len-1]

        # A query at position i can attend to keys from (i-sliding_window+1) up to position i
        # Creating the mask where -inf means "don't attend"
        # We mask if key position < min_key_position (too far back) or key position > query position (in the future)
        k_positions = k_positions.unsqueeze(0)  # [1, kv_len]
        min_k_position = min_k_position  # [q_len, 1]
        q_positions = q_positions.unsqueeze(-1)  # [q_len, 1]

        # Mask where the key is too far back or in the future
        mask_condition = (k_positions < min_k_position) | (k_positions > q_positions)
        causal_mask = torch.where(
            mask_condition,
            torch.tensor(float("-inf"), device=device, dtype=dtype),
            torch.tensor(0.0, device=device, dtype=dtype),
        )  # [q_len, kv_len]

        # Expanding the mask to match the expected shape [batch_size, 1, q_len, kv_len]
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        # Applying the batch dimension if needed
        if batch_size > 1:
            causal_mask = causal_mask.expand(batch_size, 1, q_len, kv_len)

        # Combining with the padding mask if provided
        if input_padding_mask is not None:
            # Processing the padding mask to match the dimensions
            if input_padding_mask.size(2) == 1 and q_len > 1:
                # Expanding the singleton query dimension
                input_padding_mask = input_padding_mask.expand(-1, -1, q_len, -1)

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

            # Ensuring the causal mask has the same batch size as the padding mask
            if causal_mask.size(0) != input_padding_mask.size(0):
                causal_mask = causal_mask.expand(input_padding_mask.size(0), -1, -1, -1)

            # Adding the masks (both contain 0 for attended positions, -inf for masked)
            return causal_mask + input_padding_mask

        return causal_mask
