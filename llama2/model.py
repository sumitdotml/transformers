import math
from typing import Tuple, Optional, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


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

        # (for ↑) Or I could do this (same thing, just different syntax):
        # x_even = x[:, :, :, 0::2]
        # x_odd = x[:, :, :, 1::2]

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

        # Aplying RoPE using the static helper method
        # The apply_rope method handles broadcasting cos/sin to match x's dimensions
        return self.apply_rope(x, cos_values, sin_values)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization

    Args:
        d_model (int): The dimension of the model.
        eps (float): A small constant to avoid division by zero.

    Returns:
        torch.Tensor: The normalized tensor.

    Formula (as per the paper https://arxiv.org/pdf/1910.07467, page 3):
        ```
        RMS(x) = sqrt(1/n * sum((x_i)**2 for i in range(1, n))) # where x_i is the i-th element of x
        x_norm = (x / RMS(x)) * γ
        where γ (gamma) = a learnable parameter, can be initialized to 1 (in case of torch tensor, can be initialized to torch.ones(d_model))
        ```
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.eps = eps  # epsilon
        self.scale = nn.Parameter(torch.ones(d_model))  # called gamma in the paper

    def _RMS(self, x: torch.Tensor) -> torch.Tensor:
        # sum has dim=-1 because I'm normalizing over the last dimension
        # keepdim=True to keep the same shape as x
        # self.eps is added to ensure the denominator is not zero
        root_mean_square = torch.sqrt(
            torch.sum(x**2, dim=-1, keepdim=True) / x.shape[-1] + self.eps
        )
        return root_mean_square

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x / self._RMS(x)) * self.scale


class GroupedMultiQueryAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
        max_seq_len: int,
        rope_base: float,
    ) -> None:
        super().__init__()
        assert (
            hidden_dim % num_heads == 0
        ), "hidden_dim (or d_model) should be divisible by num_heads"

        assert (
            num_heads % num_kv_heads == 0
        ), "num_heads should be divisible by num_kv_heads since this is a GQA"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_dim // num_heads
        self.num_query_groups = num_heads // num_kv_heads

        self.W_q = nn.Linear(
            in_features=hidden_dim, out_features=hidden_dim, bias=False
        )
        self.W_k = nn.Linear(
            in_features=hidden_dim,
            out_features=num_kv_heads * self.head_dim,
            bias=False,
        )
        self.W_v = nn.Linear(
            in_features=hidden_dim,
            out_features=num_kv_heads * self.head_dim,
            bias=False,
        )
        self.W_o = nn.Linear(
            in_features=hidden_dim, out_features=hidden_dim, bias=False
        )
        self.rope = RoPE(d_model=self.head_dim, max_seq_len=max_seq_len, base=rope_base)

    def forward(
        self,
        x: torch.Tensor,
        offset: int = 0,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch, seq_len, hidden_dim = x.shape  # [batch, seq_len_curr, hidden_dim]
        q = self.W_q(x)  # [batch, seq_len_curr, hidden_dim]
        k = self.W_k(x)  # [batch, seq_len_curr, num_kv_heads * head_dim]
        v = self.W_v(x)  # [batch, seq_len_curr, num_kv_heads * head_dim]

        # [batch, seq_len_curr, hidden_dim] -> [batch, seq_len_curr, num_heads, head_dim] -> [batch, num_heads, seq_len_curr, head_dim]
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # [batch, seq_len_curr, num_kv_heads * head_dim] -> [batch, seq_len_curr, num_kv_heads, head_dim] ->
        # -> [batch, num_kv_heads, seq_len_curr, head_dim]
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.rope(q, offset=offset)  # [batch, num_heads, seq_len_curr, head_dim]
        k = self.rope(k, offset=offset)  # [batch, num_kv_heads, seq_len_curr, head_dim]

        cache_k, cache_v = (
            past_key_value if past_key_value is not None else (None, None)
        )
        if cache_k is not None:
            # concatenating along the sequence length dimension (dim=2)

            # k shape before: [batch, num_kv_heads, seq_len_curr, head_dim]
            # cache_k shape: [batch, num_kv_heads, seq_len_cache, head_dim]
            k = torch.cat(
                [cache_k, k], dim=2
            )  # dim = 2 means sequence length dimension

            # v shape before: [batch, num_kv_heads, seq_len_curr, head_dim]
            # cache_v shape: [batch, num_kv_heads, seq_len_cache, head_dim]
            v = torch.cat(
                [cache_v, v], dim=2
            )  # dim = 2 means sequence length dimension

            # k, v shape after: [batch, num_kv_heads, seq_len_total, head_dim]
            # (where seq_len_total = seq_len_curr + seq_len_cache)

        # Tuple containing tensors of shape [batch, num_kv_heads, seq_len_total, head_dim]
        current_key_value = (
            k,
            v,
        )

        # since the num_heads in q is not the same as the num_kv_heads in k and v, I need to match them
        # for instance, if num_heads = 8 and num_kv_heads = 4, I need to make num_kv_heads = 8 with the help
        # of torch's interpolation.
        k_repeated = k.repeat_interleave(
            self.num_query_groups, dim=1
        )  # [batch, num_kv_heads, seq_len_total, head_dim] -> [batch, num_heads, seq_len_total, head_dim]
        v_repeated = v.repeat_interleave(
            self.num_query_groups, dim=1
        )  # [batch, num_kv_heads, seq_len_total, head_dim] -> [batch, num_heads, seq_len_total, head_dim]

        #########################
        # Attention Calculation #
        #########################

        # q shape: [batch, num_heads, seq_len_curr, head_dim]
        # k_repeated transposed shape: [batch, num_heads, head_dim, seq_len_total]

        # attn_scores shape: [batch, num_heads, seq_len_curr, seq_len_total]
        attn_scores = (q @ torch.transpose(k_repeated, -1, -2)) / math.sqrt(
            self.head_dim
        )

        if attention_mask is not None:
            # reshaping mask from [batch, seq_len_total] to [batch, 1, 1, seq_len_total]
            # for proper broadcasting with attn_scores [batch, num_heads, seq_len_curr, seq_len_total]
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            attn_scores = attn_scores.masked_fill(attention_mask == 0, float("-inf"))

        # attn_weight shape: [batch, num_heads, seq_len_curr, seq_len_total]
        attn_weight = torch.softmax(attn_scores, dim=-1)

        # v_repeated shape: [batch, num_heads, seq_len_total, head_dim]
        # output shape: [batch, num_heads, seq_len_curr, head_dim]
        output = attn_weight @ v_repeated

        ##################
        # Reshape Output #
        ##################

        # Input shape: [batch, num_heads, seq_len_curr, head_dim]
        # Output shape: [batch, seq_len_curr, num_heads, head_dim]
        output = output.transpose(1, 2).contiguous()

        # [batch, seq_len_curr, num_heads, head_dim] -> [batch, seq_len_curr, hidden_dim]
        output = output.view(batch, seq_len, self.hidden_dim)

        # [batch, seq_len_curr, hidden_dim] -> [batch, seq_len_curr, hidden_dim]
        output = self.W_o(output)

        # this current_key_value will be passed back as past_key_value for the next generation step
        return output, current_key_value


class DecoderBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
        ffn_dim: int,
        max_seq_len: int,
        rope_base: float,
        norm_eps: float,
    ) -> None:
        super().__init__()

        ##################################
        ###### Attention Components ######
        ##################################
        self.attn_norm = RMSNorm(d_model=hidden_dim, eps=norm_eps)
        self.attention = GroupedMultiQueryAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_seq_len=max_seq_len,
            rope_base=rope_base,
        )

        #############################
        ###### FFN Components #######
        #############################
        self.ffn_norm = RMSNorm(d_model=hidden_dim, eps=norm_eps)

        # 3 nn.Linear layers for the SwiGLU FFN
        self.w_gate = nn.Linear(
            in_features=hidden_dim, out_features=ffn_dim, bias=False
        )
        self.w_up = nn.Linear(in_features=hidden_dim, out_features=ffn_dim, bias=False)
        self.w_down = nn.Linear(
            in_features=ffn_dim, out_features=hidden_dim, bias=False
        )

    def forward(
        self,
        x: torch.Tensor,
        offset: int = 0,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        ##################################
        ######## Attention Block #########
        ##################################

        residual = x
        normalized_x = self.attn_norm(x)

        # GQA
        attn_output, current_key_value = self.attention(
            normalized_x, offset=offset, past_key_value=past_key_value
        )

        # ---------- Residual 1 ----------
        h = residual + attn_output  # h here is also called the "hidden state"

        ###################################
        ####### FFN Block (SwiGLU) ########
        ###################################

        residual = h
        normalized_h = self.ffn_norm(h)  # FFN pre-normalization

        # two projections for SwiGLU
        gate_projection = self.w_gate(normalized_h)  # Input to SiLU gate
        up_projection = self.w_up(normalized_h)  # Content to be gated

        # SiLU activation to the gate projection
        activated_gate = F.silu(gate_projection)

        # Element-wise multiplying gate and content projection (SwiGLU activation result)
        gated_content = activated_gate * up_projection

        # final down projection
        ffn_output = self.w_down(gated_content)

        # ---------- Residual 2 ----------
        out = residual + ffn_output

        return out, current_key_value


class Llama2(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_decoder_layers: int,
        num_heads: int,
        num_kv_heads: int,
        ffn_dim: int,
        max_seq_len: int,
        rope_base: float,
        norm_eps: float,
    ):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=hidden_dim
        )

        # Stacking Decoder Blocks
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    ffn_dim=ffn_dim,
                    max_seq_len=max_seq_len,
                    rope_base=rope_base,
                    norm_eps=norm_eps,
                )
                for _ in range(num_decoder_layers)
            ]
        )

        # RMSNorm after the Decoder Stack
        self.final_norm = RMSNorm(d_model=hidden_dim, eps=norm_eps)

        # Final Linear layer, often referred to as LM Head (language model head)
        self.lm_head = nn.Linear(
            in_features=hidden_dim, out_features=vocab_size, bias=False
        )

        # Weight tying
        self.lm_head.weight = self.embedding.weight

        # Max seq length storing if required
        self.max_seq_len = max_seq_len

    def forward(
        self,
        input_ids: torch.Tensor,
        offset: int = 0,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[Tuple[torch.Tensor, torch.Tensor]]]]:

        ########### 1. Embedding #############

        # input_ids [batch, seq_len] -> x (i.e., input embeddings) [batch, seq_len, hidden_dim]
        x = self.embedding(input_ids)

        # initializing cache storage
        is_prompt_processing = past_key_values is None
        new_caches = [] if not is_prompt_processing else None

        ######### 2. Decoder Blocks #############

        # looping through decoder blocks
        for i, decoder_block in enumerate(self.decoder_blocks):

            # getting the cache for the current layer
            layer_past_kv_cache = (
                past_key_values[i] if not is_prompt_processing else None
            )

            # passing input, offset, and layer-specific cache through the block
            x, current_kv_cache = decoder_block(
                x, offset=offset, past_key_value=layer_past_kv_cache
            )

            # storing the updated cache for this layer if generating
            if not is_prompt_processing:
                new_caches.append(current_kv_cache)

        returned_cache = tuple(new_caches) if not is_prompt_processing else None

        ###### 3. Final Normalization ########
        x = self.final_norm(x)

        ###### 4. Final Linear layer #######,
        logits = self.lm_head(x)  # [batch, seq_len, vocab_size]

        return logits, returned_cache
