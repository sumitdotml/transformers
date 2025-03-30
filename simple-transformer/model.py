import torch
import torch.nn as nn
from typing import Optional

# first I'll assume that I already have an encoded embedding

torch.manual_seed(1793)
encoded_embedding = torch.randn(1, 6, 512)  # [batch, seq_len, d_model]


def create_padding_mask(input_ids: torch.Tensor, padding_idx: int = 0) -> torch.Tensor:
    """
    Creates a padding mask for multi-head attention based on input token IDs.

    The mask identifies positions in the sequence that correspond to padding tokens.
    It's shaped for broadcasting compatibility with attention scores inside MHA.

    Convention:
        - 1.0 : Represents a token that should be attended to (kept).
        - 0.0 : Represents a padding token that should be masked out (ignored).

    Args:
        input_ids (torch.Tensor): A tensor of token IDs with shape
            [batch_size, sequence_length].
        padding_idx (int, optional): The index representing the padding token
            in the vocabulary. Defaults to 0.

    Returns:
        torch.Tensor: A float tensor representing the padding mask with shape
            [batch_size, 1, 1, sequence_length]. Ready to be used in
            MultiHeadAttention.
    """
    # 1. Create boolean mask: True where input is NOT padding, False where it IS padding.
    # Shape: [batch_size, sequence_length]
    mask = input_ids != padding_idx

    # 2. Convert boolean mask to float (True -> 1.0, False -> 0.0).
    # This matches the convention needed for `masked_fill(mask == 0, -inf)`.
    # Shape: [batch_size, sequence_length]
    mask = mask.float()

    # 3. Reshape for broadcasting within MHA.
    # Add dimensions for `num_heads` (dim 1) and `query_sequence_length` (dim 2).
    # The mask applies based on the key sequence length (the last dimension).
    # Shape: [batch_size, 1, 1, sequence_length]
    # The mask tensor will automatically be on the same device as input_ids.
    return mask.unsqueeze(1).unsqueeze(2)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, d_model: int, dropout: float, qkv_bias=False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        assert (
            d_model % num_heads == 0
        ), f"""
        d_model is not divisible by num_heads.
        Received d_model: {d_model}, num_heads: {num_heads}"""

        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.W_k = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.W_v = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.W_o = nn.Linear(d_model, d_model, bias=qkv_bias)

    @staticmethod
    def scaled_dot_product_attention(
        query, key, value, mask=None, dropout: Optional[nn.Dropout] = None
    ):
        # [batch, q_seq_len, d_model] @ [batch, d_model, k_seq_len] -> [batch, q_seq_len, k_seq_len]
        # or in case of multihead:
        # [batch, num_heads, q_seq_len, d_model] @ [batch, num_heads, d_model, k_seq_len] -> [batch, num_heads, q_seq_len, k_seq_len]
        attn_scores = query @ key.transpose(-2, -1) / key.shape[-1] ** 0.5

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        if dropout is not None:
            attn_weights = dropout(attn_weights)
        attn = attn_weights @ value
        return attn, attn_weights

    def forward(self, query_input, key_input=None, value_input=None, mask=None):
        # Defaulting to self-attention if key/value inputs are not provided
        is_self_attention = key_input is None and value_input is None
        if is_self_attention:
            key_input = query_input
            value_input = query_input

        batch_size, query_seq_len, _ = query_input.shape
        _, key_seq_len, _ = key_input.shape
        _, value_seq_len, _ = value_input.shape

        q = self.W_q(query_input)  # [batch, seq_len, d_model]
        k = self.W_k(key_input)  # [batch, seq_len, d_model]
        v = self.W_v(value_input)  # [batch, seq_len, d_model]

        # [batch, seq_len, d_model] -> [batch, seq_len, num_heads, d_k] -> [batch, num_heads, seq_len, d_k]
        q = q.view(batch_size, query_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # or I can also write: q = q.view(batch, -1, self.num_heads, self.d_k) since
        # -1 basically means 'please infer the remaining value'

        k = k.view(batch_size, key_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, value_seq_len, self.num_heads, self.d_k).transpose(1, 2)

        x, attn_weights = MultiHeadAttention.scaled_dot_product_attention(
            query=q, key=k, value=v, dropout=self.dropout, mask=mask
        )

        # [batch, num_heads, seq_len, d_k] -> [batch, seq_len, num_heads, d_k] -> [batch, seq_len, d_model]
        x = x.transpose(1, 2)
        x = x.contiguous().view(
            batch_size, query_seq_len, -1
        )  # output is the query sequence length
        x = self.W_o(x)
        return x, attn_weights


# --- Demonstration of Integration (Simulating usage in a larger model) ---
def run_transformer_step_simulation():
    print("--- Simulating a step in a Transformer ---")

    # --- 1. Data Preparation (Typical in DataLoader/Batching) ---
    padding_idx = 0
    # Batch of token IDs (batch_size=3, seq_len=7)
    input_ids_batch = torch.tensor(
        [
            [101, 567, 890, 102, padding_idx, padding_idx, padding_idx],  # Seq len 4
            [101, 432, 102, padding_idx, padding_idx, padding_idx, padding_idx],  # Seq len 3
            [101, 666, 777, 888, 999, 555, 102],  # Seq len 7
        ]
    ).long()  # Ensure IDs are long type for embedding layer

    batch_size, seq_len = input_ids_batch.shape
    d_model = 512
    num_heads = 8
    vocab_size = 1000  # Example vocabulary size

    # Assume running on GPU if available
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.mps.is_available()
        else "cpu"
    )
    input_ids_batch = input_ids_batch.to(device)
    print(f"Running on device: {device}")
    print(f"Input IDs shape: {input_ids_batch.shape}")

    # --- 2. Mask Creation (Happens *before* calling MHA) ---
    # This would typically be done early in the forward pass of an Encoder/Decoder layer
    padding_mask = create_padding_mask(input_ids_batch, padding_idx=padding_idx)
    # Mask is automatically created on the same device as input_ids_batch
    print(f"Padding Mask shape: {padding_mask.shape}")
    print(f"Padding Mask (Batch 1, squeezed): {padding_mask[1].squeeze()}")

    # --- 3. Embedding Layer (Standard Transformer component) ---
    embedding_layer = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx).to(
        device
    )
    # Note: nn.Embedding can optionally handle zeroing out padding embeddings,
    # but the attention mask is still crucial for the attention mechanism itself.
    embeddings = embedding_layer(input_ids_batch)
    print(f"Embeddings shape: {embeddings.shape}")

    # --- 4. Multi-Head Attention Layer ---
    mha_layer = MultiHeadAttention(
        num_heads=num_heads, d_model=d_model, dropout=0.1
    ).to(device)

    # --- 5. Applying MHA (Passing embeddings and the pre-computed mask) ---
    # Simulating self-attention within an Encoder layer
    print("\n--- Applying Multi-Head Self-Attention ---")
    attention_output, attention_weights = mha_layer(
        query_input=embeddings,
        key_input=embeddings,  # Self-attention
        value_input=embeddings,  # Self-attention
        mask=padding_mask,  # Pass the generated padding mask HERE
    )

    print(f"Attention Output shape: {attention_output.shape}")
    print(f"Attention Weights shape: {attention_weights.shape}")

    # --- Verification (Check if padding was ignored) ---
    print("\n--- Verifying Masking ---")
    # Check attention weights for a query position in a padded sequence (Batch 1)
    # Example: Query at position 0 (token 101) in Batch 1
    # It should *not* attend to key positions 3, 4, 5, 6 (indices) which are padding.
    print("Attention weights (Batch 1, Head 0, Query Pos 0):")
    print(attention_weights[1, 0, 0, :])  # Weights for keys
    # Expect weights at indices 3, 4, 5, 6 to be zero or extremely close to zero.

    # Example: Query at position 2 (token 102) in Batch 1
    print("Attention weights (Batch 1, Head 0, Query Pos 2):")
    print(attention_weights[1, 0, 2, :])  # Weights for keys
    # Expect weights at indices 3, 4, 5, 6 to be zero or extremely close to zero.

    # Check attention weights for a query position in a non-padded sequence (Batch 2)
    print("Attention weights (Batch 2, Head 0, Query Pos 0):")
    print(attention_weights[2, 0, 0, :])  # Weights for keys
    # Expect all weights to be non-zero (unless dropout made some zero by chance).


if __name__ == "__main__":
    run_transformer_step_simulation()
