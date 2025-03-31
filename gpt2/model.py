import torch
import torch.nn as nn
from transformers import GPT2Tokenizer  # , GPT2Config

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
input_text = "Another day of waking up"
input_tensor = tokenizer.encode(input_text, return_tensors="pt")

# this is the config of the gpt2 model from the transformers library
config = {
    "activation_function": "gelu_new",
    "qkv_bias": False,
    "attn_pdrop": 0.1,
    "bos_token_id": 50256,
    "embd_pdrop": 0.1,
    "eos_token_id": 50256,
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "model_type": "gpt2",
    "embed_dim": 768,
    "n_head": 12,
    "n_inner": None,
    "n_layer": 12,
    "context_len": 1024,
    "reorder_and_upcast_attn": False,
    "resid_pdrop": 0.1,
    "scale_attn_by_inverse_layer_idx": False,
    "scale_attn_weights": True,
    "summary_activation": None,
    "summary_first_dropout": 0.1,
    "summary_proj_to_labels": True,
    "summary_type": "cls_index",
    "summary_use_proj": True,
    "transformers_version": "4.50.3",
    "use_cache": True,
    "vocab_size": 50257,
}


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        dropout: float,
        context_length: int,
        n_head: int = 12,
        qkv_bias=False,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.n_head = n_head
        self.context_length = context_length

        assert d_out % n_head == 0, "d_out has to be divisible by n_head"

        self.head_dim = d_out // n_head  # same as d_k in the original transformer paper

        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_out, d_out)  # same as self.W_o
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, encoded_embedding):
        batch_size, seq_len, d_in = encoded_embedding.shape
        q = self.W_q(encoded_embedding)
        k = self.W_k(encoded_embedding)
        v = self.W_v(encoded_embedding)

        ### 1. q,  k, v
        # [batch_size, seq_len, d_in] -> [batch_size, seq_len, n_head, head_dim] -> [batch_size, n_head, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        ### 2. Scaled dot-product attention
        # [batch_size, n_head, seq_len, head_dim] @ [batch_size, n_head, head_dim, seq_len] -> [batch_size, n_head, seq_len, seq_len]
        attn_scores = q @ k.transpose(-2, -1)

        ### 2.1 Causal Masking
        mask_bool = self.mask.bool()[:seq_len, :seq_len]

        attn_scores_w_causal_mask = torch.masked_fill(
            attn_scores, mask_bool, value=torch.tensor(float("-inf"))
        )

        # [batch_size, n_head, seq_len, seq_len]
        attn_weights = torch.softmax(
            attn_scores_w_causal_mask / q.shape[-1] ** 0.5, dim=-1
        )

        ### 2.2 Dropout applied to attention weights
        attn_weights = self.dropout(attn_weights)

        ### 2.3 attn_weights @ v
        # [batch_size, n_head, seq_len, seq_len] @ [batch_size, n_head, seq_len, head_dim] = [batch_size, n_head, seq_len, head_dim]
        output = attn_weights @ v

        ### 3. Reshaping and combining back the heads with head_dim
        # [batch_size, seq_len, n_head, head_dim]
        output = output.transpose(1, 2)

        # [batch_size, seq_len, n_head, head_dim] -> [batch_size, seq_len, d_out]
        output = output.contiguous().view(batch_size, seq_len, self.d_out)

        ### 4. Projection Layer
        output = self.out_proj(output)

        return output


# -- LayerNorm
class LayerNorm(nn.Module):
    """
    LayerNorm class.

    Args:
        n_embd (int): The dimension of the input tensor, i.e, the embedding dimension.

    Returns:
        torch.Tensor: The normalized tensor.
    """

    def __init__(self, n_embd):
        super().__init__()
        self.epsilon = 1e-5
        self.scale = nn.Parameter(torch.ones(n_embd))
        self.shift = nn.Parameter(torch.zeros(n_embd))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.epsilon)
        return self.scale * norm_x + self.shift


# -- FeedForward (has GELU)
class FeedForward(nn.Module):
    """
    FeedForward class.

    Args:
        n_embd (int): The dimension of the input tensor, i.e, the embedding dimension.

    Returns:
        torch.Tensor: The output tensor.
    """

    def __init__(self, n_embd):
        super().__init__()
        self.linear1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear2 = nn.Linear(4 * n_embd, n_embd)
        self.gelu = nn.GELU(approximate="tanh")

    def forward(self, x):
        return self.linear2(self.gelu(self.linear1(x)))


# -- Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.layer_norm1 = LayerNorm(config["embed_dim"])
        self.dropout_after_mha = nn.Dropout(config["resid_pdrop"])
        self.dropout_after_ffn = nn.Dropout(config["resid_pdrop"])
        # =============================== REGARDING DROPOUTS ===============================
        # MultiHeadAttention uses the dropout `attn_pdrop` internally on the attention weights.
        # The `resid_pdrop` is applied to the output of the entire MHA module before the residual addition.
        # And also applied to the output of the feed forward layer before the residual addition.
        # =====================================================================================
        self.mha = MultiHeadAttention(
            d_in=config["embed_dim"],
            d_out=config["embed_dim"],
            dropout=config["attn_pdrop"],  # this
            context_length=config["context_len"],
            n_head=config["n_head"],
            qkv_bias=config["qkv_bias"],
        )
        self.layer_norm2 = LayerNorm(config["embed_dim"])
        self.ff = FeedForward(config["embed_dim"])

    def forward(self, x):
        # print(f"    Input to LN1 shape: {x.shape}")
        ln1_out = self.layer_norm1(x)
        # print(f"    Output of LN1 shape: {ln1_out.shape}")
        mha_out = self.mha(ln1_out)
        # print(f"    Output of MHA shape: {mha_out.shape}")
        mha_out_dropout = self.dropout_after_mha(mha_out)
        # print(f"    Output of MHA Dropout shape: {mha_out_dropout.shape}")
        residual1 = x + mha_out_dropout
        # print(f"    Shape after 1st Residual Add: {residual1.shape}")

        # print(f"    Input to LN2 shape: {residual1.shape}")
        ln2_out = self.layer_norm2(residual1)
        # print(f"    Output of LN2 shape: {ln2_out.shape}")
        ffn_out = self.ff(ln2_out)
        # print(f"    Output of FFN shape: {ffn_out.shape}")
        ffn_out_dropout = self.dropout_after_ffn(ffn_out)
        # print(f"    Output of FFN Dropout shape: {ffn_out_dropout.shape}")
        residual2 = residual1 + ffn_out_dropout
        # print(f"    Shape after 2nd Residual Add (Block Output): {residual2.shape}")
        return residual2


# -- GPT2 Model
class GPT2(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.token_embed = nn.Embedding(
            num_embeddings=config["vocab_size"], embedding_dim=config["embed_dim"]
        )
        self.pos_encoding = nn.Embedding(
            num_embeddings=config["context_len"], embedding_dim=config["embed_dim"]
        )
        self.embeddings_dropout = nn.Dropout(config["embd_pdrop"])

        # ModuleList provides easier access to individual blocks if needed later
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config["n_layer"])]
        )

        # Sequential provides a more efficient way to apply multiple layers. so I'll keep it here for now.
        # self.blocks = nn.Sequential(
        #     *[TransformerBlock(config) for _ in range(config["n_layer"])]
        # )
        self.final_layer_norm = LayerNorm(config["embed_dim"])
        self.projection = nn.Linear(
            config["embed_dim"], config["vocab_size"], bias=False
        )

        # =============================== REGARDING TYING WEIGHTS ===============================
        # standard practice in GPT-2 (and many other language models) to tie the weights of the input token embedding layer (self.token_embed) and the final output projection layer (self.projection).
        # This saves parameters and often improves performance slightly.
        # =====================================================================================
        self.projection.weight = self.token_embed.weight

    def forward(self, x):
        batch_size, seq_len = x.shape
        device = x.device

        # 1. Token Embeddings
        token_embed_out = self.token_embed(x)
        # print(f"      Token Embeddings shape: {token_embed_out.shape}")

        # 2. Positional Embeddings
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_encoding_out = self.pos_encoding(position_ids)
        # print(f"      Position IDs shape: {position_ids.shape}")
        # print(f"      Positional Embeddings shape: {pos_encoding_out.shape}")

        # 3. Combining Token Embeddings and Positional Embeddings & Dropout
        embeddings = token_embed_out + pos_encoding_out
        # print(f"      Combined Embeddings shape: {embeddings.shape}")
        embeddings_w_dropout_out = self.embeddings_dropout(embeddings)
        # print(f"      Embeddings after Dropout shape: {embeddings_w_dropout_out.shape}")

        # 4. Transformer Blocks
        transformer_block_input = embeddings_w_dropout_out
        for i, block in enumerate(self.blocks):
            # print(f"    Entering Transformer Block {i+1}/{self.config['n_layer']}")
            transformer_block_input = block(transformer_block_input)
            # print(f"    Output of Transformer Block {i+1} shape: {transformer_block_input.shape}")
        transformer_output = transformer_block_input  # Output of the last block

        # 5. Final Layer Norm
        layer_norm_out = self.final_layer_norm(transformer_output)
        # print(f"      Final LayerNorm Output shape: {layer_norm_out.shape}")

        # 6. Projection to Logits
        projection_aka_logits = self.projection(layer_norm_out)
        # print(f"      Projection (Logits) shape: {projection_aka_logits.shape}")

        return projection_aka_logits


# ============================================================
# Detailed Print Execution Start
# ============================================================
if __name__ == "__main__":
    print("--- Starting GPT-2 Forward Pass Trace ---")

    # --- Input ---
    print(f"\n1. Input:")
    print(f"  Input Text: '{input_text}'")
    print(f"  Input Tensor (Token IDs): {input_tensor}")
    print(f"  Input Tensor Shape: {input_tensor.shape} (Batch Size, Sequence Length)")

    # --- Model Instantiation ---
    print(f"\n2. Model Instantiation:")
    gpt2 = GPT2(config)
    gpt2.eval()  # Setting model to evaluation mode to disable dropout for stable shape checking
    print(f"  Instantiated GPT2 model with {config['n_layer']} layers.")
    device = input_tensor.device  # Assuming input_tensor is on the correct device
    gpt2.to(device)
    print(f"  Model moved to device: {device}")

    # --- Embedding Layer ---
    print(f"\n3. Embeddings:")
    batch_size, seq_len = input_tensor.shape
    print(f"  Batch Size: {batch_size}, Sequence Length: {seq_len}")

    # Token Embeddings
    token_embeddings = gpt2.token_embed(input_tensor)
    print(
        f"  Token Embeddings shape: {token_embeddings.shape} (Batch, SeqLen, EmbedDim)"
    )

    # Positional Embeddings
    position_ids = torch.arange(seq_len, device=device).unsqueeze(
        0
    )  # Shape: (1, SeqLen)
    positional_embeddings = gpt2.pos_encoding(
        position_ids
    )  # Shape: (1, SeqLen, EmbedDim)
    print(f"  Position IDs shape: {position_ids.shape}")
    print(f"  Positional Embeddings shape: {positional_embeddings.shape}")

    # Combining Token Embeddings and Positional Embeddings (Addition handles broadcasting of positional embeddings)
    combined_embeddings = token_embeddings + positional_embeddings
    print(
        f"  Combined Embeddings shape (Token + Positional): {combined_embeddings.shape}"
    )

    # Embedding Dropout (Note: In eval mode, dropout has no effect, shape remains same)
    embeddings_after_dropout = gpt2.embeddings_dropout(combined_embeddings)
    print(f"  Embeddings after Dropout shape: {embeddings_after_dropout.shape}")

    # --- Transformer Blocks ---
    print(f"\n4. Transformer Blocks (Tracing First Block in Detail):")
    block_input = embeddings_after_dropout
    first_block = gpt2.blocks[0]

    # Inside First Block
    print(f"  --- First Block Internal Trace ---")
    print(f"    Input to Block 1 shape: {block_input.shape}")

    # LayerNorm 1
    ln1_output = first_block.layer_norm1(block_input)
    print(f"    Output of LayerNorm1 shape: {ln1_output.shape}")

    # Multi-Head Attention
    mha_output = first_block.mha(ln1_output)
    print(f"    Output of MHA shape: {mha_output.shape}")

    # Residual Dropout 1 (No effect in eval mode)
    mha_output_dropout = first_block.dropout_after_mha(mha_output)
    print(f"    Output after MHA Dropout shape: {mha_output_dropout.shape}")

    # Residual Connection 1
    residual1_output = block_input + mha_output_dropout
    print(f"    Shape after 1st Residual Add: {residual1_output.shape}")

    # LayerNorm 2
    ln2_output = first_block.layer_norm2(residual1_output)
    print(f"    Output of LayerNorm2 shape: {ln2_output.shape}")

    # Feed Forward Network
    ffn_output = first_block.ff(ln2_output)
    print(f"    Output of FFN shape: {ffn_output.shape}")

    # Residual Dropout 2 (No effect in eval mode)
    ffn_output_dropout = first_block.dropout_after_ffn(ffn_output)
    print(f"    Output after FFN Dropout shape: {ffn_output_dropout.shape}")

    # Residual Connection 2
    block_output = residual1_output + ffn_output_dropout
    print(f"    Output of Block 1 shape: {block_output.shape}")
    print(f"  --- End First Block Internal Trace ---")

    # Output after ALL blocks (Iterating manually through ModuleList)
    print(f"\n  Calculating output after all {config['n_layer']} blocks...")
    current_block_output = (
        embeddings_after_dropout  # Starting with the input to the first block
    )
    for i, block in enumerate(gpt2.blocks):
        current_block_output = block(current_block_output)
        print(f"    Shape after block {i+1}: {current_block_output.shape}")
    all_blocks_output = (
        current_block_output  # This holds the final output after the loop
    )
    print(
        f"  Output shape after ALL {config['n_layer']} blocks: {all_blocks_output.shape}"
    )

    # --- Final Layers ---
    print(f"\n5. Final Layers:")

    # Final LayerNorm
    final_ln_output = gpt2.final_layer_norm(all_blocks_output)
    print(f"  Output of Final LayerNorm shape: {final_ln_output.shape}")

    # Final Projection
    logits = gpt2.projection(final_ln_output)
    print(
        f"  Output of Final Projection (Logits) shape: {logits.shape} (Batch, SeqLen, VocabSize)"
    )

    # --- Full Model Forward Pass (for comparison) ---
    print(f"\n6. Full Model Forward Pass Output:")
    with torch.no_grad():  # Ensuring no gradients are computed
        full_output = gpt2(input_tensor)
    print(f"  Shape returned by gpt2.forward(input_tensor): {full_output.shape}")
    print(
        f"  Does final traced shape match full forward pass shape? {logits.shape == full_output.shape}"
    )

    print("\n--- End GPT-2 Forward Pass Trace ---")
