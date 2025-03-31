import torch
import torch.nn as nn
from transformers import GPT2Tokenizer  # , GPT2Config

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
input_text = "Another day of waking up"
input_tensor = tokenizer.encode(input_text, return_tensors="pt")

# this is the config of the gpt2 model from the transformers library
config = {
    "activation_function": "gelu_new",
    "attn_pdrop": 0.1,
    "bos_token_id": 50256,
    "embd_pdrop": 0.1,
    "eos_token_id": 50256,
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "model_type": "gpt2",
    "n_embd": 768,
    "n_head": 12,
    "n_inner": None,
    "n_layer": 12,
    "n_positions": 1024,
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


def embedder_function(x):
    embedder = nn.Embedding(
        num_embeddings=config.vocab_size,  # 50257
        embedding_dim=config.n_embd,  # 768
    )
    return embedder(x)


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


if __name__ == "__main__":
    mha = MultiHeadAttention(
        d_in=config.n_embd,
        d_out=config.n_embd,
        dropout=config.attn_pdrop,
        context_length=config.n_positions,
        n_head=config.n_head,
    )

    encoded_embedding = embedder_function(input_tensor)
    print(f"encoded_embedding.shape: {encoded_embedding.shape}")
    print(f"shape of mha(encoded_embedding): {mha(encoded_embedding).shape}")
