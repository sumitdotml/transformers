# README

ToC

- [My GPT-2 Implementation: Step-by-Step Forward Pass Documentation](#my-gpt-2-implementation-step-by-step-forward-pass-documentation)
- [Notes](#notes)

#

## My GPT-2 Implementation: Step-by-Step Forward Pass Documentation <a name="my-gpt-2-implementation-step-by-step-forward-pass-documentation"></a>

Here's how I process input data through my GPT-2 model to generate output logits:

### 1. Input Preparation:

First, I take the raw input text (e.g., `"Another day of waking up"`).
I use the pre-trained GPT-2 tokenizer (`tokenizer = GPT2Tokenizer.from_pretrained("gpt2")`) to convert this text into a sequence of token IDs.
This results in an input tensor, let's call it `x`, with the shape `[batch_size, seq_len]`. For my example text, this might look like `tensor([[ 6610, 1110, 286, 23137, 510]])`, having a shape of `[1, 5]`.

### 2. Generating Embeddings:

**Token Embeddings:**

I pass the input tensor `x` through my token embedding layer (self.token_embed = nn.Embedding(...)). This layer looks up the vector representation for each token ID in its table. The output, `token_embed`, has the shape `[batch_size, seq_len, embed_dim]` (e.g., `[1, 5, 768]`).

**Positional Embeddings:**

Since the Transformer architecture itself doesn't inherently know about token order, I need to add positional information. I use a learned positional embedding layer (self.pos_encoding = nn.Embedding(config["context_len"], config["embed_dim"])).

- I first create a tensor of position indices for the current sequence length: `position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)` which gives a shape like `[1, seq_len]` (e.g., `[1, 5]`).
- I pass these `position_ids` through the self.pos_encoding layer to get the positional embeddings, `pos_encoding`, which will have a shape like `[1, seq_len, embed_dim]` (e.g., `[1, 5, 768]`).

**Combining Embeddings:**

I add the `token_embed` and `pos_encoding` tensors element-wise. PyTorch handles the broadcasting, effectively adding the unique positional vector for each position to the token vector at that position. The result, `embeddings`, still has the shape `[batch_size, seq_len, embed_dim]`.

### 3. Initial Dropout:

To help prevent overfitting during training, I apply dropout right after combining the embeddings. I pass the embeddings through `self.embeddings_dropout = nn.Dropout(config["embd_pdrop"])`. The output shape remains `[batch_size, seq_len, embed_dim]`. (Note: In `eval()` mode, this layer doesn't change the input). Let's call the result `embeddings_w_dropout`.

### 4. Processing Through Transformer Blocks:

This is the core of the model. The `embeddings_w_dropout` tensor now enters the sequence of Transformer blocks (`self.blocks = nn.ModuleList([...])`).
I iterate through each TransformerBlock in the `self.blocks` list, typically 12 times for the base GPT-2 model (`config["n_layer"]`).
The output of one block becomes the input for the next block. Let the input to a block be `block_input`.

**Inside a Single Transformer Block (TransformerBlock.forward(x)):**

(a) `Pre-Normalization 1`: I first apply Layer Normalization (self.layer_norm1) to the block_input. This stabilizes the activations.

(b) `Masked Multi-Head Self-Attention`: The normalized output goes into the `self.mha = MultiHeadAttention(...)` module.

- This module calculates Query (Q), Key (K), and Value (V) projections from the input.
- It computes scaled dot-product attention scores ($Q @ K.T / \sqrt{head\_dim}$).
- Crucially, it applies the causal mask (self.mask) using `masked_fill_` to prevent attention to future tokens (setting future scores to -inf).
- It applies dropout (`attn_pdrop`) to the attention weights after the softmax.
- It computes the weighted sum of Value vectors and projects the result back to embed_dim.

(c) `Residual Dropout 1`: The output of the MHA module is passed through a dropout layer (self.dropout_after_mha which uses config["resid_pdrop"]).

(d) `Residual Connection 1`: I add the result from step (c) back to the original input of the block (`block_input`). This skip connection is vital for training deep networks. Let's call this sum `residual1_output`.

(e) `Pre-Normalization 2`: I apply the second Layer Normalization (self.layer_norm2) to `residual1_output`.

(f) `Feed-Forward Network`: The normalized output goes into the `self.ff = FeedForward(...)` module. This typically consists of two linear layers with a GELU activation in between, expanding the dimension temporarily (4 * embed_dim).

(g) `Residual Dropout 2`: The output of the FFN module is passed through another dropout layer (`self.dropout_after_ffn` which also uses `config["resid_pdrop"]`).

(h) `Residual Connection 2`: I add the result from step (g) back to the input of the second LayerNorm (`residual1_output`). This is the final output of the Transformer block and has the shape `[batch_size, seq_len, embed_dim]`. This output then serves as the input to the next block in the sequence.

### 5. Final Layer Normalization:

After the data has passed through all `n_layer` Transformer blocks, I apply one final Layer Normalization (`self.final_layer_norm`) to the output of the last block. This ensures the input to the final projection layer is well-scaled.

### 6. Projection to Vocabulary (Logits):

Finally, I take the normalized output from the previous step and pass it through a linear layer (`self.projection = nn.Linear(...)`).

- This layer projects the embed_dim dimension down to the vocab_size (e.g., 50257).
- Crucially, the weights of this layer are tied to the weights of the initial token embedding layer (self.projection.weight = self.token_embed.weight).
- The output of this layer is the final result of the model: the logits. It has the shape [batch_size, seq_len, vocab_size]. Each value `logits[b, t, v]` represents the model's raw, unnormalized prediction score for token `v` being the next token following the token at position `t` in batch `b`. These logits can then be used (often with softmax) for text generation or passed into a loss function (like CrossEntropyLoss, which typically includes softmax) during training.

#

## Notes <a name="notes"></a>

### On dropouts

There are three distinct dropout applications in a standard GPT-2 model, often with different rates:

- Attention Weight Dropout (`attn_pdrop`): Applied inside the MultiHeadAttention module, directly to the attention weights after the softmax but before multiplying by the Value vectors. My [MultiHeadAttention class](model.py#L179) correctly uses `self.dropout` (initialized with `attn_pdrop`) for this purpose.
- Residual Path Dropout (`resid_pdrop`): Applied after the main operation of each sub-layer (MHA and FFN) within the TransformerBlock, but before adding the result back to the residual pathway (the skip connection). My [TransformerBlock class](model.py#L169-170) now correctly implements this using `self.dropout_after_mha` and `self.dropout_after_ffn`, both initialized with `resid_pdrop`.
- Embedding Dropout (`embd_pdrop`): Applied once, right after the token embeddings and positional embeddings have been summed together, before the input goes into the first TransformerBlock. My [GPT2 class](model.py#L224) forward method now correctly implements this using `self.embeddings_dropout`.