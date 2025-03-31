# README

## ToC

- [My GPT-2 Implementation: Step-by-Step Forward Pass Documentation](#my-gpt-2-implementation-step-by-step-forward-pass-documentation)
- [Notes](#notes)
    - [On Dropouts](#on-dropouts)
    - [On Embedding Dimensions (768 vs 50257) and Comparison to Original Transformer](#on-embedding-dimensions-768-vs-50257-and-comparison-to-original-transformer)
    - [On the difference between `context_length` and `seq_len`](#on-the-difference-between-context_length-and-seq_len)

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

### On dropouts <a name="on-dropouts"></a>

There are three distinct dropout applications in a standard GPT-2 model, often with different rates:

- Attention Weight Dropout (`attn_pdrop`): Applied inside the MultiHeadAttention module, directly to the attention weights after the softmax but before multiplying by the Value vectors. My [MultiHeadAttention class](model.py#L179) correctly uses `self.dropout` (initialized with `attn_pdrop`) for this purpose.
- Residual Path Dropout (`resid_pdrop`): Applied after the main operation of each sub-layer (MHA and FFN) within the TransformerBlock, but before adding the result back to the residual pathway (the skip connection). My [TransformerBlock class](model.py#L169-170) now correctly implements this using `self.dropout_after_mha` and `self.dropout_after_ffn`, both initialized with `resid_pdrop`.
- Embedding Dropout (`embd_pdrop`): Applied once, right after the token embeddings and positional embeddings have been summed together, before the input goes into the first TransformerBlock. My [GPT2 class](model.py#L224) forward method now correctly implements this using `self.embeddings_dropout`.

### On Embedding Dimensions (768 vs 50257) and Comparison to Original Transformer <a name="on-embedding-dimensions-768-vs-50257-and-comparison-to-original-transformer"></a>

The embedding dimensions are 768 throughout the whole process in GPT-2 and reach 50257 at the very end (due to [final nn.Linear projection](model.py#L232) having `vocab_size` as `d_out`). The original transformer had 512 `d_model` number from the embedding step all the way to the output step. But GPT-2 goes from 768 to 50257. The logic?

Internal Dimension (`embed_dim = 768`): Throughout the model, from the initial combined embeddings through all 12 Transformer blocks and the final LayerNorm, the tensor shape for each token's representation is `[batch_size, seq_len, 768]`. This 768 is the model's internal working dimension (`d_model` in the original paper's terminology, `embed_dim` in my config).

**Logic:** This dimension represents the "richness" or "capacity" of the vector space where the model learns to represent tokens and their context. All the core components (MHA, FFN, LayerNorm) are designed to operate on vectors of this size and produce output vectors of the same size. This allows the blocks to be stacked sequentially – the output of one block is a valid input for the next. As data flows through the layers, the meaning captured within these 768 dimensions becomes increasingly complex and context-aware, but the size of the representation space for each token remains constant.

Final Projection (vocab_size = 50257): The very last step uses:

```python
self.projection = nn.Linear(config["embed_dim"], config["vocab_size"], bias=False).
```

**Logic:** The ultimate goal of GPT-2 (as a generative language model) is to predict the next token in the sequence. To do this, for each position in the input sequence, it needs to produce a score for every possible token in its vocabulary. <a style="color: skyblue; font-weight: bold;">The vocabulary size for GPT-2 is 50257</a>. Therefore, the final linear layer takes the final 768-dimensional representation of each token (which encodes all the contextual understanding) and projects it into a 50257-dimensional vector. Each element in this vector corresponds to a token in the vocabulary, and its value (the logit) represents the model's confidence that this token should come next.

**Comparison to Original Transformer:** The original Transformer paper indeed used `d_model=512`. However, the principle was exactly the same. The encoder and decoder stacks maintained the 512 dimension internally. The very final step of the original Transformer's decoder also involved a linear layer that projected the 512-dimensional output vectors to the vocabulary size to produce logits for the next token prediction.

Therefore, my GPT-2 implementation follows the same architectural pattern as the original Transformer in this regard: maintain a constant internal dimension (`d_model/embed_dim`) throughout the main processing blocks, and only project to the vocabulary size (`vocab_size`) at the very end to get the prediction scores (`logits`). The specific numbers (768 vs 512, 50257 vs the original paper's vocab size) have changed, but the concept hasn't.

> [!NOTE]
> TL;DR: The model works internally with rich 768-dimensional vectors. Only at the final step does it map these internal representations to scores across the entire vocabulary (50257 options) to make its next-token prediction.

### On the difference between `context_length` and `seq_len` <a name="on-the-difference-between-context_length-and-seq_len"></a>

The original Transformer paper used a fixed `context_length` of 512. Why do we have two different parameters in GPT-2 (fixed `context_length` of 1024 and variable `seq_len`)?

This is a critical concept for practical Transformer implementations, especially for models like GPT-2 that handle variable-length text.

**context_length (e.g., 1024):** The Maximum Capacity

This is a fixed parameter defined when I initialize the model (`__init__`). It dictates the absolute maximum sequence length the model architecture is prepared to handle.

Why is it needed? Certain components need to be pre-allocated with a fixed size:

- Positional Embeddings (self.pos_encoding): This is an nn.Embedding layer. It needs to know the maximum number of positions it might ever need to provide an embedding for. So, it's created with `config["context_len"]` rows (`nn.Embedding(config["context_len"], config["embed_dim"])`). It holds 1024 unique positional vectors.
- Causal Attention Mask (self.mask): In my implementation, I pre-compute the upper triangular mask using `torch.ones(context_length, context_length)`. Doing this once in `__init__` and storing it in a buffer is more efficient than recreating it on every forward pass. This buffer holds the mask pattern for the largest possible attention matrix the model might encounter.

Analogy: I can think of `context_length` as the number of seats built into a theater. The theater is constructed with a fixed maximum capacity (1024 seats).

**seq_len (e.g., 5):** The Actual Input Length

This is a dynamic value that depends on the specific input tensor we provide during a forward pass. It's determined by `input_tensor.shape[1]`. It represents the actual number of tokens in the sequence(s) being processed right now.

Why is it needed? Real-world text comes in all lengths. We don't want to force every input to be exactly 1024 tokens long (by excessive padding) – that would be incredibly inefficient. The model needs to adapt to the actual length of the input it receives.

How is it used?

- Positional Embeddings: We only look up embeddings for the positions actually present: `torch.arange(seq_len, ...)` generates indices from 0 to `seq_len-1`, and these are used to query the `self.pos_encoding` table.
- Causal Attention Mask: We slice the pre-computed full mask to fit the current sequence length: `self.mask.bool()[:seq_len, :seq_len]`. This ensures that the attention calculation ($Q @ K.T$) and the masking only operate on the relevant `[seq_len, seq_len]` portion corresponding to the actual input tokens.

Analogy: I can think of `seq_len` as the number of people who actually came to the theater for tonight's show (5 people). I only need to interact with those 5 people and manage their seating/interactions, even though the theater could hold 1024.

**Comparison to Original Transformer:** The original paper often described experiments where inputs might have been padded or batched to a fixed length for simplicity. However, the core self-attention mechanism can handle variable lengths. Modern implementations like the one I implemented make this dynamic handling explicit and efficient. The distinction between the pre-allocated maximum (`context_length`) and the runtime actual (`seq_len`) is essential for building flexible models that don't waste computation on padding. GPT-2 was designed from the ground up to work effectively with the variable lengths common in natural language text.

> [!NOTE]
> TL;DR: `context_length` sets the architectural limit and allows for efficient pre-allocation of resources. `seq_len` is the actual length of the current input, used dynamically during the forward pass to ensure the model only processes and attends to the relevant parts of the sequence and mask. This makes the model flexible and efficient for real-world text.