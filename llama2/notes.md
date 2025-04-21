# Notes

## Important cache bits in the model.py file

So there's this following code snippet in the `forward` method of the `GroupedMultiQueryAttention` class:

```python
# .... previous code ....

# 1. Retrieve cached states (if any)
cache_k, cache_v = (
    past_key_value if past_key_value is not None else (None, None)
)

# 2. Check if we have a history
if cache_k is not None:
    # 3. Concatenate history with current states

    # k shape before: [batch, num_kv_heads, seq_len_curr, head_dim]
    # cache_k shape: [batch, num_kv_heads, seq_len_cache, head_dim]
    k = torch.cat([cache_k, k], dim=2) # Join along sequence length dim

    # v shape before: [batch, num_kv_heads, seq_len_curr, head_dim]
    # cache_v shape: [batch, num_kv_heads, seq_len_cache, head_dim]
    v = torch.cat([cache_v, v], dim=2) # Join along sequence length dim

    # k, v shape after: [batch, num_kv_heads, seq_len_total, head_dim]
    # (where seq_len_total = seq_len_cache + seq_len_curr)

# 4. Prepare cache for the *next* step (will be used as past_key_value for the next step)
current_key_value = (k, v,)

# ... rest of the code ...
```

### Notations

- `b` = batch size
- `N` (or `num_heads`) = number of attention heads
- `G` = number of key/value heads (not actual attention heads)
- `seq_len_cache` = sequence length of the cached states
- `seq_len_curr` = sequence length of the current state
- `d` = head dimension
- `offset` = the position of the current token in the sequence
- `t` = the current time step (or position in the sequence)
- `p` = the position of a token in the sequence


### Cache Breakdown:

A. `cache_k, cache_v = past_key_value if ...:`

- What it does: This line unpacks the `past_key_value` tuple. This tuple is an input to the `forward` function and contains the <b><a style="color: green;">Key</a></b> and <b><a style="color: yellow;">Value</a></b> tensors calculated and accumulated from all previous generation steps.
- When is `past_key_value` NOT `None`? During autoregressive generation, after the very first step (processing the initial prompt), the model will pass the calculated `k` and `v` from step `t` as the `past_key_value` for step `t+1`.
- When IS `past_key_value` `None`? Only during the very first pass, typically when processing the initial prompt. In this case, `cache_k` and `cache_v` are set to `None`, indicating there's no history yet.

B. `if cache_k is not None:`

- What it does: This condition checks if we are in a generation step after the initial prompt processing. If `cache_k` exists, it means we have a history of <b><a style="color: green;">keys</a></b> (and <b><a style="color: yellow;">values</a></b>) from previous tokens.
- Why check? We only need to concatenate if there's something to concatenate with.

C. `k = torch.cat([cache_k, k], dim=2) and v = torch.cat([cache_v, v], dim=2):`

- What it does: This is the core caching operation. It takes the tensors from the cache (`cache_k`, `cache_v`) and concatenates them with the current step's <b><a style="color: green;">key</a></b> and <b><a style="color: yellow;">value</a></b> tensors (`k`, `v`) along dim=2.
- Dimension 2: Remember the shapes: `[batch, num_kv_heads, seq_len, head_dim]`. Dimension 2 is the sequence length dimension. Concatenating here means appending the current token's <b><a style="color: green;">K</a></b> and <b><a style="color: yellow;">V</a></b> state(s) to the end of the sequence history.
- Inputs to `torch.cat`:
  - `cache_k/cache_v`: These contain the <b><a style="color: green;">K</a></b> and <b><a style="color: yellow;">V</a></b> states for tokens at positions 0 to offset - 1. Shape: `[b, G, seq_len_cache, d]`.
  - `k/v`: These are the <b><a style="color: green;">K</a></b> and <b><a style="color: yellow;">V</a></b>    states for the current token(s) we just processed (positions offset to offset + seq_len_curr - 1). Shape: `[b, G, seq_len_curr, d]`.
- Output: The `k` and `v` variables are updated in-place (conceptually) to hold the combined history + current states. Their sequence length dimension (`dim=2`) grows. Shape: `[b, G, seq_len_total, d]`.

D. `current_key_value = (k, v,)`

- What it does: This simply packages the potentially updated <b><a style="color: green;">k</a></b> and <b><a style="color: yellow;">v</a></b> tensors (which now contain the full sequence history up to the current step) into a tuple.
- Why? This tuple will be returned by the `forward` function and passed back in as the `past_key_value` argument for the next generation step. This is how the history is maintained across steps.


### Why Cache K and V for Past Tokens?

Let's consider generating token at position `t`. The attention mechanism needs the <b><a style="color: red;">Query</a></b> `Q_t` (based on the token at `t-1`) to compare against the <b><a style="color: green;">Keys</a></b> `K_0, K_1, ..., K_{t-1}, K_t` and aggregate the <b><a style="color: yellow;">Values</a></b> `V_0, V_1, ..., V_{t-1}, V_t`.
The key insight is: The calculated `K_p` and `V_p` for a token at a past position `p` (where `p < t`) will always be the same, no matter what token `t` we are currently generating. They only depend on the original token at position `p` and the fixed model weights (`W_k`, `W_v`, `RoPE` frequencies for position `p`).
Therefore, recalculating `K_0...K_{t-1}` and `V_0...V_{t-1}` at every step `t` is redundant and computationally expensive.

### What Does the Cache Store?

The cache stores the complete <b><a style="color: green;">K</a></b> and <b><a style="color: yellow;">V</a></b> tensors for all the tokens processed so far.
Let's look at the shape again: `[batch, num_kv_heads, seq_len, head_dim]`
This tensor holds `num_kv_heads` different <b><a style="color: green;">Key</a></b> (or <b><a style="color: yellow;">Value</a></b>) vectors, each of dimension `head_dim`, for each of the `seq_len` tokens in the sequence history.

### Why Concatenate Along dim=2 (i.e. the Sequence Length)?

_A detailed code visualization of the cache concatenation and why it is done along `dim=2` is available in the [kv_cache_practice.ipynb](./kv_cache_practice.ipynb) file. I am a learner who understands things clearly only when I see them in action, so this code was really helpful for me._

When we process the new token(s) at the current step (let's say `seq_len_curr` new tokens starting at `offset`), we calculate their <b><a style="color: green;">K</a></b> and <b><a style="color: yellow;">V</a></b> vectors (shape `[b, G, seq_len_curr, d]`).
The cache `(cache_k, cache_v)` holds the <b><a style="color: green;">K</a></b>/<b><a style="color: yellow;">V</a></b> vectors for the previous `seq_len_cache` tokens (shape `[b, G, seq_len_cache, d]`).
We need to combine these so that the attention mechanism can access the <b><a style="color: green;">K</a></b>/<b><a style="color: yellow;">V</a></b> vectors for the entire sequence (past + current).
Think of the sequence length dimension (dim=2) as representing the timeline of tokens. We are adding the new tokens' <b><a style="color: green;">K</a></b>/<b><a style="color: yellow;">V</a></b> information to the end of the timeline history stored in the cache.
`torch.cat([cache_k, k], dim=2)` literally takes the block of historical <b><a style="color: green;">K</a></b>/<b><a style="color: yellow;">V</a></b> vectors and appends the block of current <b><a style="color: green;">K</a></b>/<b><a style="color: yellow;">V</a></b> vectors along the dimension that represents the sequence of tokens.

<b><a style="color: skyblue;">Why not other dimensions?</a></b>
- `dim=1` (Heads): Concatenating here would mean we are adding more <b><a style="color: green;">K</a></b>/<b><a style="color: yellow;">V</a></b> heads, which isn't happening. The number of heads is fixed.
- `dim=3` (Head Dim/Features): Concatenating here would mean we are making the feature vector for each token longer. This doesn't make sense; the <b><a style="color: green;">K</a></b>/<b><a style="color: yellow;">V</a></b> vectors for past tokens are already computed and have the correct head_dim. Caching saves recalculating those vectors, not changing their dimensionality.

#

## Understanding `p` (Position Index) and `t` (Time Step)

### `p` (Position Index)

`p` represents the location of a token within the sequence. Since models like Llama 2 are built with a maximum context window or sequence length (`max_seq_len`), the position index `p` is effectively limited by this value (specifically, `0 <= p < max_seq_len`). If I'd tried to use a `p` equal to or greater than `max_seq_len`, components like the RoPE cache lookup would fail because the necessary rotation values haven't been precomputed for that position. So, `p` is constrained by the model's architectural limit (`max_seq_len`).

### `t` (Time Step)

`t` functions as a counter for the discrete steps or actions taken during the autoregressive generation process. Each step `t` involves taking the output from step `t-1`, feeding it back as input, performing a forward pass (calculating <b><a style="color: red;">Q</a></b>, <b><a style="color: green;">K</a></b>, <b><a style="color: yellow;">V</a></b>, attending, predicting), and generating the next token. It tracks the progression of the generation loop itself.

In essence:
- p tells me <b><i>WHERE</i></b> a token is (its address in the sequence, limited by `max_seq_len`).
- t tells me <b><i>WHEN</i></b> a token was generated or processed (the step number in the generation algorithm).


This distinction is why `K_p` and `V_p` make sense (properties tied to a fixed location `p`) and `Q_t` makes sense (the query formulated at a specific step `t` in the generation process).


### Why is the <b><a style="color: red;">Query</a></b> tied to `t`?

- The <b><a style="color: red;">Query</a></b>'s job is to ask: "Based on everything I've seen up to now (time `t-1`), what should I pay attention to in order to predict the next token (the one for position `p=t`)?"
- The "up to now" perspective is defined by the last token processed, which is the one at position `p = t-1`.
- The <b><a style="color: red;">Query</a></b> vector is calculated based on the hidden state resulting from processing the token at `p = t-1`. It's inherently dynamic because the perspective (the last token seen) changes at each time step.
- Therefore, `Q_t` represents the query formulated during time step `t` based on the state after processing token `t-1`.


### Why are the <b><a style="color: green;">Key</a></b> and <b><a style="color: yellow;">Value</a></b> vectors tied to `p`?

- The <b><a style="color: green;">Key</a></b> and <b><a style="color: yellow;">Value</a></b> vectors represent the properties (K) and content (V) of the token located specifically at position `p`.
- Once the token at position `p` is processed, its corresponding <b><a style="color: green;">Key</a></b> vector `K_p` (content via `W_k` + position via `RoPE(offset=p)`) and <b><a style="color: yellow;">Value</a></b> vector `V_p` (content via `W_v`) are fixed. They don't change just because we are later generating a token at time step `t=p+10`.
- `K_p` answers the question: "What are the queryable properties of the token at position `p`?"
- `V_p` answers the question: "What information does the token at position `p` hold?"
- These properties are tied to the location (`p`) within the sequence, not the time (`t`) they are being accessed.

#

