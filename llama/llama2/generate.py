import torch
from model import Llama2
from config import CONFIG
from transformers import AutoTokenizer

# 1. initializing tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 2. getting vocab size from tokenizer
effective_vocab_size = tokenizer.vocab_size

# 3. initializing model using the tokenizer's vocab size
model = Llama2(
    vocab_size=effective_vocab_size,
    hidden_dim=CONFIG["hidden_size"],
    num_decoder_layers=CONFIG["num_hidden_layers"],
    num_heads=CONFIG["num_attention_heads"],
    num_kv_heads=CONFIG["num_key_value_heads"],
    ffn_dim=CONFIG["intermediate_size"],
    max_seq_len=CONFIG["max_position_embeddings"],
    rope_base=CONFIG["rope_theta"],
    norm_eps=CONFIG["rms_norm_eps"],
)

# 4. loading pre-trained weights if available
# model.load_state_dict(torch.load("path_to_weights.pt"))


def generate_text(prompt, max_new_tokens=20, temperature=0.8, top_p=0.9):
    # 1. tokenizing prompt
    print(f"Tokenizing prompt: '{prompt}'")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    print(f"Prompt token IDs: {input_ids.tolist()}")

    # 2. phase 1: processing prompt (no past_key_values)
    print("Phase 1: Processing prompt...")
    with torch.no_grad():
        # 1. first forward pass processes the whole prompt
        logits, past_key_values = model(input_ids=input_ids)
        print("Prompt processed, initial KV cache generated.")

        # 2. getting the last token's logits to predict the next token
        next_token_logits = logits[:, -1, :]

        # 3. sampling next token
        probs = torch.softmax(next_token_logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # 4. initializing generated sequence with the next token
        generated_ids = [next_token.item()]
        print(f"First generated token ID: {next_token.item()}")

        # 5. generating tokens one by one using past_key_values cache
        print(f"Phase 2: Generating up to {max_new_tokens -1} more tokens...")
        for i in range(max_new_tokens - 1):
            print(f"--- Generating token {i+1}/{max_new_tokens -1} ---")
            # 1. creating attention mask for the new token (always visible)
            # we don't need complex attention mask for generation since we only process one token

            # 2. forward pass with single token and cached KV
            print(f"   Input token ID: {next_token.item()}")
            print(f"   Offset: {input_ids.shape[1] + len(generated_ids) - 1}")
            logits, past_key_values = model(
                input_ids=next_token,  # Only the most recent token
                offset=input_ids.shape[1] + len(generated_ids) - 1,  # Position tracking
                past_key_values=past_key_values,  # Using the cache
            )
            print(f"   Model forward pass complete for this token.")

            # 3. sampling next token
            next_token_logits = logits[:, -1, :]
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            print(f"   Sampled next token ID: {next_token.item()}")

            # 4. adding to generated sequence
            generated_ids.append(next_token.item())

            # 5. stopping if EOS token
            if next_token.item() == tokenizer.eos_token_id:
                print(
                    f"   EOS token ({tokenizer.eos_token_id}) detected. Stopping generation."
                )
                break
        print(f"Token generation loop finished.")

    # 6. decoding and returning the generated text
    print(f"Generated token IDs: {generated_ids}")
    output = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"Decoded output: '{output}'")
    return prompt + output


# 7. testing generation
prompt = "Once upon a time in a land far away,"
generated_text = generate_text(prompt)
print(generated_text)
