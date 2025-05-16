"""
Simple Rolling Buffer Cache Exercise for Intuition
"""

prompt = "The quick brown fox jumps over"

tokenizer_dict = {
    "The": 1,
    "quick": 2,
    "brown": 3,
    "fox": 4,
    "jumps": 5,
    "over": 6,
}

tokens = [tokenizer_dict[token] for token in prompt.split()]


def cache_initial_tokens(window_size: int = 4):
    """
    In rolling buffer cache, the generating token should attend to itself and the previous (window_size - 1) tokens.
    """
    token_index_position = 0
    initial_cache = [None] * window_size

    # during prompt processing, the cache should be filled with the initial (window_size - 1) tokens
    while token_index_position < len(tokens):
        cache_index = token_index_position % window_size
        initial_cache[cache_index] = tokens[token_index_position]
        token_index_position += 1

    return initial_cache


new_generated_tokens_one_at_a_time = [7, 8]


def generate_tokens(window_size: int = 4):
    initial_cache = cache_initial_tokens(window_size)
    token_index_position = len(tokens)
    while token_index_position < len(tokens) + len(new_generated_tokens_one_at_a_time):
        cache_index = token_index_position % window_size
        initial_cache[cache_index] = new_generated_tokens_one_at_a_time[
            token_index_position - len(tokens)
        ]
        token_index_position += 1
    return initial_cache


if __name__ == "__main__":
    print(tokens)
    initial_cache = cache_initial_tokens()
    print(initial_cache)
    print(generate_tokens())
