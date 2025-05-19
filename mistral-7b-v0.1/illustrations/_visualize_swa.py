"""Visualizations generated with the help of Claude Sonnet 3.7."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path so we can import from there
sys.path.append("..")
from mask import SlidingWindowMask

print("\n=== SLIDING WINDOW ATTENTION VISUALIZATION ===\n")

# Configuration
window_size = 4  # As described in the example
sequence_length = 8  # Total sequence length to visualize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the mask generator
mask_generator = SlidingWindowMask(sliding_window=window_size)

# Create a visual representation of the attention pattern
fig, axes = plt.subplots(2, 1, figsize=(10, 10))
fig.suptitle("Sliding Window Attention", fontsize=18, y=0.98)

# 1. Visualize full attention (for comparison)
full_attention = torch.zeros((sequence_length, sequence_length))
for i in range(sequence_length):
    for j in range(i + 1):  # Lower triangular (causal)
        full_attention[i, j] = 1

# Show full attention pattern - more minimalistic with no markers
im = axes[0].imshow(full_attention, cmap="Blues", vmin=0, vmax=1)
axes[0].set_title(f"Full Attention (Sequence Length = {sequence_length})", fontsize=14)
axes[0].set_xlabel("Key Position")
axes[0].set_ylabel("Query Position")

# Add grid lines - much more subtle
axes[0].set_xticks(np.arange(-0.5, sequence_length, 1), minor=True)
axes[0].set_yticks(np.arange(-0.5, sequence_length, 1), minor=True)
axes[0].grid(which="minor", color="#E5E5E5", linestyle="-", linewidth=0.3, alpha=0.2)

# Add tick marks
axes[0].set_xticks(range(sequence_length))
axes[0].set_yticks(range(sequence_length))
axes[0].set_xticklabels([f"{j+1}" for j in range(sequence_length)])
axes[0].set_yticklabels([f"{i+1}" for i in range(sequence_length)])

# 2. Visualize sliding window attention
# We'll generate the attention mask for each position and combine them
sliding_window_attention = torch.zeros((sequence_length, sequence_length))

# Generate masks for each position and extract the pattern
for pos in range(sequence_length):
    # For each position, create a mask with a single query token
    mask = mask_generator.get_mask(
        batch_size=1,
        q_len=1,
        kv_len=pos + 1,  # Keys up to current position
        offset=pos,  # Current absolute position
        device=device,
    )

    # Extract the mask values (0 means attend, -inf means don't attend)
    mask_values = mask[0, 0, 0, :].cpu()

    # Mark positions that can be attended to
    for j in range(pos + 1):
        # If mask value is not -inf, this position is attended to
        if mask_values[j] > -1e6:  # Not -inf
            sliding_window_attention[pos, j] = 1

# Create a visualization of the sliding window attention - minimalistic
im = axes[1].imshow(sliding_window_attention, cmap="Blues", vmin=0, vmax=1)
axes[1].set_title(
    f"Sliding Window Attention (Window Size = {window_size})", fontsize=14
)
axes[1].set_xlabel("Key Position")
axes[1].set_ylabel("Query Position")

# Add grid lines - much more subtle
axes[1].set_xticks(np.arange(-0.5, sequence_length, 1), minor=True)
axes[1].set_yticks(np.arange(-0.5, sequence_length, 1), minor=True)
axes[1].grid(which="minor", color="#E5E5E5", linestyle="-", linewidth=0.3, alpha=0.2)

# Add tick marks
axes[1].set_xticks(range(sequence_length))
axes[1].set_yticks(range(sequence_length))
axes[1].set_xticklabels([f"{j+1}" for j in range(sequence_length)])
axes[1].set_yticklabels([f"{i+1}" for i in range(sequence_length)])

# Add explanation - moved lower to avoid overlap
explanation = (
    "Sliding Window Attention (SWA) limits each token to attend only to a fixed window of previous tokens.\n"
    f"In this example, each token can attend to itself and up to {window_size-1} previous tokens.\n"
    "This reduces computational complexity from O(n²) to O(n·w), enabling efficient processing of long sequences."
)

# Adjust layout first
plt.tight_layout(rect=[0, 0.14, 1, 0.96])

# Add text with more space at the bottom
plt.figtext(
    0.5,
    0.02,
    explanation,
    ha="center",
    fontsize=13,
    bbox=dict(
        boxstyle="round", facecolor="white", alpha=0.8, edgecolor="lightgray", pad=1.0
    ),
)

# Save the visualization
output_file = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "sliding_window_attention.png"
)
plt.savefig(output_file, dpi=200, bbox_inches="tight")
print(f"Visualization saved as '{output_file}'")

# Print pattern visualization with filled and empty circles
print("\n=== ATTENTION PATTERNS ===")
print("\nFull Attention:")
for i in range(sequence_length):
    pattern = ""
    for j in range(sequence_length):
        if full_attention[i, j] == 1:
            pattern += "● "
        else:
            pattern += "○ "
    print(f"Token {i+1}: {pattern}")

print("\nSliding Window Attention (W=4):")
for i in range(sequence_length):
    pattern = ""
    for j in range(sequence_length):
        if sliding_window_attention[i, j] == 1:
            pattern += "● "
        else:
            pattern += "○ "
    print(f"Token {i+1}: {pattern}")

# Add detailed testing section
print("\n=== DETAILED MASK TESTING ===")

# Test cases illustrating the sliding window effect
test_cases = [
    {"name": "Token 1", "q_len": 1, "kv_len": 1, "offset": 0},
    {"name": "Token 2", "q_len": 1, "kv_len": 2, "offset": 1},
    {"name": "Token 3", "q_len": 1, "kv_len": 3, "offset": 2},
    {"name": "Token 4", "q_len": 1, "kv_len": 4, "offset": 3},
    {"name": "Token 5", "q_len": 1, "kv_len": 5, "offset": 4},
    {"name": "Token 6", "q_len": 1, "kv_len": 6, "offset": 5},
    {"name": "Token 7", "q_len": 1, "kv_len": 7, "offset": 6},
    {"name": "Token 8", "q_len": 1, "kv_len": 8, "offset": 7},
]

for tc in test_cases:
    mask = mask_generator.get_mask(
        batch_size=1,
        q_len=tc["q_len"],
        kv_len=tc["kv_len"],
        offset=tc["offset"],
        device=device,
    )

    # Convert mask values to a readable format
    mask_values = mask[0, 0, 0].cpu().numpy()
    mask_str = ""
    for j in range(tc["kv_len"]):
        if mask_values[j] > -1e6:  # Not -inf
            mask_str += f"Token {j+1} "

    print(f"{tc['name']} can attend to: {mask_str}")

print("\nVerification:")
print(f"- Window size (W) = {window_size}")
print("- Each token should be able to attend to itself and up to (W-1) previous tokens")
print("- Tokens beyond position W should show a sliding window pattern")
