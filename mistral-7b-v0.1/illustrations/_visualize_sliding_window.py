"""Visualizations generated with the help of Claude Sonnet 3.7."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path so we can import from there
sys.path.append("..")
from cache import RollingBufferCache

# Set seed for reproducibility
torch.manual_seed(0)

# Configuration
batch_size = 1
num_heads = 1
head_dim = 6
buffer_size = 3  # Smaller window to make visualization easier

# Create figure for visualizations
plt.style.use("seaborn-v0_8-whitegrid")

# Set up the cache
cache = RollingBufferCache(buffer_size=buffer_size, kv_dim=head_dim)

# Tracking variables to store states for visualization
window_states = []
current_seq_lens = []

# FIRST PASS - 3 tokens (exactly fills the buffer)
seq_len_1 = 3
current_seq_len_1 = seq_len_1

# Generate dummy K/V tensors for first pass
k1 = torch.zeros(batch_size, num_heads, seq_len_1, head_dim)
v1 = torch.zeros(batch_size, num_heads, seq_len_1, head_dim)

# Set values to make them recognizable
for i in range(seq_len_1):
    k1[0, 0, i] = torch.tensor([i + 1, i + 1, i + 1, i + 1, i + 1, i + 1]) * 0.1
    v1[0, 0, i] = torch.tensor([i + 1, i + 1, i + 1, i + 1, i + 1, i + 1]) * -0.1

# Update cache
k_window1, v_window1 = cache.update(k1, v1, current_seq_len_1)

# Store state for visualization
window_states.append(k_window1[0, 0, :, 0].clone().numpy())
current_seq_lens.append(current_seq_len_1)

# SECOND PASS - 2 more tokens
seq_len_2 = 2
current_seq_len_2 = current_seq_len_1 + seq_len_2

# Generate dummy K/V tensors for second pass
k2 = torch.zeros(batch_size, num_heads, seq_len_2, head_dim)
v2 = torch.zeros(batch_size, num_heads, seq_len_2, head_dim)

# Set values
for i in range(seq_len_2):
    pos = seq_len_1 + i  # 3, 4
    k2[0, 0, i] = (
        torch.tensor([pos + 1, pos + 1, pos + 1, pos + 1, pos + 1, pos + 1]) * 0.1
    )
    v2[0, 0, i] = (
        torch.tensor([pos + 1, pos + 1, pos + 1, pos + 1, pos + 1, pos + 1]) * -0.1
    )

# Update cache
k_window2, v_window2 = cache.update(k2, v2, current_seq_len_2)

# Store state for visualization
window_states.append(k_window2[0, 0, :, 0].clone().numpy())
current_seq_lens.append(current_seq_len_2)

# THIRD PASS - 1 more token
seq_len_3 = 1
current_seq_len_3 = current_seq_len_2 + seq_len_3

# Generate new K/V tensor for the next token
k3 = torch.zeros(batch_size, num_heads, seq_len_3, head_dim)
v3 = torch.zeros(batch_size, num_heads, seq_len_3, head_dim)

# Set values
pos = seq_len_1 + seq_len_2  # 5
k3[0, 0, 0] = torch.tensor([pos + 1, pos + 1, pos + 1, pos + 1, pos + 1, pos + 1]) * 0.1
v3[0, 0, 0] = (
    torch.tensor([pos + 1, pos + 1, pos + 1, pos + 1, pos + 1, pos + 1]) * -0.1
)

# Update cache
k_window3, v_window3 = cache.update(k3, v3, current_seq_len_3)

# Store state for visualization
window_states.append(k_window3[0, 0, :, 0].clone().numpy())
current_seq_lens.append(current_seq_len_3)

# Show logical visualization of what the attention layer would see
fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # Slightly larger figure
fig.suptitle("Sliding Window Evolution (Logical View)", fontsize=18)

for i, (pass_idx, ax) in enumerate(zip(range(3), axes)):
    tokens_processed = current_seq_lens[pass_idx]
    window = window_states[pass_idx]
    window_size = len(window)

    # Create a colorful representation of the sliding window
    # First, create the full sequence for context
    full_sequence = np.zeros(tokens_processed)

    # Fill in values
    for j in range(min(tokens_processed, buffer_size)):
        pos = tokens_processed - window_size + j
        if pos < tokens_processed:
            full_sequence[pos] = j + 1

    # Create a grid for visualization
    data = np.zeros((2, tokens_processed))
    mask = np.zeros((2, tokens_processed), dtype=bool)

    # First row: all tokens in sequence
    for j in range(tokens_processed):
        data[0, j] = j + 1

    # Second row: tokens in window
    for j in range(tokens_processed):
        if tokens_processed - j <= window_size and j < tokens_processed:
            # This token is in the window
            data[1, j] = j + 1
        else:
            # This token is not in the window (mask it)
            mask[1, j] = True

    # Create a colorful matrix plot with better colors
    cmap = plt.cm.viridis
    cmap.set_bad("lightgray")

    # Apply mask
    masked_data = np.ma.array(data, mask=mask)
    im = ax.matshow(masked_data, cmap=cmap, aspect="auto")

    # Set row labels with larger font
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["All Tokens", "Window Tokens"], fontsize=13)

    # Set column labels (token positions) with larger font
    ax.set_xticks(range(tokens_processed))
    ax.set_xticklabels([f"Token {j+1}" for j in range(tokens_processed)], fontsize=12)

    # Rotate labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add info about window position
    window_start = max(0, tokens_processed - window_size)
    window_end = tokens_processed - 1
    title = f"Pass {i+1}: Window [{window_start+1}-{window_end+1}]"
    ax.set_title(title, fontsize=15, pad=10)

    # Add grid
    ax.set_xticks(np.arange(-0.5, tokens_processed), minor=True)
    ax.set_yticks(np.arange(-0.5, 2), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=1)

    # Add tokens to cells with larger font
    for row in range(2):
        for col in range(tokens_processed):
            if not mask[row, col]:
                ax.text(
                    col,
                    row,
                    f"{data[row, col]:.0f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontweight="bold",
                    fontsize=14,
                )

    # Add explanation to each subplot
    if i == 0:
        ax.text(
            -0.1,
            -0.2,
            "Initial buffer fill",
            transform=ax.transAxes,
            fontsize=12,
            ha="left",
            va="top",
        )
    elif i == 1:
        ax.text(
            -0.1,
            -0.2,
            "Window slides as new tokens arrive",
            transform=ax.transAxes,
            fontsize=12,
            ha="left",
            va="top",
        )
    else:
        ax.text(
            -0.1,
            -0.2,
            "Oldest tokens fall out of window",
            transform=ax.transAxes,
            fontsize=12,
            ha="left",
            va="top",
        )

# Add explanation text with better formatting
explanation = (
    "The sliding window attention mechanism allows each token to attend only to a fixed-size\n"
    "window of the most recent tokens. This provides O(N·W) complexity instead of O(N²),\n"
    "making it more efficient for processing long sequences."
)

# Adjust layout first
plt.tight_layout(rect=[0, 0.14, 1, 0.95])

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
    os.path.dirname(os.path.abspath(__file__)), "sliding_window_evolution.png"
)
plt.savefig(output_file, dpi=200)
print(f"Visualization saved as '{output_file}'")
print("\nVisualization created successfully!")
