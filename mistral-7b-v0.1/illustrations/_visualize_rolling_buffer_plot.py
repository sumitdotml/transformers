"""Visualizations generated with the help of Claude Sonnet 3.7."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from cache import RollingBufferCache
import os
import matplotlib.colors as mcolors

# Set seed for reproducibility
torch.manual_seed(0)

# Configuration
batch_size = 1
num_heads = 1
head_dim = 6
buffer_size = 3  # Smaller window to make visualization easier

# Create figure for visualizations
plt.style.use("seaborn-v0_8-whitegrid")
fig = plt.figure(figsize=(16, 22))  # Increased figure size for better spacing

# Set up the cache
cache = RollingBufferCache(buffer_size=buffer_size, kv_dim=head_dim)

# Tracking variables to store states for visualization
physical_states = []
window_states = []
token_labels = []
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
physical_states.append(cache.k_cache[0, 0, :, 0].clone().numpy())
window_states.append(k_window1[0, 0, :, 0].clone().numpy())
token_labels.append([1, 2, 3])
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
physical_states.append(cache.k_cache[0, 0, :, 0].clone().numpy())
window_states.append(k_window2[0, 0, :, 0].clone().numpy())
token_labels.append([4, 5, 3])  # Physical layout in cache
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
physical_states.append(cache.k_cache[0, 0, :, 0].clone().numpy())
window_states.append(k_window3[0, 0, :, 0].clone().numpy())
token_labels.append([4, 5, 6])  # Physical layout
current_seq_lens.append(current_seq_len_3)

# ====== VISUALIZATION ======

# Color mapping for tokens - using more distinct colors with better contrast
token_colors = {
    1: "#FF6B6B",  # Brighter red
    2: "#4ECDC4",  # Teal
    3: "#7986CB",  # Blue-purple
    4: "#FFD166",  # Golden yellow
    5: "#C77DFF",  # Bright purple
    6: "#56CFE1",  # Cyan
}


# Function to visualize buffer state
def visualize_pass(ax, pass_idx, title):
    physical_state = physical_states[pass_idx]
    window_state = window_states[pass_idx]
    token_label = token_labels[pass_idx]
    total_tokens = current_seq_lens[pass_idx]

    # Clear previous plots
    ax.clear()

    # Set up the plot with more space
    ax.set_xlim(-1.5, 9.5)
    ax.set_ylim(-5.5, 4)  # Increased bottom margin for timeline
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)

    # Draw section labels
    ax.text(
        -1,
        1.5,
        "Physical\nBuffer",
        fontsize=14,
        fontweight="bold",
        ha="left",
        va="center",
    )
    ax.text(
        -1,
        -3,
        "Logical\nWindow",
        fontsize=14,
        fontweight="bold",
        ha="left",
        va="center",
    )

    # Draw the circular buffer with slightly larger radius
    circle = plt.Circle(
        (4, 1), 2.8, fill=False, edgecolor="black", linestyle="--", linewidth=2
    )
    ax.add_patch(circle)

    # Add a light background to the circular buffer area
    circle_bg = plt.Circle((4, 1), 2.8, fill=True, color="#F5F5F5", alpha=0.3, zorder=0)
    ax.add_patch(circle_bg)

    # Draw slots in the buffer
    angles = np.linspace(0, 2 * np.pi, buffer_size + 1)[:-1]

    # Draw token slots as boxes with larger size
    for i in range(buffer_size):
        angle = angles[i]
        x = 4 + 2.0 * np.cos(angle)
        y = 1 + 2.0 * np.sin(angle)

        # Draw a rectangle for the slot
        token_id = token_label[i]
        color = token_colors[token_id]
        rect = Rectangle(
            (x - 0.8, y - 0.8),
            1.6,
            1.6,
            facecolor=color,
            alpha=0.8,
            edgecolor="black",
            linewidth=2,
            zorder=2,
        )
        ax.add_patch(rect)

        # Add slot number and token ID with increased font size
        ax.text(
            x, y + 0.3, f"Slot {i}", ha="center", va="center", fontsize=12, zorder=3
        )
        ax.text(
            x,
            y - 0.1,
            f"Token {token_id}",
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            zorder=3,
        )

        # Add value
        value = physical_state[i]
        ax.text(
            x, y - 0.5, f"{value:.1f}", ha="center", va="center", fontsize=11, zorder=3
        )

    # Draw a light background behind the window area
    window_bg = Rectangle((1.5, -4.5), 5, 3, facecolor="#ECF6FF", alpha=0.3, zorder=0)
    ax.add_patch(window_bg)

    # Draw window (chronological tokens) with more spacing
    window_width = min(buffer_size, total_tokens)
    window_spacing = 2.0  # Increased spacing between window tokens

    for i in range(window_width):
        # Position window tokens in a row below with more spacing
        x = 4 - ((window_width - 1) / 2 - i) * window_spacing
        y = -3.0  # Lower position for better separation

        # Determine token position
        pos = total_tokens - window_width + i
        token_id = pos + 1

        # Find which physical slot this token is in
        physical_slot = token_label.index(token_id) if token_id in token_label else None
        color = token_colors[token_id]

        # Draw window token with larger size
        rect = Rectangle(
            (x - 0.8, y - 0.8),
            1.6,
            1.6,
            facecolor=color,
            alpha=0.8,
            edgecolor="black",
            linewidth=2,
            zorder=2,
        )
        ax.add_patch(rect)

        # Add position and token ID with increased font size
        ax.text(
            x, y + 0.3, f"Pos {pos}", ha="center", va="center", fontsize=12, zorder=3
        )
        ax.text(
            x,
            y - 0.1,
            f"Token {token_id}",
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            zorder=3,
        )

        # Add value
        value = window_state[i]
        ax.text(
            x, y - 0.5, f"{value:.1f}", ha="center", va="center", fontsize=11, zorder=3
        )

        # Draw an arrow connecting physical slot to window position if token is in cache
        if physical_slot is not None:
            slot_angle = angles[physical_slot]
            slot_x = 4 + 2.0 * np.cos(slot_angle)
            slot_y = 1 + 2.0 * np.sin(slot_angle)

            # Create a slightly curved arrow with better styling
            # Use different colors for different tokens to make it easier to follow
            arrow = FancyArrowPatch(
                (slot_x, slot_y - 0.9),
                (x, y + 0.9),
                connectionstyle="arc3,rad=-0.2",
                arrowstyle="-|>",
                color=mcolors.to_rgba(
                    color, alpha=0.7
                ),  # Use token color for arrow with transparency
                linewidth=2.0,
                zorder=1,
            )
            ax.add_patch(arrow)

    # Add timeline showing current sequence position - more visually distinct and lower position
    timeline_y = -4.8  # Move timeline down to avoid overlap
    ax.axhline(
        y=timeline_y,
        xmin=0.05,
        xmax=0.95,
        color="black",
        linestyle="-",
        linewidth=2.5,
        zorder=1,
    )

    # Add buffer size label above timeline and to the left
    ax.text(
        1.0,
        -4.4,
        f"Buffer Size: {buffer_size}",
        ha="left",
        va="center",
        fontsize=12,
        fontweight="bold",
        zorder=2,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2),
    )

    # Add position markers with better visibility and more space between labels
    spacing = 7.0 / (total_tokens + 2)  # More space between timeline markers
    for i in range(total_tokens + 1):
        x_pos = 1.0 + (i + 0.5) * spacing  # Adjust position for better spacing
        if i < total_tokens:  # Position markers
            ax.plot([x_pos], [timeline_y], "o", color="black", markersize=8, zorder=2)
            # Position the numbers below the line with more space
            ax.text(
                x_pos,
                timeline_y - 0.4,
                f"{i}",
                ha="center",
                va="center",
                fontsize=11,
                zorder=2,
            )

    # Add current position indicator - more prominent
    current_x = 1.0 + (total_tokens + 0.5) * spacing
    ax.plot([current_x], [timeline_y], "o", color="red", markersize=12, zorder=3)
    ax.text(
        current_x,
        timeline_y - 0.4,
        f"{total_tokens}",
        ha="center",
        va="center",
        fontsize=11,
        color="red",
        fontweight="bold",
        zorder=3,
    )

    # Move "Current" label to the right with more space
    ax.text(
        current_x + 0.7,
        timeline_y - 0.4,
        "Current",
        ha="left",
        va="center",
        fontsize=11,
        color="red",
        fontweight="bold",
        zorder=3,
    )


# Create subplots for each pass with more spacing
axes = [plt.subplot(3, 1, i + 1) for i in range(3)]

# Visualize each pass
visualize_pass(axes[0], 0, "Pass 1: Initial Fill (Tokens 1-3)")
visualize_pass(axes[1], 1, "Pass 2: First Rolling (Tokens 1-2 Overwritten)")
visualize_pass(axes[2], 2, "Pass 3: Continued Rolling (Token 3 Overwritten)")

# Add overall title
plt.suptitle("Rolling Buffer Cache Visualization", fontsize=20, y=0.98)

# Add explanation text with better formatting
explanation = (
    "The Rolling Buffer Cache maintains a fixed-size circular buffer of the most recent tokens.\n"
    "As new tokens are added, they overwrite the oldest ones in the buffer.\n"
    "When tokens are retrieved from the cache, they are returned in chronological order,\n"
    "regardless of their physical location in the buffer."
)

# Adjust layout first
plt.tight_layout(
    rect=[0, 0.14, 1, 0.95], h_pad=3.0
)  # Add more vertical space between subplots

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
    os.path.dirname(os.path.abspath(__file__)), "rolling_buffer_visualization.png"
)
plt.savefig(output_file, dpi=200)  # Higher DPI for better quality
print(f"Visualization saved as '{output_file}'")
print("\nVisualization created successfully!")
